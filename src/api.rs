//! Local HTTP API on 127.0.0.1 so other programs (editors, agents, scripts)
//! can transcribe files with the same model the tray app uses.
//!
//! Routes:
//!   GET  /health
//!   GET  /v1/models
//!   POST /v1/audio/transcriptions   OpenAI-compatible: multipart `file` part,
//!                                   or JSON `{"path": "<local file>"}`,
//!                                   or a raw audio body.
//!
//! Audio and video containers are decoded in-process with symphonia (wav, mp3,
//! flac, ogg/vorbis, m4a/mp4/mov AAC and ALAC, mkv); anything it cannot read
//! falls back to `ffmpeg` if one is on PATH. Long inputs are cut at pauses into
//! chunks of at most `API_MAX_CHUNK_SECS` and sent through the same inference
//! worker as dictation, so both share one loaded model and dictation segments
//! interleave with API chunks instead of waiting for a whole file.

use crate::app::AppEvent;
use crate::audio::segmenter::{segment_all, SegmenterConfig};
use crate::audio::TARGET_SAMPLE_RATE;
use crate::config::Config;
use crossbeam_channel::{bounded, Sender};
use serde_json::{json, Value};
use std::fs;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use tiny_http::{Header, Method, Request, Response, Server};
use tracing::{debug, info, warn};

/// Largest request body accepted (uploads; local paths are streamed from disk).
const MAX_BODY_BYTES: usize = 2 * 1024 * 1024 * 1024;
/// How long one chunk may sit in the inference queue plus run.
const CHUNK_TIMEOUT: Duration = Duration::from_secs(900);
/// Chunking for files: longer than dictation segments (fewer prefills), short
/// enough that a dictation segment queued behind one is not delayed much.
const API_MIN_CHUNK_SECS: f32 = 30.0;
const API_MAX_CHUNK_SECS: f32 = 90.0;
const API_PAUSE_SECS: f32 = 0.5;

type Resp = Response<Cursor<Vec<u8>>>;

pub struct ApiServer {
    shutdown: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

struct Ctx {
    event_tx: Sender<AppEvent>,
    token: Option<String>,
    model: String,
    use_gpu: bool,
    temp_dir: PathBuf,
    counter: AtomicU64,
}

impl ApiServer {
    /// Bind 127.0.0.1:`api_port` and serve on a background thread.
    pub fn start(config: &Config, event_tx: Sender<AppEvent>) -> Result<Self, String> {
        let addr = format!("127.0.0.1:{}", config.api_port);
        let server = Server::http(&addr).map_err(|e| format!("cannot listen on {}: {}", addr, e))?;
        let temp_dir = crate::config::project_dirs()
            .map(|d| d.cache_dir().join("api"))
            .unwrap_or_else(std::env::temp_dir);
        let _ = fs::create_dir_all(&temp_dir);
        let ctx = Arc::new(Ctx {
            event_tx,
            token: config.api_token.clone().filter(|t| !t.trim().is_empty()),
            model: config.model.clone(),
            use_gpu: config.use_gpu,
            temp_dir,
            counter: AtomicU64::new(0),
        });
        let shutdown = Arc::new(AtomicBool::new(false));
        let flag = shutdown.clone();
        let handle = thread::Builder::new()
            .name("api-server".to_string())
            .spawn(move || {
                while !flag.load(Ordering::Relaxed) {
                    match server.recv_timeout(Duration::from_millis(250)) {
                        Ok(Some(request)) => {
                            let ctx = ctx.clone();
                            let _ = thread::Builder::new()
                                .name("api-request".to_string())
                                .spawn(move || handle_request(request, &ctx));
                        }
                        Ok(None) => {}
                        Err(e) => {
                            warn!("API accept error: {}", e);
                            thread::sleep(Duration::from_millis(100));
                        }
                    }
                }
                info!("API server stopped");
            })
            .map_err(|e| e.to_string())?;
        info!("API listening on http://{} (POST /v1/audio/transcriptions)", addr);
        Ok(Self { shutdown, handle: Some(handle) })
    }

    pub fn stop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for ApiServer {
    fn drop(&mut self) {
        self.stop();
    }
}

// ---------------------------------------------------------------------------
// Request handling
// ---------------------------------------------------------------------------

fn handle_request(mut request: Request, ctx: &Ctx) {
    let started = Instant::now();
    let method = request.method().clone();
    let url = request.url().to_string();
    let (path, query) = match url.split_once('?') {
        Some((p, q)) => (p.to_string(), q.to_string()),
        None => (url.clone(), String::new()),
    };

    let result: Result<Resp, (u16, String)> = match (&method, path.as_str()) {
        (Method::Get, "/") | (Method::Get, "/health") => Ok(json_response(
            200,
            json!({
                "status": "ok",
                "version": env!("CARGO_PKG_VERSION"),
                "model": ctx.model,
                "use_gpu": ctx.use_gpu,
                "endpoints": ["POST /v1/audio/transcriptions", "GET /v1/models", "GET /health"],
            }),
        )),
        (Method::Get, "/v1/models") => Ok(json_response(
            200,
            json!({
                "object": "list",
                "data": [
                    {"id": ctx.model, "object": "model", "owned_by": "voclaude", "active": true},
                ]
            }),
        )),
        (Method::Post, "/v1/audio/transcriptions") | (Method::Post, "/transcribe") => {
            if !authorized(&request, ctx) {
                Err((401, "missing or invalid bearer token".to_string()))
            } else {
                transcribe_request(&mut request, &query, ctx)
            }
        }
        _ => Err((404, format!("no route {} {}", method, path))),
    };

    let (code, response) = match result {
        Ok(r) => (200u16, r),
        Err((code, msg)) => (
            code,
            json_response(code, json!({"error": {"message": msg, "type": "invalid_request_error"}})),
        ),
    };
    debug!("API {} {} -> {} in {:.1?}", method, path, code, started.elapsed());
    if let Err(e) = request.respond(response) {
        debug!("API client went away: {}", e);
    }
}

fn authorized(request: &Request, ctx: &Ctx) -> bool {
    match &ctx.token {
        None => true,
        Some(token) => header(request, "Authorization")
            .and_then(|v| v.strip_prefix("Bearer ").map(str::trim).map(String::from))
            .map(|v| v == *token)
            .unwrap_or(false),
    }
}

fn header(request: &Request, name: &str) -> Option<String> {
    request
        .headers()
        .iter()
        .find(|h| h.field.as_str().as_str().eq_ignore_ascii_case(name))
        .map(|h| h.value.as_str().to_string())
}

fn json_response(code: u16, value: Value) -> Resp {
    Response::from_string(value.to_string())
        .with_status_code(code)
        .with_header(Header::from_bytes("Content-Type", "application/json; charset=utf-8").unwrap())
}

fn text_response(text: &str) -> Resp {
    Response::from_string(text)
        .with_status_code(200)
        .with_header(Header::from_bytes("Content-Type", "text/plain; charset=utf-8").unwrap())
}

struct Upload {
    name: String,
    data: Vec<u8>,
}

#[derive(Default)]
struct Input {
    file: Option<Upload>,
    path: Option<String>,
    response_format: Option<String>,
}

fn transcribe_request(request: &mut Request, query: &str, ctx: &Ctx) -> Result<Resp, (u16, String)> {
    let content_type = header(request, "Content-Type").unwrap_or_default();
    if let Some(len) = request.body_length() {
        if len > MAX_BODY_BYTES {
            return Err((413, format!("body larger than {} bytes", MAX_BODY_BYTES)));
        }
    }
    let mut body = Vec::new();
    request
        .as_reader()
        .read_to_end(&mut body)
        .map_err(|e| (400, format!("failed to read body: {}", e)))?;

    let mut input = if content_type.starts_with("multipart/form-data") {
        parse_multipart(&content_type, &body)?
    } else if content_type.starts_with("application/json") {
        parse_json_body(&body)?
    } else if !body.is_empty() {
        let name = header(request, "X-Filename").unwrap_or_else(|| "upload.bin".to_string());
        Input { file: Some(Upload { name, data: body }), ..Default::default() }
    } else {
        Input::default()
    };
    // Query-string parameters (useful for curl and for the CLI).
    for (k, v) in parse_query(query) {
        match k.as_str() {
            "path" => input.path = Some(v),
            "response_format" => input.response_format = Some(v),
            _ => {}
        }
    }

    let (samples, source) = match (input.path.as_deref(), input.file.as_ref()) {
        (Some(p), _) => {
            let path = Path::new(p);
            if !path.is_file() {
                return Err((400, format!("path not found: {}", p)));
            }
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| p.to_string());
            (decode_path(path)?, name)
        }
        (None, Some(f)) => (decode_bytes(&f.data, &f.name, &ctx.temp_dir)?, f.name.clone()),
        (None, None) => {
            return Err((
                400,
                "send multipart/form-data with a `file` part, JSON {\"path\": \"<local file>\"}, or a raw audio body"
                    .to_string(),
            ))
        }
    };

    let duration = samples.len() as f32 / TARGET_SAMPLE_RATE as f32;
    info!("API transcribing {} ({:.1} s of audio)", source, duration);
    let (text, segments) = transcribe_samples(ctx, &samples)?;
    let _ = ctx.event_tx.send(AppEvent::ApiTranscribed {
        source: source.clone(),
        text: text.clone(),
        sample_count: samples.len(),
    });

    Ok(match input.response_format.as_deref().unwrap_or("json") {
        "text" => text_response(&text),
        "verbose_json" => json_response(
            200,
            json!({
                "task": "transcribe",
                "language": Value::Null,
                "duration": duration,
                "text": text,
                "segments": segments,
            }),
        ),
        _ => json_response(200, json!({"text": text, "duration": duration, "source": source})),
    })
}

/// Guard that removes the temporary sample file however the request ends.
struct TempFile(PathBuf);
impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn transcribe_samples(ctx: &Ctx, samples: &[f32]) -> Result<(String, Vec<Value>), (u16, String)> {
    if samples.is_empty() {
        return Ok((String::new(), Vec::new()));
    }
    let n = ctx.counter.fetch_add(1, Ordering::Relaxed);
    let path = ctx.temp_dir.join(format!("api-{}-{}.f32", std::process::id(), n));
    {
        let mut f = fs::File::create(&path).map_err(|e| (500, format!("cannot write temp file: {}", e)))?;
        let mut bytes = Vec::with_capacity(samples.len() * 4);
        for s in samples {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        f.write_all(&bytes).map_err(|e| (500, format!("cannot write temp file: {}", e)))?;
    }
    let _guard = TempFile(path.clone());

    let bounds = segment_all(
        samples,
        SegmenterConfig { min_secs: API_MIN_CHUNK_SECS, max_secs: API_MAX_CHUNK_SECS, pause_secs: API_PAUSE_SECS },
    );
    let mut texts = Vec::new();
    let mut segments = Vec::new();
    for (i, b) in bounds.iter().enumerate() {
        let (reply_tx, reply_rx) = bounded(1);
        ctx.event_tx
            .send(AppEvent::ApiTranscribeRange { path: path.clone(), start: b.start, end: b.end, reply: reply_tx })
            .map_err(|_| (503, "application is shutting down".to_string()))?;
        let text = match reply_rx.recv_timeout(CHUNK_TIMEOUT) {
            Ok(Ok(t)) => t,
            Ok(Err(e)) => return Err((500, format!("chunk {} failed: {}", i, e))),
            Err(_) => return Err((504, format!("chunk {} timed out", i))),
        };
        let text = text.trim().to_string();
        segments.push(json!({
            "id": i,
            "start": b.start as f32 / TARGET_SAMPLE_RATE as f32,
            "end": b.end as f32 / TARGET_SAMPLE_RATE as f32,
            "text": text,
        }));
        if !text.is_empty() {
            texts.push(text);
        }
    }
    Ok((texts.join(" "), segments))
}

// ---------------------------------------------------------------------------
// Body parsing
// ---------------------------------------------------------------------------

fn parse_query(query: &str) -> Vec<(String, String)> {
    query
        .split('&')
        .filter(|kv| !kv.is_empty())
        .map(|kv| {
            let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
            (percent_decode(k), percent_decode(v))
        })
        .collect()
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => match u8::from_str_radix(&s[i + 1..i + 3], 16) {
                Ok(b) => {
                    out.push(b);
                    i += 3;
                }
                Err(_) => {
                    out.push(b'%');
                    i += 1;
                }
            },
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).to_string()
}

fn parse_json_body(body: &[u8]) -> Result<Input, (u16, String)> {
    let v: Value = serde_json::from_slice(body).map_err(|e| (400, format!("invalid JSON body: {}", e)))?;
    Ok(Input {
        file: None,
        path: v.get("path").and_then(Value::as_str).map(String::from),
        response_format: v.get("response_format").and_then(Value::as_str).map(String::from),
    })
}

fn find(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if needle.is_empty() || from >= hay.len() {
        return None;
    }
    hay[from..].windows(needle.len()).position(|w| w == needle).map(|p| p + from)
}

fn parse_multipart(content_type: &str, body: &[u8]) -> Result<Input, (u16, String)> {
    let boundary = content_type
        .split(';')
        .map(str::trim)
        .find_map(|p| p.strip_prefix("boundary="))
        .map(|b| b.trim_matches('"'))
        .ok_or((400, "multipart/form-data without boundary".to_string()))?;
    let delim = format!("--{}", boundary).into_bytes();
    let mut input = Input::default();
    let mut pos = find(body, &delim, 0).ok_or((400, "malformed multipart body".to_string()))? + delim.len();
    loop {
        if body.len() < pos + 2 || body[pos..].starts_with(b"--") {
            break;
        }
        if body[pos..].starts_with(b"\r\n") {
            pos += 2;
        }
        let hdr_end = find(body, b"\r\n\r\n", pos).ok_or((400, "malformed multipart part".to_string()))?;
        let headers = String::from_utf8_lossy(&body[pos..hdr_end]).to_string();
        let data_start = hdr_end + 4;
        let next = find(body, &delim, data_start).ok_or((400, "unterminated multipart part".to_string()))?;
        let data_end = next.saturating_sub(2).max(data_start);
        let (name, filename) = parse_disposition(&headers);
        let data = &body[data_start..data_end];
        match name.as_deref() {
            Some("file") => {
                input.file = Some(Upload {
                    name: filename.unwrap_or_else(|| "upload.bin".to_string()),
                    data: data.to_vec(),
                })
            }
            Some("path") => input.path = Some(String::from_utf8_lossy(data).trim().to_string()),
            Some("response_format") => {
                input.response_format = Some(String::from_utf8_lossy(data).trim().to_string())
            }
            Some(other) => debug!("API: ignoring form field {}", other),
            None => {}
        }
        pos = next + delim.len();
    }
    Ok(input)
}

fn parse_disposition(headers: &str) -> (Option<String>, Option<String>) {
    let line = headers
        .lines()
        .find(|l| l.to_ascii_lowercase().starts_with("content-disposition:"))
        .unwrap_or("");
    let mut name = None;
    let mut filename = None;
    for part in line.split(';').map(str::trim) {
        if let Some(v) = part.strip_prefix("name=") {
            name = Some(v.trim_matches('"').to_string());
        } else if let Some(v) = part.strip_prefix("filename=") {
            let v = v.trim_matches('"');
            // Keep only the basename; clients may send full paths.
            let base = v.rsplit(['/', '\\']).next().unwrap_or(v);
            filename = Some(base.to_string());
        }
    }
    (name, filename)
}

// ---------------------------------------------------------------------------
// Audio decoding
// ---------------------------------------------------------------------------

fn extension_of(name: &str) -> Option<String> {
    Path::new(name).extension().map(|e| e.to_string_lossy().to_ascii_lowercase())
}

/// Decode a local file to mono 16 kHz samples.
fn decode_path(path: &Path) -> Result<Vec<f32>, (u16, String)> {
    let ext = path.extension().map(|e| e.to_string_lossy().to_ascii_lowercase());
    if ext.as_deref() == Some("f32") {
        let bytes = fs::read(path).map_err(|e| (400, format!("cannot read {}: {}", path.display(), e)))?;
        return Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect());
    }
    let file = fs::File::open(path).map_err(|e| (400, format!("cannot open {}: {}", path.display(), e)))?;
    match decode_symphonia(Box::new(file), ext.as_deref()) {
        Ok((samples, rate)) => Ok(resample(&samples, rate, TARGET_SAMPLE_RATE)),
        Err(sym_err) => {
            debug!("symphonia could not decode {}: {}; trying ffmpeg", path.display(), sym_err);
            decode_ffmpeg(path)
                .map_err(|ff_err| (415, format!("cannot decode {}: {} ({})", path.display(), sym_err, ff_err)))
        }
    }
}

/// Decode an uploaded body to mono 16 kHz samples.
fn decode_bytes(data: &[u8], name: &str, temp_dir: &Path) -> Result<Vec<f32>, (u16, String)> {
    let ext = extension_of(name);
    if ext.as_deref() == Some("f32") {
        return Ok(data.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect());
    }
    let cursor = Cursor::new(data.to_vec());
    match decode_symphonia(Box::new(cursor), ext.as_deref()) {
        Ok((samples, rate)) => Ok(resample(&samples, rate, TARGET_SAMPLE_RATE)),
        Err(sym_err) => {
            debug!("symphonia could not decode upload {}: {}; trying ffmpeg", name, sym_err);
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let tmp = temp_dir.join(format!(
                "upload-{}-{}.{}",
                std::process::id(),
                nanos,
                ext.unwrap_or_else(|| "bin".to_string())
            ));
            fs::write(&tmp, data).map_err(|e| (500, format!("cannot write temp file: {}", e)))?;
            let _guard = TempFile(tmp.clone());
            decode_ffmpeg(&tmp).map_err(|ff_err| (415, format!("cannot decode {}: {} ({})", name, sym_err, ff_err)))
        }
    }
}

fn decode_symphonia(
    source: Box<dyn symphonia::core::io::MediaSource>,
    ext: Option<&str>,
) -> Result<(Vec<f32>, u32), String> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
    use symphonia::core::errors::Error as SymError;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let mss = MediaSourceStream::new(source, Default::default());
    let mut hint = Hint::new();
    if let Some(e) = ext {
        hint.with_extension(e);
    }
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| format!("unrecognized container: {}", e))?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| "no audio track".to_string())?;
    let track_id = track.id;
    let params = track.codec_params.clone();
    let mut decoder = symphonia::default::get_codecs()
        .make(&params, &DecoderOptions::default())
        .map_err(|e| format!("unsupported codec: {}", e))?;

    let mut rate = params.sample_rate.unwrap_or(0);
    let mut out: Vec<f32> = Vec::new();
    let mut sample_buf: Option<SampleBuffer<f32>> = None;
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymError::ResetRequired) => break,
            Err(e) => return Err(format!("read error: {}", e)),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(SymError::DecodeError(e)) => {
                debug!("skipping undecodable packet: {}", e);
                continue;
            }
            Err(e) => return Err(format!("decode error: {}", e)),
        };
        let spec = *decoded.spec();
        rate = spec.rate;
        let channels = spec.channels.count().max(1);
        let frames = decoded.capacity() as u64;
        let need_new = sample_buf
            .as_ref()
            .map(|b| b.capacity() < (frames as usize) * channels)
            .unwrap_or(true);
        if need_new {
            sample_buf = Some(SampleBuffer::<f32>::new(frames, spec));
        }
        let buf = sample_buf.as_mut().unwrap();
        buf.copy_interleaved_ref(decoded);
        for frame in buf.samples().chunks_exact(channels) {
            out.push(frame.iter().sum::<f32>() / channels as f32);
        }
    }
    if rate == 0 {
        return Err("unknown sample rate".to_string());
    }
    if out.is_empty() {
        return Err("no decodable audio".to_string());
    }
    Ok((out, rate))
}

fn decode_ffmpeg(path: &Path) -> Result<Vec<f32>, String> {
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-nostdin", "-loglevel", "error", "-i"])
        .arg(path)
        .args(["-vn", "-f", "f32le", "-ac", "1", "-ar", "16000", "-"]);
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let out = cmd.output().map_err(|e| format!("ffmpeg not available: {}", e))?;
    if !out.status.success() {
        return Err(format!("ffmpeg failed: {}", String::from_utf8_lossy(&out.stderr).trim()));
    }
    Ok(out.stdout.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}

/// Windowed-sinc resampler (Hann window, 24 taps per side). Good enough for
/// speech at any common rate; a few seconds per hour of audio.
pub fn resample(input: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || input.is_empty() || from == 0 || to == 0 {
        return input.to_vec();
    }
    use std::f64::consts::PI;
    const TAPS: i64 = 24;
    let ratio = from as f64 / to as f64; // input samples per output sample
    let cutoff = (0.5f64).min(0.5 / ratio) * 0.95; // cycles per input sample
    let out_len = (input.len() as f64 / ratio).floor() as usize;
    let mut out = Vec::with_capacity(out_len);
    let len = input.len() as i64;
    for i in 0..out_len {
        let center = i as f64 * ratio;
        let c0 = center.floor() as i64;
        let mut acc = 0.0f64;
        let mut wsum = 0.0f64;
        for j in (c0 - TAPS + 1)..=(c0 + TAPS) {
            if j < 0 || j >= len {
                continue;
            }
            let x = j as f64 - center;
            let sinc = if x.abs() < 1e-9 { 2.0 * cutoff } else { (2.0 * PI * cutoff * x).sin() / (PI * x) };
            let window = 0.5 * (1.0 + (PI * x / TAPS as f64).cos());
            let coef = sinc * window;
            acc += coef * input[j as usize] as f64;
            wsum += coef;
        }
        out.push(if wsum.abs() > 1e-9 { (acc / wsum) as f32 } else { 0.0 });
    }
    out
}

// ---------------------------------------------------------------------------
// CLI client: `voclaude transcribe <file>...`
// ---------------------------------------------------------------------------

/// Send local files to the running app's endpoint and print the text.
/// Returns the process exit code.
pub fn cli_transcribe(args: &[String]) -> i32 {
    let mut files = Vec::new();
    let mut format = "text".to_string();
    let mut port: Option<u16> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--json" => format = "json".to_string(),
            "--verbose-json" => format = "verbose_json".to_string(),
            "--port" => {
                i += 1;
                port = args.get(i).and_then(|p| p.parse().ok());
            }
            "-h" | "--help" => {
                eprintln!("Usage: voclaude transcribe [--json|--verbose-json] [--port N] <file>...");
                return 0;
            }
            other => files.push(other.to_string()),
        }
        i += 1;
    }
    if files.is_empty() {
        eprintln!("Usage: voclaude transcribe [--json|--verbose-json] [--port N] <file>...");
        return 2;
    }
    let config = Config::load().unwrap_or_default();
    let port = port.unwrap_or(config.api_port);
    let url = format!("http://127.0.0.1:{}/v1/audio/transcriptions", port);
    let agent = ureq::AgentBuilder::new().timeout_read(Duration::from_secs(4 * 3600)).build();
    let mut failures = 0;
    for file in &files {
        let abs = match fs::canonicalize(file) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("{}: {}", file, e);
                failures += 1;
                continue;
            }
        };
        // canonicalize() yields \\?\ paths on Windows; strip the prefix for readability.
        let abs = abs.to_string_lossy().trim_start_matches(r"\\?\").to_string();
        let body = json!({"path": abs, "response_format": format}).to_string();
        let mut req = agent.post(&url).set("Content-Type", "application/json");
        if let Some(t) = config.api_token.as_deref().filter(|t| !t.trim().is_empty()) {
            req = req.set("Authorization", &format!("Bearer {}", t));
        }
        match req.send_string(&body) {
            Ok(resp) => {
                let text = resp.into_string().unwrap_or_default();
                if files.len() > 1 && format == "text" {
                    println!("== {}", file);
                }
                println!("{}", text.trim_end());
            }
            Err(ureq::Error::Status(code, resp)) => {
                let msg = resp.into_string().unwrap_or_default();
                eprintln!("{}: HTTP {}: {}", file, code, msg.trim());
                failures += 1;
            }
            Err(e) => {
                eprintln!(
                    "{}: cannot reach Voclaude at {} ({}). Is voclaude running with api_enabled = true?",
                    file, url, e
                );
                failures += 1;
            }
        }
    }
    if failures == 0 {
        0
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multipart_extracts_file_and_fields() {
        let boundary = "XyZ";
        let body = format!(
            "--{b}\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\ntext\r\n--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"C:\\clips\\a.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFFdata\r\n--{b}--\r\n",
            b = boundary
        );
        let input =
            parse_multipart(&format!("multipart/form-data; boundary={}", boundary), body.as_bytes()).unwrap();
        assert_eq!(input.response_format.as_deref(), Some("text"));
        let f = input.file.unwrap();
        assert_eq!(f.name, "a.wav");
        assert_eq!(f.data, b"RIFFdata");
    }

    #[test]
    fn resample_preserves_tone() {
        // 440 Hz at 48 kHz -> 16 kHz keeps the tone and the amplitude.
        let from = 48_000u32;
        let input: Vec<f32> = (0..48_000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / from as f32).sin())
            .collect();
        let out = resample(&input, from, 16_000);
        assert_eq!(out.len(), 16_000);
        let peak = out[1000..15_000].iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!((peak - 1.0).abs() < 0.05, "peak {}", peak);
        // zero crossings per second ~ 880
        let zc = out[1000..15_000].windows(2).filter(|w| (w[0] < 0.0) != (w[1] < 0.0)).count();
        let per_sec = zc as f32 / (14_000.0 / 16_000.0);
        assert!((per_sec - 880.0).abs() < 20.0, "zero crossings/s {}", per_sec);
    }

    #[test]
    fn query_parsing_decodes() {
        let q = parse_query("path=C%3A%5Cclips%5Ca+b.wav&response_format=text");
        assert_eq!(q[0], ("path".to_string(), "C:\\clips\\a b.wav".to_string()));
        assert_eq!(q[1].1, "text");
    }
}

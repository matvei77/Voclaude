//! Inference in a child process.
//!
//! The model (and the whole CUDA runtime with its host-side memory) lives in a
//! `voclaude.exe --worker` child. The app talks to it over stdin/stdout with
//! newline-delimited JSON. "Unloading" the model means letting the child exit,
//! which returns every byte of GPU and host memory to the OS and keeps a CUDA
//! crash from taking the recorder down with it.

use crate::app::AppEvent;
use crate::config::Config;
use crate::inference::{AsrEngine, InferenceProgress, InferenceStage, QwenEngine};
use crossbeam_channel::{bounded, unbounded, Receiver, RecvTimeoutError, Sender};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use tracing::{debug, error, info, warn};

/// Maximum characters of the previous segment fed back as context.
pub const SEGMENT_CONTEXT_CHARS: usize = 200;

/// How long one segment may take before the worker is declared hung.
const SEGMENT_TIMEOUT: Duration = Duration::from_secs(600);
/// How long a whole-file transcription may take.
const FILE_TIMEOUT: Duration = Duration::from_secs(3600);
/// How long a model load may take (first-time download included).
const PRELOAD_TIMEOUT: Duration = Duration::from_secs(1800);

/// Commands the app sends to the inference side.
#[derive(Debug)]
pub enum InferenceCommand {
    /// Load the model now (sent when recording starts, so the load overlaps speech).
    Preload,
    /// Transcribe samples `[start, end)` of the file.
    TranscribeSegment {
        session_id: String,
        index: usize,
        path: PathBuf,
        start: usize,
        end: usize,
        use_context: bool,
    },
    TranscribeFile(PathBuf),
    /// Settings changed: use this config for the next worker start. A running
    /// idle worker is stopped so the change takes effect on the next recording.
    UpdateConfig(Config),
    Unload,
    Shutdown,
}

// ---------------------------------------------------------------------------
// Wire protocol
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
enum WireCommand {
    Init { config: Config },
    Preload,
    Segment {
        session_id: String,
        index: usize,
        path: PathBuf,
        start: usize,
        end: usize,
        use_context: bool,
    },
    File { path: PathBuf },
    Shutdown,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "ev", rename_all = "snake_case")]
enum WireEvent {
    Ready,
    Progress { loading: bool, message: String },
    EngineInfo { using_gpu: bool, model: String, model_size_mb: u64 },
    Preloaded { ok: bool, error: Option<String> },
    Segment { session_id: String, index: usize, ok: bool, text: Option<String>, error: Option<String> },
    File { ok: bool, text: Option<String>, error: Option<String> },
}

// ---------------------------------------------------------------------------
// Helpers shared by both sides
// ---------------------------------------------------------------------------

/// Read samples `[start, end)` from a raw little-endian f32 file.
pub fn read_f32_range(path: &Path, start: usize, end: usize) -> Result<Vec<f32>, String> {
    if end <= start {
        return Ok(Vec::new());
    }
    let mut file = fs::File::open(path).map_err(|e| format!("open {}: {}", path.display(), e))?;
    let len = file.metadata().map_err(|e| e.to_string())?.len() as usize / 4;
    let end = end.min(len);
    if end <= start {
        return Ok(Vec::new());
    }
    file.seek(SeekFrom::Start((start * 4) as u64)).map_err(|e| e.to_string())?;
    let mut bytes = vec![0u8; (end - start) * 4];
    file.read_exact(&mut bytes).map_err(|e| format!("read segment: {}", e))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Last `max_chars` characters of `text`, cut at a word boundary.
pub fn context_tail(text: &str, max_chars: usize) -> String {
    let total = text.chars().count();
    if total <= max_chars {
        return text.trim().to_string();
    }
    let skip = total - max_chars;
    let tail: String = text.chars().skip(skip).collect();
    match tail.find(' ') {
        Some(pos) => tail[pos + 1..].trim().to_string(),
        None => tail.trim().to_string(),
    }
}

// ---------------------------------------------------------------------------
// Child side: `voclaude.exe --worker`
// ---------------------------------------------------------------------------

/// Entry point of the child process. Logs go to stderr (the parent forwards them).
pub fn run_child() -> Result<(), Box<dyn std::error::Error>> {
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    let stdout = std::io::stdout();

    let emit = |ev: &WireEvent| {
        let mut out = stdout.lock();
        if let Ok(json) = serde_json::to_string(ev) {
            let _ = writeln!(out, "{}", json);
            let _ = out.flush();
        }
    };

    // First line must be Init.
    let first = lines.next().ok_or("worker: no init line")??;
    let config = match serde_json::from_str::<WireCommand>(&first)? {
        WireCommand::Init { config } => config,
        other => return Err(format!("worker: expected init, got {:?}", other).into()),
    };
    let mut engine = QwenEngine::new_with_config(&config)?;
    emit(&WireEvent::Ready);

    let mut ctx_session: Option<String> = None;
    let mut ctx_text = String::new();

    for line in lines {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let cmd: WireCommand = match serde_json::from_str(&line) {
            Ok(c) => c,
            Err(e) => {
                warn!("worker: bad command line: {}", e);
                continue;
            }
        };
        match cmd {
            WireCommand::Init { .. } => {}
            WireCommand::Preload => {
                let mut cb = |p: InferenceProgress| {
                    emit(&WireEvent::Progress {
                        loading: p.stage == InferenceStage::LoadingModel,
                        message: p.message,
                    });
                };
                match engine.prepare(Some(&mut cb)) {
                    Ok(()) => {
                        emit(&WireEvent::EngineInfo {
                            using_gpu: engine.active_gpu(),
                            model: engine.model_label(),
                            model_size_mb: engine.model_size_mb(),
                        });
                        emit(&WireEvent::Preloaded { ok: true, error: None });
                    }
                    Err(e) => emit(&WireEvent::Preloaded { ok: false, error: Some(e.to_string()) }),
                }
            }
            WireCommand::Segment { session_id, index, path, start, end, use_context } => {
                if ctx_session.as_deref() != Some(session_id.as_str()) {
                    ctx_session = Some(session_id.clone());
                    ctx_text.clear();
                }
                let context = if use_context && !ctx_text.is_empty() {
                    Some(context_tail(&ctx_text, SEGMENT_CONTEXT_CHARS))
                } else {
                    None
                };
                let was_loaded = engine_is_loaded(&engine);
                if !was_loaded {
                    emit(&WireEvent::Progress { loading: true, message: "Loading model...".to_string() });
                }
                let result = read_f32_range(&path, start, end).and_then(|samples| {
                    engine
                        .transcribe_segment(&samples, context.as_deref())
                        .map(|(text, _stats)| text)
                        .map_err(|e| e.to_string())
                });
                if !was_loaded && engine_is_loaded(&engine) {
                    emit(&WireEvent::EngineInfo {
                        using_gpu: engine.active_gpu(),
                        model: engine.model_label(),
                        model_size_mb: engine.model_size_mb(),
                    });
                }
                match result {
                    Ok(text) => {
                        if !text.trim().is_empty() {
                            ctx_text = text.clone();
                        }
                        emit(&WireEvent::Segment { session_id, index, ok: true, text: Some(text), error: None });
                    }
                    Err(e) => emit(&WireEvent::Segment { session_id, index, ok: false, text: None, error: Some(e) }),
                }
            }
            WireCommand::File { path } => {
                let mut cb = |p: InferenceProgress| {
                    emit(&WireEvent::Progress {
                        loading: p.stage == InferenceStage::LoadingModel,
                        message: p.message,
                    });
                };
                if let Err(e) = engine.prepare(Some(&mut cb)) {
                    emit(&WireEvent::File { ok: false, text: None, error: Some(format!("Failed to load model: {}", e)) });
                    continue;
                }
                emit(&WireEvent::EngineInfo {
                    using_gpu: engine.active_gpu(),
                    model: engine.model_label(),
                    model_size_mb: engine.model_size_mb(),
                });
                match engine.transcribe_file_with_progress(&path, Some(&mut cb)) {
                    Ok(text) => emit(&WireEvent::File { ok: true, text: Some(text), error: None }),
                    Err(e) => emit(&WireEvent::File { ok: false, text: None, error: Some(e.to_string()) }),
                }
            }
            WireCommand::Shutdown => break,
        }
    }
    info!("worker: exiting");
    Ok(())
}

fn engine_is_loaded(engine: &QwenEngine) -> bool {
    engine.is_loaded()
}

// ---------------------------------------------------------------------------
// Parent side: proxy thread that owns the child process
// ---------------------------------------------------------------------------

/// Spawn the proxy thread. Same shape as the old in-process worker: a bounded
/// command channel and a join handle the watchdog can poll. `shutdown` makes
/// an in-flight wait return immediately (the child is killed) so quitting
/// never blocks on a long segment.
pub fn spawn_proxy(
    event_tx: Sender<AppEvent>,
    config: Config,
    shutdown: Arc<AtomicBool>,
) -> (Sender<InferenceCommand>, thread::JoinHandle<()>) {
    let (tx, rx) = bounded::<InferenceCommand>(8);
    let handle = thread::Builder::new()
        .name("inference-proxy".to_string())
        .spawn(move || proxy_loop(rx, event_tx, config, shutdown))
        .expect("spawn inference proxy");
    (tx, handle)
}

struct ChildHandle {
    child: Child,
    stdin: ChildStdin,
    lines: Receiver<String>,
}

impl ChildHandle {
    fn send(&mut self, cmd: &WireCommand) -> Result<(), String> {
        let json = serde_json::to_string(cmd).map_err(|e| e.to_string())?;
        writeln!(self.stdin, "{}", json).map_err(|e| format!("worker stdin: {}", e))?;
        self.stdin.flush().map_err(|e| format!("worker stdin flush: {}", e))
    }

    fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn spawn_child(config: &Config) -> Result<ChildHandle, String> {
    let exe = std::env::current_exe().map_err(|e| format!("current_exe: {}", e))?;
    let mut cmd = Command::new(&exe);
    cmd.arg("--worker")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let mut child = cmd.spawn().map_err(|e| format!("spawn worker {}: {}", exe.display(), e))?;
    let stdin = child.stdin.take().ok_or("worker stdin missing")?;
    let stdout = child.stdout.take().ok_or("worker stdout missing")?;
    let stderr = child.stderr.take().ok_or("worker stderr missing")?;
    let pid = child.id();

    // stdout -> line channel
    let (line_tx, line_rx) = unbounded::<String>();
    thread::Builder::new()
        .name("worker-stdout".to_string())
        .spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                match line {
                    Ok(l) => {
                        if line_tx.send(l).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        })
        .map_err(|e| e.to_string())?;
    // stderr -> our log
    thread::Builder::new()
        .name("worker-stderr".to_string())
        .spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines().map_while(Result::ok) {
                let l = line.trim_end();
                if !l.is_empty() {
                    info!("[worker {}] {}", pid, l);
                }
            }
        })
        .map_err(|e| e.to_string())?;

    let mut handle = ChildHandle { child, stdin, lines: line_rx };
    handle.send(&WireCommand::Init { config: config.clone() })?;
    // Wait for Ready (engine construction is quick; the model loads later).
    match handle.lines.recv_timeout(Duration::from_secs(60)) {
        Ok(line) => match serde_json::from_str::<WireEvent>(&line) {
            Ok(WireEvent::Ready) => {}
            other => {
                handle.kill();
                return Err(format!("worker did not become ready: {:?}", other));
            }
        },
        Err(_) => {
            handle.kill();
            return Err("worker did not answer within 60 s".to_string());
        }
    }
    info!("Inference worker process started (pid {})", pid);
    Ok(handle)
}

/// Outcome of waiting for one command's terminal event.
enum Wait {
    Done(WireEvent),
    Died,
    Timeout,
}

fn wait_for(
    child: &mut ChildHandle,
    event_tx: &Sender<AppEvent>,
    shutdown: &AtomicBool,
    timeout: Duration,
    is_terminal: impl Fn(&WireEvent) -> bool,
) -> Wait {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        if shutdown.load(Ordering::SeqCst) {
            child.kill();
            return Wait::Died;
        }
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        if remaining.is_zero() {
            return Wait::Timeout;
        }
        // Poll in short slices so a shutdown request is noticed promptly.
        match child.lines.recv_timeout(remaining.min(Duration::from_millis(250))) {
            Ok(line) => {
                let ev: WireEvent = match serde_json::from_str(&line) {
                    Ok(ev) => ev,
                    Err(e) => {
                        debug!("worker: unparsable line ({}): {}", e, line);
                        continue;
                    }
                };
                if is_terminal(&ev) {
                    return Wait::Done(ev);
                }
                forward_event(&ev, event_tx);
            }
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => return Wait::Died,
        }
    }
}

fn forward_event(ev: &WireEvent, event_tx: &Sender<AppEvent>) {
    match ev {
        WireEvent::Progress { loading, message } => {
            let _ = event_tx.send(AppEvent::InferenceProgress(InferenceProgress {
                stage: if *loading { InferenceStage::LoadingModel } else { InferenceStage::Transcribing },
                message: message.clone(),
            }));
        }
        WireEvent::EngineInfo { using_gpu, model, model_size_mb } => {
            let _ = event_tx.send(AppEvent::InferenceEngineInfo {
                using_gpu: *using_gpu,
                model: model.clone(),
                model_size_mb: *model_size_mb,
            });
        }
        _ => {}
    }
}

fn proxy_loop(rx: Receiver<InferenceCommand>, event_tx: Sender<AppEvent>, config: Config, shutdown: Arc<AtomicBool>) {
    let mut child: Option<ChildHandle> = None;
    let mut config = config;

    let ensure_child = |child: &mut Option<ChildHandle>, config: &Config| -> Result<(), String> {
        if let Some(c) = child.as_mut() {
            // Still alive?
            match c.child.try_wait() {
                Ok(None) => return Ok(()),
                Ok(Some(status)) => {
                    warn!("Inference worker exited ({}); restarting", status);
                    *child = None;
                }
                Err(e) => {
                    warn!("Inference worker status unknown ({}); restarting", e);
                    *child = None;
                }
            }
        }
        let _ = event_tx.send(AppEvent::InferenceProgress(InferenceProgress {
            stage: InferenceStage::LoadingModel,
            message: "Starting inference worker...".to_string(),
        }));
        *child = Some(spawn_child(config)?);
        Ok(())
    };

    for command in rx.iter() {
        if shutdown.load(Ordering::SeqCst) {
            break;
        }
        match command {
            InferenceCommand::UpdateConfig(new_config) => {
                let changed = new_config.model != config.model
                    || new_config.model_path != config.model_path
                    || new_config.use_gpu != config.use_gpu
                    || new_config.quantization != config.quantization
                    || new_config.max_new_tokens != config.max_new_tokens
                    || new_config.adaptive_max_new_tokens != config.adaptive_max_new_tokens
                    || new_config.max_chunk_seconds != config.max_chunk_seconds
                    || new_config.language != config.language
                    || new_config.legacy_prompt != config.legacy_prompt
                    || new_config.require_gpu != config.require_gpu;
                config = new_config;
                if changed {
                    if let Some(mut c) = child.take() {
                        info!("Inference settings changed; restarting worker on next use");
                        stop_child(&mut c);
                    }
                }
            }
            InferenceCommand::Preload => {
                if let Err(e) = ensure_child(&mut child, &config) {
                    warn!("Preload: {}", e);
                    continue;
                }
                let c = child.as_mut().unwrap();
                if let Err(e) = c.send(&WireCommand::Preload) {
                    warn!("Preload send failed: {}", e);
                    c.kill();
                    child = None;
                    continue;
                }
                match wait_for(c, &event_tx, &shutdown, PRELOAD_TIMEOUT, |ev| matches!(ev, WireEvent::Preloaded { .. })) {
                    Wait::Done(WireEvent::Preloaded { ok: false, error }) => {
                        warn!("Preload failed: {}", error.unwrap_or_default());
                    }
                    Wait::Done(_) => {}
                    Wait::Died => {
                        warn!("Inference worker died during preload");
                        child = None;
                    }
                    Wait::Timeout => {
                        warn!("Inference worker hung during preload; killing");
                        c.kill();
                        child = None;
                    }
                }
            }
            InferenceCommand::TranscribeSegment { session_id, index, path, start, end, use_context } => {
                let fail = |msg: String| {
                    let _ = event_tx.send(AppEvent::SegmentTranscribed {
                        session_id: session_id.clone(),
                        index,
                        result: Err(msg),
                    });
                };
                if let Err(e) = ensure_child(&mut child, &config) {
                    fail(format!("worker start failed: {}", e));
                    continue;
                }
                let c = child.as_mut().unwrap();
                let cmd = WireCommand::Segment {
                    session_id: session_id.clone(),
                    index,
                    path,
                    start,
                    end,
                    use_context,
                };
                if let Err(e) = c.send(&cmd) {
                    c.kill();
                    child = None;
                    fail(format!("worker send failed: {}", e));
                    continue;
                }
                match wait_for(c, &event_tx, &shutdown, SEGMENT_TIMEOUT, |ev| matches!(ev, WireEvent::Segment { .. })) {
                    Wait::Done(WireEvent::Segment { ok, text, error, .. }) => {
                        let result = if ok { Ok(text.unwrap_or_default()) } else { Err(error.unwrap_or_else(|| "unknown worker error".into())) };
                        let _ = event_tx.send(AppEvent::SegmentTranscribed { session_id, index, result });
                    }
                    Wait::Done(_) => fail("unexpected worker reply".to_string()),
                    Wait::Died => {
                        child = None;
                        fail("inference worker exited unexpectedly".to_string());
                    }
                    Wait::Timeout => {
                        c.kill();
                        child = None;
                        fail("inference worker timed out".to_string());
                    }
                }
            }
            InferenceCommand::TranscribeFile(path) => {
                let fail = |msg: String| {
                    let _ = event_tx.send(AppEvent::TranscriptionComplete(Err(msg)));
                };
                if let Err(e) = ensure_child(&mut child, &config) {
                    fail(format!("worker start failed: {}", e));
                    continue;
                }
                let c = child.as_mut().unwrap();
                if let Err(e) = c.send(&WireCommand::File { path }) {
                    c.kill();
                    child = None;
                    fail(format!("worker send failed: {}", e));
                    continue;
                }
                match wait_for(c, &event_tx, &shutdown, FILE_TIMEOUT, |ev| matches!(ev, WireEvent::File { .. })) {
                    Wait::Done(WireEvent::File { ok, text, error }) => {
                        let result = if ok { Ok(text.unwrap_or_default()) } else { Err(error.unwrap_or_else(|| "unknown worker error".into())) };
                        let _ = event_tx.send(AppEvent::TranscriptionComplete(result));
                    }
                    Wait::Done(_) => fail("unexpected worker reply".to_string()),
                    Wait::Died => {
                        child = None;
                        fail("inference worker exited unexpectedly".to_string());
                    }
                    Wait::Timeout => {
                        c.kill();
                        child = None;
                        fail("inference worker timed out".to_string());
                    }
                }
            }
            InferenceCommand::Unload => {
                if let Some(mut c) = child.take() {
                    info!("Stopping inference worker (idle)");
                    stop_child(&mut c);
                }
            }
            InferenceCommand::Shutdown => break,
        }
    }
    if let Some(mut c) = child.take() {
        stop_child(&mut c);
    }
}

/// Ask the child to exit; kill it if it does not within a couple of seconds.
fn stop_child(c: &mut ChildHandle) {
    let _ = c.send(&WireCommand::Shutdown);
    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    loop {
        match c.child.try_wait() {
            Ok(Some(_)) => return,
            Ok(None) if std::time::Instant::now() < deadline => thread::sleep(Duration::from_millis(50)),
            _ => {
                error!("Inference worker did not exit; killing");
                c.kill();
                return;
            }
        }
    }
}

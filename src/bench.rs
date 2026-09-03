//! `voclaude --bench <file|dir>... [--tag NAME] [--out results.json]`
//!
//! Transcribes every `.f32` / `.wav` file given (directories are scanned),
//! prints per-file and aggregate timing, and — when a `<stem>.ref.txt` sits
//! next to a file — the word error rate against it. Hypotheses are written to
//! `<stem>.<tag>.hyp.txt` so runs with different settings can be diffed.

use crate::audio::segmenter::{segment_all, SegmenterConfig};
use crate::config::Config;
use crate::inference::{AsrEngine, QwenEngine};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Serialize)]
struct FileResult {
    file: String,
    audio_secs: f64,
    wall_ms: f64,
    mel_ms: f64,
    encode_ms: f64,
    prefill_ms: f64,
    decode_ms: f64,
    n_audio_tokens: usize,
    n_generated: usize,
    decode_tok_per_s: f64,
    realtime_factor: f64,
    chars: usize,
    wer: Option<f64>,
    error: Option<String>,
    /// Stream mode: number of segments the recording was cut into.
    #[serde(skip_serializing_if = "Option::is_none")]
    segments: Option<usize>,
    /// Stream mode: simulated seconds from "stop" to the final transcript.
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_latency_s: Option<f64>,
    /// Stream mode: worst backlog (seconds the worker lagged behind live audio).
    #[serde(skip_serializing_if = "Option::is_none")]
    max_lag_s: Option<f64>,
}

#[derive(Debug, Serialize)]
struct BenchReport {
    tag: String,
    model: String,
    gpu: bool,
    prompt_style: String,
    load_ms: f64,
    files: Vec<FileResult>,
    total_audio_secs: f64,
    total_wall_ms: f64,
    mean_decode_tok_per_s: f64,
    mean_wer: Option<f64>,
}

pub fn run(config: &Config, args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let mut inputs: Vec<PathBuf> = Vec::new();
    let mut tag = "run".to_string();
    let mut out: Option<PathBuf> = None;
    let stream = args.iter().any(|a| a == "--stream");
    let no_context = args.iter().any(|a| a == "--no-context");
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if a == "--tag" {
            tag = args.get(i + 1).cloned().ok_or("--tag needs a value")?;
            i += 2;
            continue;
        }
        if a == "--out" {
            out = Some(PathBuf::from(args.get(i + 1).ok_or("--out needs a value")?));
            i += 2;
            continue;
        }
        if a.starts_with("--") {
            // Engine overrides (--cpu, --model-tier, ...) were consumed by apply_cli_overrides.
            if matches!(a.as_str(), "--model-tier" | "--model" | "--model-dir" | "--max-new-tokens" | "--chunk-seconds") {
                i += 2;
            } else {
                i += 1;
            }
            continue;
        }
        let p = PathBuf::from(a);
        if p.is_dir() {
            let mut files: Vec<PathBuf> = std::fs::read_dir(&p)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|f| is_audio(f))
                .collect();
            files.sort();
            inputs.extend(files);
        } else if p.is_file() {
            inputs.push(p);
        } else {
            return Err(format!("Not found: {}", a).into());
        }
        i += 1;
    }
    if inputs.is_empty() {
        return Err("No audio files given. Usage: voclaude --bench <file|dir>... [--tag NAME] [--out results.json]".into());
    }

    let mut engine = QwenEngine::new_with_config(config)?;
    eprintln!("Loading {} (gpu={}, prompt={:?})...", config.model, config.use_gpu, engine.prompt_style());
    let t_load = Instant::now();
    engine.prepare(None)?;
    let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;
    eprintln!("Model loaded in {:.2}s", load_ms / 1000.0);

    // Warm-up: first call pays kernel-load/JIT costs that would skew file 1.
    if let Some(first) = inputs.first() {
        let samples = crate::inference::load_audio_file(first)?;
        let warm: Vec<f32> = samples.iter().take(16000 * 3).copied().collect();
        if !warm.is_empty() {
            let _ = engine.transcribe_segment(&warm, None);
            let _ = engine.take_stats();
        }
    }

    if stream {
        println!(
            "{:<28} {:>7} {:>8} {:>4} {:>7} {:>7} {:>6} {:>7} {:>6} {:>6}",
            "file", "audio_s", "wall_ms", "segs", "stop_s", "maxlag", "toks", "tok/s", "xRT", "WER"
        );
    } else {
        println!(
            "{:<28} {:>7} {:>8} {:>6} {:>7} {:>7} {:>8} {:>6} {:>7} {:>6} {:>6}",
            "file", "audio_s", "wall_ms", "mel", "encode", "prefill", "decode", "toks", "tok/s", "xRT", "WER"
        );
    }
    let seg_cfg = SegmenterConfig {
        min_secs: config.segment_min_seconds,
        max_secs: config.segment_max_seconds,
        pause_secs: config.segment_pause_seconds,
    };
    let mut results = Vec::new();
    for path in &inputs {
        let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        let t = Instant::now();
        let mut segments_n = None;
        let mut stop_latency = None;
        let mut max_lag = None;
        let result = if stream {
            simulate_stream(&mut engine, path, seg_cfg, !no_context).map(|r| {
                segments_n = Some(r.segments);
                stop_latency = Some(r.stop_latency_s);
                max_lag = Some(r.max_lag_s);
                r.text
            })
        } else {
            engine.transcribe_file_with_progress(path, None)
        };
        let wall_ms = t.elapsed().as_secs_f64() * 1000.0;
        let stats = engine.take_stats();
        let (text, error) = match result {
            Ok(text) => (text, None),
            Err(e) => (String::new(), Some(e.to_string())),
        };
        let audio_secs = if stats.audio_secs > 0.0 {
            stats.audio_secs
        } else {
            std::fs::metadata(path).map(|m| m.len() as f64 / 64000.0).unwrap_or(0.0)
        };
        let hyp_path = path.with_extension(format!("{}.hyp.txt", tag));
        if error.is_none() {
            let _ = std::fs::write(&hyp_path, &text);
        }
        let ref_path = path.with_extension("ref.txt");
        let wer = if ref_path.exists() {
            std::fs::read_to_string(&ref_path).ok().map(|r| word_error_rate(&r, &text))
        } else {
            None
        };
        let fr = FileResult {
            file: name.clone(),
            audio_secs,
            wall_ms,
            mel_ms: stats.mel_ms,
            encode_ms: stats.encode_ms,
            prefill_ms: stats.prefill_ms,
            decode_ms: stats.decode_ms,
            n_audio_tokens: stats.n_audio_tokens,
            n_generated: stats.n_generated,
            decode_tok_per_s: stats.decode_tokens_per_sec(),
            realtime_factor: if wall_ms > 0.0 { audio_secs * 1000.0 / wall_ms } else { 0.0 },
            chars: text.chars().count(),
            wer,
            error,
            segments: segments_n,
            stop_latency_s: stop_latency,
            max_lag_s: max_lag,
        };
        if stream {
            println!(
                "{:<28} {:>7.1} {:>8.0} {:>4} {:>7.2} {:>7.2} {:>6} {:>7.1} {:>6.1} {:>6}",
                truncate(&fr.file, 28),
                fr.audio_secs,
                fr.wall_ms,
                fr.segments.unwrap_or(0),
                fr.stop_latency_s.unwrap_or(0.0),
                fr.max_lag_s.unwrap_or(0.0),
                fr.n_generated,
                fr.decode_tok_per_s,
                fr.realtime_factor,
                fr.wer.map(|w| format!("{:.1}%", w * 100.0)).unwrap_or_else(|| "-".to_string()),
            );
        } else {
            println!(
                "{:<28} {:>7.1} {:>8.0} {:>6.0} {:>7.0} {:>7.0} {:>8.0} {:>6} {:>7.1} {:>6.1} {:>6}",
                truncate(&fr.file, 28),
                fr.audio_secs,
                fr.wall_ms,
                fr.mel_ms,
                fr.encode_ms,
                fr.prefill_ms,
                fr.decode_ms,
                fr.n_generated,
                fr.decode_tok_per_s,
                fr.realtime_factor,
                fr.wer.map(|w| format!("{:.1}%", w * 100.0)).unwrap_or_else(|| "-".to_string()),
            );
        }
        if let Some(err) = &fr.error {
            println!("    ERROR: {}", err);
        }
        results.push(fr);
    }

    if stream {
        let lat: Vec<f64> = results.iter().filter_map(|r| r.stop_latency_s).collect();
        if !lat.is_empty() {
            let max = lat.iter().cloned().fold(0.0, f64::max);
            let mean = lat.iter().sum::<f64>() / lat.len() as f64;
            println!("STREAM stop-to-result latency: mean {:.2}s, max {:.2}s", mean, max);
        }
    }
    let total_audio: f64 = results.iter().map(|r| r.audio_secs).sum();
    let total_wall: f64 = results.iter().map(|r| r.wall_ms).sum();
    let toks: Vec<f64> = results.iter().filter(|r| r.n_generated > 20).map(|r| r.decode_tok_per_s).collect();
    let mean_tok = if toks.is_empty() { 0.0 } else { toks.iter().sum::<f64>() / toks.len() as f64 };
    let wers: Vec<f64> = results.iter().filter_map(|r| r.wer).collect();
    let mean_wer = if wers.is_empty() { None } else { Some(wers.iter().sum::<f64>() / wers.len() as f64) };
    println!(
        "TOTAL audio {:.1}s, wall {:.1}s ({:.1}x realtime), mean decode {:.1} tok/s{}",
        total_audio,
        total_wall / 1000.0,
        if total_wall > 0.0 { total_audio * 1000.0 / total_wall } else { 0.0 },
        mean_tok,
        mean_wer.map(|w| format!(", mean WER {:.1}%", w * 100.0)).unwrap_or_default()
    );

    let report = BenchReport {
        tag,
        model: config.model.clone(),
        gpu: engine.active_gpu(),
        prompt_style: format!("{:?}", engine.prompt_style()),
        load_ms,
        files: results,
        total_audio_secs: total_audio,
        total_wall_ms: total_wall,
        mean_decode_tok_per_s: mean_tok,
        mean_wer,
    };
    if let Some(out) = out {
        if let Some(parent) = out.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(&out, serde_json::to_string_pretty(&report)?)?;
        eprintln!("Wrote {}", out.display());
    }
    Ok(())
}

struct StreamSim {
    text: String,
    segments: usize,
    stop_latency_s: f64,
    max_lag_s: f64,
}

/// Replay a recording through the live pipeline's logic: cut it with the same
/// segmenter, transcribe segments in order with context, and model the timeline
/// a real recording would have (a segment can start no earlier than it closes;
/// the worker is sequential). `stop_latency_s` is the simulated wait after the
/// user presses stop.
fn simulate_stream(
    engine: &mut QwenEngine,
    path: &Path,
    seg_cfg: SegmenterConfig,
    use_context: bool,
) -> Result<StreamSim, Box<dyn std::error::Error>> {
    let samples = crate::inference::load_audio_file(path)?;
    let bounds = segment_all(&samples, seg_cfg);
    let sr = 16000.0;
    let mut clock = 0.0f64; // simulated worker time
    let mut max_lag = 0.0f64;
    let mut texts: Vec<String> = Vec::new();
    let mut prev = String::new();
    for b in &bounds {
        let available_at = b.end as f64 / sr;
        let start_at = clock.max(available_at);
        let t = Instant::now();
        let ctx = if use_context && !prev.is_empty() { Some(tail(&prev, 200)) } else { None };
        let (text, _stats) = engine.transcribe_segment(&samples[b.start..b.end], ctx.as_deref())?;
        let took = t.elapsed().as_secs_f64();
        clock = start_at + took;
        max_lag = max_lag.max(clock - available_at);
        if !text.trim().is_empty() {
            prev = text.clone();
            texts.push(text.trim().to_string());
        }
    }
    let audio_end = samples.len() as f64 / sr;
    Ok(StreamSim {
        text: texts.join(" "),
        segments: bounds.len(),
        stop_latency_s: (clock - audio_end).max(0.0),
        max_lag_s: max_lag,
    })
}

fn tail(text: &str, max_chars: usize) -> String {
    let total = text.chars().count();
    if total <= max_chars {
        return text.trim().to_string();
    }
    let t: String = text.chars().skip(total - max_chars).collect();
    match t.find(' ') {
        Some(pos) => t[pos + 1..].trim().to_string(),
        None => t.trim().to_string(),
    }
}

fn is_audio(p: &Path) -> bool {
    p.extension()
        .map(|e| {
            let e = e.to_string_lossy().to_ascii_lowercase();
            e == "f32" || e == "wav"
        })
        .unwrap_or(false)
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        s.chars().take(n - 1).collect::<String>() + "~"
    }
}

/// Normalise for WER: lowercase, strip punctuation, split on whitespace.
pub fn normalize_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric())
                .flat_map(|c| c.to_lowercase())
                .collect::<String>()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// Word error rate = (substitutions + deletions + insertions) / reference words.
pub fn word_error_rate(reference: &str, hypothesis: &str) -> f64 {
    let r = normalize_words(reference);
    let h = normalize_words(hypothesis);
    if r.is_empty() {
        return if h.is_empty() { 0.0 } else { 1.0 };
    }
    let mut prev: Vec<usize> = (0..=h.len()).collect();
    let mut cur = vec![0usize; h.len() + 1];
    for i in 1..=r.len() {
        cur[0] = i;
        for j in 1..=h.len() {
            let cost = if r[i - 1] == h[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[h.len()] as f64 / r.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wer_basic() {
        assert_eq!(word_error_rate("a b c", "a b c"), 0.0);
        assert!((word_error_rate("a b c", "a x c") - 1.0 / 3.0).abs() < 1e-9);
        assert!((word_error_rate("a b c", "a c") - 1.0 / 3.0).abs() < 1e-9);
        assert!((word_error_rate("Hello, world!", "hello world") - 0.0).abs() < 1e-9);
        assert!((word_error_rate("Привет, мир.", "привет мир") - 0.0).abs() < 1e-9);
    }
}

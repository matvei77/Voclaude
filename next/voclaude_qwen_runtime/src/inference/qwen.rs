//! Qwen speech-to-text inference engine (persistent Python server).
//!
//! Spawns a persistent Python server process that loads the model once and
//! handles multiple transcription requests via JSON lines on stdin/stdout.
//! This eliminates cold-start latency after the first request.

use crate::config::Config;
use crate::inference::{InferenceProgress, InferenceStage};
use hound::{SampleFormat, WavSpec, WavWriter};
use serde::Deserialize;
use serde_json::json;
use std::fs;
use std::io::{BufRead, BufReader, Write as IoWrite};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use tempfile::NamedTempFile;
use tracing::{debug, info, warn};

const DEFAULT_MODEL_SIZE_MB: u64 = 3300;

/// Holds the persistent Python server process.
struct ServerProcess {
    child: Child,
    stdin: ChildStdin,
    reader: BufReader<ChildStdout>,
}

impl ServerProcess {
    /// Read one JSON line from the server's stdout.
    fn read_line(&mut self) -> Result<String, Box<dyn std::error::Error>> {
        let mut line = String::new();
        let n = self.reader.read_line(&mut line)?;
        if n == 0 {
            return Err("Server process closed stdout unexpectedly".into());
        }
        Ok(line)
    }

    /// Send a JSON line to the server's stdin.
    fn send(&mut self, value: &serde_json::Value) -> Result<(), Box<dyn std::error::Error>> {
        let serialized = serde_json::to_string(value)?;
        writeln!(self.stdin, "{}", serialized)?;
        self.stdin.flush()?;
        Ok(())
    }

    /// Send shutdown command and wait for the process to exit.
    fn shutdown(&mut self) {
        let _ = self.send(&json!({"command": "shutdown"}));
        // Give it a moment to exit gracefully
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) => {
                    if Instant::now() >= deadline {
                        warn!("Server did not exit gracefully, killing");
                        let _ = self.child.kill();
                        let _ = self.child.wait();
                        break;
                    }
                    thread::sleep(Duration::from_millis(100));
                }
                Err(_) => {
                    let _ = self.child.kill();
                    let _ = self.child.wait();
                    break;
                }
            }
        }
    }

    /// Check if the server process is still alive.
    fn is_alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }
}

impl Drop for ServerProcess {
    fn drop(&mut self) {
        self.shutdown();
    }
}

pub struct QwenEngine {
    active_gpu: bool,
    use_gpu: bool,
    language: Option<String>,
    python_path: Option<String>,
    script_path: Option<String>,
    model: String,
    dtype: String,
    device: String,
    max_new_tokens: u32,
    chunk_seconds: f32,
    chunk_overlap_seconds: f32,
    timeout_seconds: u64,
    require_gpu: bool,
    resolved_python: Option<String>,
    resolved_script: Option<PathBuf>,
    server: Option<ServerProcess>,
}

impl QwenEngine {
    pub fn new(use_gpu: bool) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            active_gpu: false,
            use_gpu,
            language: None,
            python_path: None,
            script_path: None,
            model: "Qwen/Qwen3-ASR-1.7B".to_string(),
            dtype: "bfloat16".to_string(),
            device: "cuda:0".to_string(),
            max_new_tokens: 2048,
            chunk_seconds: 60.0,
            chunk_overlap_seconds: 2.0,
            timeout_seconds: 7200,
            require_gpu: true,
            resolved_python: None,
            resolved_script: None,
            server: None,
        })
    }

    pub fn new_with_config(config: &Config) -> Result<Self, Box<dyn std::error::Error>> {
        let mut engine = Self::new(config.use_gpu)?;
        engine.language = normalize_language(config.language.as_deref());
        engine.python_path = config.qwen_python_path.clone();
        engine.script_path = config.qwen_script_path.clone();
        engine.model = config.qwen_model.clone();
        engine.dtype = config.qwen_dtype.clone();
        engine.device = config.qwen_device.clone();
        engine.max_new_tokens = config.qwen_max_new_tokens;
        engine.chunk_seconds = config.qwen_chunk_seconds;
        engine.chunk_overlap_seconds = config.qwen_chunk_overlap_seconds;
        engine.timeout_seconds = config.qwen_timeout_seconds;
        engine.require_gpu = config.qwen_require_gpu;
        Ok(engine)
    }

    pub fn active_gpu(&self) -> bool {
        self.active_gpu
    }

    pub fn model_label(&self) -> String {
        format!("qwen-python ({})", self.model)
    }

    pub fn model_size_mb(&self) -> u64 {
        if self.model.contains("0.6B") {
            1400
        } else if self.model.contains("1.7B") {
            3300
        } else {
            DEFAULT_MODEL_SIZE_MB
        }
    }

    pub fn unload(&mut self) {
        if let Some(mut server) = self.server.take() {
            info!("Shutting down Qwen server process");
            server.shutdown();
        }
    }

    fn resolve_python(&self) -> Result<String, Box<dyn std::error::Error>> {
        if let Some(path) = self.python_path.as_deref() {
            if !path.trim().is_empty() {
                let candidate = PathBuf::from(path);
                if candidate.exists() {
                    return Ok(candidate.to_string_lossy().to_string());
                }
                return Err(format!("Configured qwen_python_path not found: {}", path).into());
            }
        }

        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Ok(cwd) = std::env::current_dir() {
            candidates.push(cwd.join("../../tools/qwen3_asr_smoke/.venv/Scripts/python.exe"));
            candidates.push(cwd.join("../../tools/qwen3_asr_smoke/.venv/bin/python"));
            candidates.push(cwd.join("tools/qwen3_asr_smoke/.venv/Scripts/python.exe"));
            candidates.push(cwd.join("tools/qwen3_asr_smoke/.venv/bin/python"));
        }

        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                candidates.push(
                    dir.join("../../../tools/qwen3_asr_smoke/.venv/Scripts/python.exe"),
                );
                candidates.push(dir.join("../../../tools/qwen3_asr_smoke/.venv/bin/python"));
            }
        }

        if let Some(found) = candidates.into_iter().find(|p| p.exists()) {
            return Ok(found.to_string_lossy().to_string());
        }

        Err("Could not find Qwen Python environment. Set qwen_python_path in config.".into())
    }

    fn resolve_script(&self) -> Result<PathBuf, Box<dyn std::error::Error>> {
        if let Some(path) = self.script_path.as_deref() {
            if !path.trim().is_empty() {
                let candidate = PathBuf::from(path);
                if candidate.exists() {
                    return Ok(candidate);
                }
                return Err(format!("Configured qwen_script_path not found: {}", path).into());
            }
        }

        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Ok(cwd) = std::env::current_dir() {
            candidates.push(cwd.join("../../tools/qwen3_asr_smoke/transcribe.py"));
            candidates.push(cwd.join("tools/qwen3_asr_smoke/transcribe.py"));
        }

        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                candidates.push(dir.join("../../../tools/qwen3_asr_smoke/transcribe.py"));
            }
        }

        candidates
            .into_iter()
            .find(|p| p.exists())
            .ok_or_else(|| "Could not auto-discover qwen transcribe.py path".into())
    }

    fn resolve_paths(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.resolved_python.is_none() {
            self.resolved_python = Some(self.resolve_python()?);
        }
        if self.resolved_script.is_none() {
            self.resolved_script = Some(self.resolve_script()?);
        }
        Ok(())
    }

    /// Ensure the persistent server is running and the model is loaded.
    fn ensure_server_ready(
        &mut self,
        progress: &mut Option<&mut dyn FnMut(InferenceProgress)>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // If we have a server, check it's still alive
        if let Some(ref mut server) = self.server {
            if server.is_alive() {
                return Ok(());
            }
            warn!("Qwen server process died, restarting");
            self.server = None;
        }

        self.resolve_paths()?;

        let python = self.resolved_python.as_ref().unwrap().clone();
        let script = self.resolved_script.as_ref().unwrap().clone();

        if let Some(cb) = progress.as_deref_mut() {
            cb(InferenceProgress {
                stage: InferenceStage::LoadingModel,
                message: "Starting Qwen server...".to_string(),
                percent: None,
            });
        }

        let device = if self.use_gpu {
            self.device.clone()
        } else {
            "cpu".to_string()
        };

        let dtype = if device.starts_with("cuda") {
            self.dtype.clone()
        } else {
            "float32".to_string()
        };

        let mut args = vec![
            script.to_string_lossy().to_string(),
            "--server".to_string(),
            "--model".to_string(),
            self.model.clone(),
            "--device".to_string(),
            device,
            "--dtype".to_string(),
            dtype,
            "--max-new-tokens".to_string(),
            self.max_new_tokens.to_string(),
        ];

        if let Some(lang) = self.language.as_ref() {
            args.push("--language".to_string());
            args.push(lang.clone());
        }

        debug!("Launching Qwen server: {} {:?}", python, args);

        if let Some(cb) = progress.as_deref_mut() {
            cb(InferenceProgress {
                stage: InferenceStage::LoadingModel,
                message: "Loading Qwen model (first time may take a while)...".to_string(),
                percent: None,
            });
        }

        let mut cmd = Command::new(&python);
        cmd.args(&args);
        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        #[cfg(target_os = "windows")]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
        }

        let mut child = cmd.spawn().map_err(|e| {
            format!("Failed to spawn Qwen server process: {}", e)
        })?;

        let stdin = child.stdin.take().ok_or("Failed to capture server stdin")?;
        let stdout = child.stdout.take().ok_or("Failed to capture server stdout")?;

        // Drain stderr in a background thread to prevent deadlocks
        if let Some(stderr) = child.stderr.take() {
            thread::spawn(move || {
                let reader = BufReader::new(stderr);
                for line in reader.lines() {
                    match line {
                        Ok(text) => {
                            if !text.trim().is_empty() {
                                debug!("Qwen server stderr: {}", text.trim());
                            }
                        }
                        Err(_) => break,
                    }
                }
            });
        }

        let mut server = ServerProcess {
            child,
            stdin,
            reader: BufReader::new(stdout),
        };

        // Wait for the "ready" signal with timeout
        let timeout = Duration::from_secs(self.timeout_seconds.max(60));
        let started = Instant::now();

        let ready_line = loop {
            if started.elapsed() > timeout {
                server.shutdown();
                return Err(format!(
                    "Qwen server did not become ready within {:?}",
                    timeout
                ).into());
            }

            // Check if process died
            if !server.is_alive() {
                return Err("Qwen server process exited before becoming ready".into());
            }

            // Try to read a line (blocking, but stderr is drained separately)
            match server.read_line() {
                Ok(line) => break line,
                Err(e) => {
                    return Err(format!("Failed to read server ready signal: {}", e).into());
                }
            }
        };

        // Parse the ready response
        let ready: ServerReady = serde_json::from_str(ready_line.trim()).map_err(|e| {
            let truncated = if ready_line.len() > 500 {
                &ready_line[..500]
            } else {
                &ready_line
            };
            format!("Failed to parse server ready signal: {}. Got: {}", e, truncated)
        })?;

        if ready.status == "error" {
            let err_msg = ready.error.unwrap_or_else(|| "unknown error".to_string());
            return Err(format!("Qwen server failed to start: {}", err_msg).into());
        }

        if ready.status != "ready" {
            return Err(format!(
                "Unexpected server status: '{}' (expected 'ready')",
                ready.status
            ).into());
        }

        self.active_gpu = ready
            .device
            .as_deref()
            .map(|d| d.to_ascii_lowercase().contains("cuda"))
            .unwrap_or(false);

        if self.use_gpu && self.require_gpu && !self.active_gpu {
            server.shutdown();
            return Err("CUDA is required but Qwen server started without CUDA".into());
        }

        info!(
            "Qwen server ready: model={} device={} load={:.2}s",
            ready.model.as_deref().unwrap_or("?"),
            ready.device.as_deref().unwrap_or("?"),
            ready.load_seconds.unwrap_or(0.0),
        );

        self.server = Some(server);
        Ok(())
    }

    pub fn prepare(
        &mut self,
        mut progress: Option<&mut dyn FnMut(InferenceProgress)>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.ensure_server_ready(&mut progress)
    }

    pub fn transcribe(&mut self, samples: &[f32]) -> Result<String, Box<dyn std::error::Error>> {
        self.transcribe_with_progress(samples, None)
    }

    pub fn transcribe_file_with_progress(
        &mut self,
        path: &Path,
        progress: Option<&mut dyn FnMut(InferenceProgress)>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let prepared_path;
        let mut temp_file: Option<NamedTempFile> = None;

        if path
            .extension()
            .map(|v| v.to_string_lossy().eq_ignore_ascii_case("f32"))
            .unwrap_or(false)
        {
            let samples = load_f32_file(path)?;
            let file = NamedTempFile::new()?;
            write_wav_mono_16k(file.path(), &samples)?;
            prepared_path = file.path().to_path_buf();
            temp_file = Some(file);
        } else {
            prepared_path = path.to_path_buf();
        }

        let result = self.send_transcribe_request(&prepared_path, progress);
        drop(temp_file);
        result
    }

    pub fn transcribe_with_progress(
        &mut self,
        samples: &[f32],
        progress: Option<&mut dyn FnMut(InferenceProgress)>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let tmp = NamedTempFile::new()?;
        write_wav_mono_16k(tmp.path(), samples)?;
        self.send_transcribe_request(tmp.path(), progress)
    }

    /// Send a transcription request to the persistent server.
    fn send_transcribe_request(
        &mut self,
        audio_path: &Path,
        mut progress: Option<&mut dyn FnMut(InferenceProgress)>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.ensure_server_ready(&mut progress)?;

        if let Some(cb) = progress.as_deref_mut() {
            cb(InferenceProgress {
                stage: InferenceStage::Transcribing,
                message: "Transcribing with Qwen...".to_string(),
                percent: None,
            });
        }

        let request = json!({
            "audio_path": audio_path.to_string_lossy(),
            "language": self.language,
            "chunk_seconds": if self.chunk_seconds > 0.0 { Some(self.chunk_seconds) } else { None },
            "chunk_overlap_seconds": if self.chunk_overlap_seconds > 0.0 { Some(self.chunk_overlap_seconds) } else { None },
        });

        let server = self.server.as_mut().ok_or("Server not running")?;

        debug!("Sending transcribe request: {}", request);
        server.send(&request)?;

        // Read response (blocking)
        let response_line = server.read_line().map_err(|e| {
            format!("Failed to read transcription response: {}", e)
        })?;

        let response: ServerResponse = serde_json::from_str(response_line.trim()).map_err(|e| {
            let truncated = if response_line.len() > 500 {
                &response_line[..500]
            } else {
                &response_line
            };
            format!("Failed to parse server response: {}. Got: {}", e, truncated)
        })?;

        if response.status == "error" {
            let err_msg = response.error.unwrap_or_else(|| "unknown error".to_string());
            return Err(format!("Qwen transcription failed: {}", err_msg).into());
        }

        if response.status != "ok" {
            return Err(format!(
                "Unexpected server response status: '{}'",
                response.status
            ).into());
        }

        // Update GPU state from response
        if let Some(ref device) = response.device {
            self.active_gpu = device.to_ascii_lowercase().contains("cuda");
        }

        let first = response
            .results
            .as_ref()
            .and_then(|r| r.first())
            .ok_or("Server response contained no results")?;

        info!(
            "Qwen transcription complete: infer={:.2}s",
            response.inference_seconds.unwrap_or(0.0),
        );

        Ok(first.text.trim().to_string())
    }
}

impl Default for QwenEngine {
    fn default() -> Self {
        Self::new(true).expect("Failed to create Qwen engine")
    }
}

impl Drop for QwenEngine {
    fn drop(&mut self) {
        self.unload();
    }
}

// ---------------------------------------------------------------------------
// Server protocol types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ServerReady {
    status: String,
    model: Option<String>,
    device: Option<String>,
    #[allow(dead_code)]
    dtype: Option<String>,
    load_seconds: Option<f32>,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ServerResponse {
    status: String,
    #[allow(dead_code)]
    model: Option<String>,
    device: Option<String>,
    #[allow(dead_code)]
    load_seconds: Option<f32>,
    inference_seconds: Option<f32>,
    results: Option<Vec<ServerResult>>,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ServerResult {
    text: String,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn normalize_language(lang: Option<&str>) -> Option<String> {
    let trimmed = lang.map(str::trim).filter(|value| !value.is_empty());
    match trimmed {
        Some("auto") => None,
        Some(value) => Some(value.to_string()),
        None => None,
    }
}

fn load_f32_file(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err("Invalid f32 audio file length".into());
    }
    let mut samples = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        samples.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(samples)
}

fn write_wav_mono_16k(path: &Path, samples: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut writer = WavWriter::create(path, spec)?;
    for &sample in samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(())
}

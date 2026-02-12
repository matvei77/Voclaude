use std::path::Path;
use std::process::Command;
use std::process::Stdio;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use tracing::{debug, info};

use super::{InferenceBackend, TranscriptionResult};
use crate::config::QwenPythonConfig;

pub struct QwenPythonBackend {
    cfg: QwenPythonConfig,
}

impl QwenPythonBackend {
    pub fn new(cfg: QwenPythonConfig) -> Self {
        Self { cfg }
    }

    fn run_command(&self, args: &[String], timeout: Duration) -> Result<std::process::Output> {
        let mut cmd = Command::new(&self.cfg.python_path);
        cmd.args(args);
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child = cmd
            .spawn()
            .with_context(|| format!("failed to spawn {}", self.cfg.python_path.display()))?;

        let started = std::time::Instant::now();
        loop {
            if let Some(_status) = child.try_wait()? {
                let output = child.wait_with_output()?;
                return Ok(output);
            }
            if started.elapsed() > timeout {
                let _ = child.kill();
                let _ = child.wait();
                bail!("qwen python command timed out after {:?}", timeout);
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    fn gpu_check(&self) -> Result<()> {
        if !self.cfg.require_gpu {
            return Ok(());
        }

        let script = "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 42)";
        let output = Command::new(&self.cfg.python_path)
            .args(["-c", script])
            .output()
            .with_context(|| format!("failed to run {}", self.cfg.python_path.display()))?;

        if output.status.success() {
            return Ok(());
        }

        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "CUDA is mandatory but unavailable for qwen backend. stderr={}",
            stderr.trim()
        );
    }
}

impl InferenceBackend for QwenPythonBackend {
    fn name(&self) -> &'static str {
        "qwen_python"
    }

    fn health_check(&self) -> Result<()> {
        if !self.cfg.python_path.exists() {
            bail!("python_path not found: {}", self.cfg.python_path.display());
        }
        if !self.cfg.script_path.exists() {
            bail!("script_path not found: {}", self.cfg.script_path.display());
        }
        self.gpu_check()?;

        let output = Command::new(&self.cfg.python_path)
            .arg(&self.cfg.script_path)
            .arg("--help")
            .output()
            .with_context(|| "failed to run transcribe.py --help")?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!("qwen transcribe.py --help failed: {}", stderr.trim());
        }
        info!("healthcheck passed: qwen python backend");
        Ok(())
    }

    fn transcribe_path(
        &self,
        audio_path: &Path,
        chunk_seconds: f32,
        overlap_seconds: f32,
    ) -> Result<TranscriptionResult> {
        self.gpu_check()?;

        let mut args: Vec<String> = vec![
            self.cfg.script_path.display().to_string(),
            audio_path.display().to_string(),
            "--model".to_string(),
            self.cfg.model.clone(),
            "--device".to_string(),
            self.cfg.device.clone(),
            "--dtype".to_string(),
            self.cfg.dtype.clone(),
            "--max-new-tokens".to_string(),
            self.cfg.max_new_tokens.to_string(),
            "--json".to_string(),
        ];

        if !self.cfg.language.eq_ignore_ascii_case("auto") {
            args.push("--language".to_string());
            args.push(self.cfg.language.clone());
        }

        if chunk_seconds > 0.0 {
            args.push("--chunk-seconds".to_string());
            args.push(format!("{:.3}", chunk_seconds));
        }
        if overlap_seconds > 0.0 {
            args.push("--chunk-overlap-seconds".to_string());
            args.push(format!("{:.3}", overlap_seconds));
        }

        debug!("running qwen command: {} {:?}", self.cfg.python_path.display(), args);

        let output = self.run_command(&args, Duration::from_secs(self.cfg.timeout_seconds))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!("qwen transcription failed: {}", stderr.trim());
        }

        let stdout = String::from_utf8(output.stdout).context("qwen stdout was not utf-8")?;
        let parsed: ScriptResponse = serde_json::from_str(&stdout)
            .with_context(|| "failed to parse qwen json response")?;

        let first = parsed
            .results
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("qwen returned no results"))?;

        Ok(TranscriptionResult {
            backend: self.name().to_string(),
            model: parsed.model,
            language: first.language,
            text: first.text,
            load_seconds: parsed.load_seconds,
            inference_seconds: parsed.inference_seconds,
            chunk_count: first.chunk_count,
        })
    }
}

#[derive(Debug, Deserialize)]
struct ScriptResponse {
    model: String,
    load_seconds: f32,
    inference_seconds: f32,
    results: Vec<ScriptResultItem>,
}

#[derive(Debug, Deserialize)]
struct ScriptResultItem {
    language: Option<String>,
    text: String,
    #[serde(default)]
    chunk_count: Option<usize>,
}

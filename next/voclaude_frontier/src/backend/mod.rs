mod qwen_python;

use std::path::Path;

use anyhow::{bail, Result};
use serde::Serialize;

use crate::config::{BackendConfig, BackendKind};

pub use qwen_python::QwenPythonBackend;

#[derive(Debug, Clone, Serialize)]
pub struct TranscriptionResult {
    pub backend: String,
    pub model: String,
    pub language: Option<String>,
    pub text: String,
    pub load_seconds: f32,
    pub inference_seconds: f32,
    pub chunk_count: Option<usize>,
}

pub trait InferenceBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn health_check(&self) -> Result<()>;
    fn transcribe_path(
        &self,
        audio_path: &Path,
        chunk_seconds: f32,
        overlap_seconds: f32,
    ) -> Result<TranscriptionResult>;
}

pub fn build_backend(cfg: &BackendConfig) -> Result<Box<dyn InferenceBackend>> {
    match cfg.kind {
        BackendKind::QwenPython => {
            let Some(qwen_cfg) = cfg.qwen_python.clone() else {
                bail!("backend.qwen_python config is missing");
            };
            Ok(Box::new(QwenPythonBackend::new(qwen_cfg)))
        }
    }
}

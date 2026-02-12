use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    pub backend: BackendConfig,
    pub chunking: ChunkingConfig,
    pub runtime: RuntimeConfig,
}

impl AppConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("failed to read config {}", path.display()))?;
        let mut cfg: AppConfig = toml::from_str(&content)
            .with_context(|| format!("failed to parse config {}", path.display()))?;

        let base = path.parent().unwrap_or_else(|| Path::new("."));
        cfg.backend.rewrite_relative_paths(base);
        Ok(cfg)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct BackendConfig {
    pub kind: BackendKind,

    #[serde(default)]
    pub qwen_python: Option<QwenPythonConfig>,
}

impl BackendConfig {
    fn rewrite_relative_paths(&mut self, base: &Path) {
        if let Some(qwen) = self.qwen_python.as_mut() {
            qwen.python_path = absolutize(base, &qwen.python_path);
            qwen.script_path = absolutize(base, &qwen.script_path);
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    QwenPython,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QwenPythonConfig {
    pub python_path: PathBuf,
    pub script_path: PathBuf,
    pub model: String,
    pub device: String,
    pub dtype: String,
    pub max_new_tokens: u32,
    pub language: String,
    pub require_gpu: bool,
    pub timeout_seconds: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChunkingConfig {
    pub seconds: f32,
    pub overlap_seconds: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RuntimeConfig {
    #[serde(default)]
    pub write_json: bool,
}

fn absolutize(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

//! Configuration management with sensible defaults.

use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Hotkey to toggle recording (e.g., "F4", "Ctrl+Alt+V")
    pub hotkey: String,

    /// Hotkey to toggle the history window
    #[serde(default = "default_history_hotkey")]
    pub history_hotkey: String,

    /// Language hint for Qwen ASR (None = auto-detect)
    pub language: Option<String>,

    /// Add trailing space after pasted text
    pub add_trailing_space: bool,

    /// Capitalize first letter
    pub capitalize_first: bool,

    /// Unload model after N seconds idle (saves VRAM)
    pub idle_unload_seconds: u64,

    /// Show notifications
    pub show_notifications: bool,

    /// Maximum number of history entries to retain
    #[serde(default = "default_history_max_entries")]
    pub history_max_entries: usize,

    /// Enable GPU acceleration when available
    #[serde(default = "default_use_gpu")]
    pub use_gpu: bool,

    /// Optional path to python executable used for Qwen backend.
    /// If empty, runtime attempts auto-discovery.
    #[serde(default)]
    pub qwen_python_path: Option<String>,

    /// Optional path to transcribe.py (qwen smoke script).
    /// If empty, runtime attempts auto-discovery.
    #[serde(default)]
    pub qwen_script_path: Option<String>,

    /// Qwen model id.
    #[serde(default = "default_qwen_model")]
    pub qwen_model: String,

    /// Torch dtype passed to qwen script.
    #[serde(default = "default_qwen_dtype")]
    pub qwen_dtype: String,

    /// Device passed to qwen script.
    /// Typical values: auto, cuda:0, cpu.
    #[serde(default = "default_qwen_device")]
    pub qwen_device: String,

    /// Max new tokens for model generation.
    #[serde(default = "default_qwen_max_new_tokens")]
    pub qwen_max_new_tokens: u32,

    /// Manual chunk size in seconds (0 disables).
    #[serde(default = "default_qwen_chunk_seconds")]
    pub qwen_chunk_seconds: f32,

    /// Chunk overlap in seconds.
    #[serde(default = "default_qwen_chunk_overlap_seconds")]
    pub qwen_chunk_overlap_seconds: f32,

    /// Hard timeout for one transcription request.
    #[serde(default = "default_qwen_timeout_seconds")]
    pub qwen_timeout_seconds: u64,

    /// Fail request if CUDA is unavailable.
    #[serde(default = "default_qwen_require_gpu")]
    pub qwen_require_gpu: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            hotkey: "F4".to_string(),
            history_hotkey: default_history_hotkey(),
            language: None,
            add_trailing_space: true,
            capitalize_first: true,
            idle_unload_seconds: 300, // 5 minutes
            show_notifications: true,
            history_max_entries: default_history_max_entries(),
            use_gpu: true,
            qwen_python_path: None,
            qwen_script_path: None,
            qwen_model: default_qwen_model(),
            qwen_dtype: default_qwen_dtype(),
            qwen_device: default_qwen_device(),
            qwen_max_new_tokens: default_qwen_max_new_tokens(),
            qwen_chunk_seconds: default_qwen_chunk_seconds(),
            qwen_chunk_overlap_seconds: default_qwen_chunk_overlap_seconds(),
            qwen_timeout_seconds: default_qwen_timeout_seconds(),
            qwen_require_gpu: default_qwen_require_gpu(),
        }
    }
}

impl Config {
    /// Get the config directory
    pub fn config_dir() -> Option<PathBuf> {
        ProjectDirs::from("com", "voclaude", "VoclaudeQwenRuntime")
            .map(|dirs| dirs.config_dir().to_path_buf())
    }

    /// Get the config file path
    pub fn config_path() -> Option<PathBuf> {
        Self::config_dir().map(|dir| dir.join("config.toml"))
    }

    /// Load config from disk, or create default
    pub fn load() -> Result<Self, Box<dyn std::error::Error>> {
        let path = Self::config_path().ok_or("Could not determine config path")?;

        if path.exists() {
            let contents = std::fs::read_to_string(&path)?;
            let config: Config = toml::from_str(&contents)?;
            Ok(config)
        } else {
            let config = Config::default();
            config.save()?;
            Ok(config)
        }
    }

    /// Save config to disk
    pub fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        let path = Self::config_path().ok_or("Could not determine config path")?;

        // Ensure directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let contents = toml::to_string_pretty(self)?;
        std::fs::write(&path, contents)?;
        Ok(())
    }
}

fn default_use_gpu() -> bool {
    true
}

fn default_history_max_entries() -> usize {
    500
}

fn default_history_hotkey() -> String {
    "Ctrl+Shift+H".to_string()
}

fn default_qwen_model() -> String {
    "Qwen/Qwen3-ASR-1.7B".to_string()
}

fn default_qwen_dtype() -> String {
    "bfloat16".to_string()
}

fn default_qwen_device() -> String {
    "cuda:0".to_string()
}

fn default_qwen_max_new_tokens() -> u32 {
    2048
}

fn default_qwen_chunk_seconds() -> f32 {
    60.0
}

fn default_qwen_chunk_overlap_seconds() -> f32 {
    2.0
}

fn default_qwen_timeout_seconds() -> u64 {
    7200
}

fn default_qwen_require_gpu() -> bool {
    true
}

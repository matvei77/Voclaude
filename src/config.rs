//! Configuration management with sensible defaults.

use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tracing::{info, warn};

/// Return the canonical ProjectDirs for the application.
pub fn project_dirs() -> Option<ProjectDirs> {
    ProjectDirs::from("com", "voclaude", "Voclaude")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Hotkey to toggle recording (e.g., "F4", "Ctrl+Alt+V")
    pub hotkey: String,

    /// Hotkey to toggle the history window
    #[serde(default = "default_history_hotkey")]
    pub history_hotkey: String,

    /// Language hint for ASR (None = auto-detect)
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

    /// Model id (HuggingFace repo id).
    #[serde(default = "default_model", alias = "qwen_model")]
    pub model: String,

    /// Optional path to local safetensors model directory.
    /// If not set, falls back to HF cache or downloads from HuggingFace.
    #[serde(default, alias = "qwen_model_path")]
    pub model_path: Option<String>,

    /// Max new tokens for model generation.
    #[serde(default = "default_max_new_tokens", alias = "qwen_max_new_tokens")]
    pub max_new_tokens: u32,

    /// Fail request if CUDA is unavailable.
    #[serde(default = "default_require_gpu", alias = "qwen_require_gpu")]
    pub require_gpu: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            hotkey: "F4".to_string(),
            history_hotkey: default_history_hotkey(),
            language: None,
            add_trailing_space: true,
            capitalize_first: true,
            idle_unload_seconds: 30,
            show_notifications: true,
            history_max_entries: default_history_max_entries(),
            use_gpu: true,
            model: default_model(),
            model_path: None,
            max_new_tokens: default_max_new_tokens(),
            require_gpu: false,
        }
    }
}

impl Config {
    /// Get the config directory
    pub fn config_dir() -> Option<PathBuf> {
        project_dirs().map(|dirs| dirs.config_dir().to_path_buf())
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

    /// Extract a short display name from the model id.
    pub fn model_display_name(&self) -> String {
        // "Qwen/Qwen3-ASR-1.7B" → "qwen3-asr-1.7b"
        self.model
            .rsplit('/')
            .next()
            .unwrap_or(&self.model)
            .to_lowercase()
    }

    /// Copy config/data files from the legacy `VoclaudeQwenRuntime` path to the
    /// new `Voclaude` path.  Copies (not moves) for safety — the old directory
    /// is left intact.
    pub fn migrate_from_legacy() {
        let legacy = match ProjectDirs::from("com", "voclaude", "VoclaudeQwenRuntime") {
            Some(dirs) => dirs,
            None => return,
        };
        let current = match project_dirs() {
            Some(dirs) => dirs,
            None => return,
        };

        // If the new config already exists, nothing to migrate.
        let new_config = current.config_dir().join("config.toml");
        if new_config.exists() {
            return;
        }

        // Config files
        let legacy_config = legacy.config_dir().join("config.toml");
        if legacy_config.exists() {
            if let Err(e) = copy_file(&legacy_config, &new_config) {
                warn!("Migration: failed to copy config.toml: {}", e);
            } else {
                info!("Migration: copied config.toml from legacy path");
            }
        }

        // Data files
        let data_files = ["history.json", "session.json", "audio.f32"];
        for name in &data_files {
            let src = legacy.data_dir().join(name);
            let dst = current.data_dir().join(name);
            if src.exists() && !dst.exists() {
                if let Err(e) = copy_file(&src, &dst) {
                    warn!("Migration: failed to copy {}: {}", name, e);
                } else {
                    info!("Migration: copied {}", name);
                }
            }
        }

        // Transcripts directory
        let src_transcripts = legacy.data_dir().join("transcripts");
        let dst_transcripts = current.data_dir().join("transcripts");
        if src_transcripts.is_dir() && !dst_transcripts.exists() {
            if let Err(e) = copy_dir(&src_transcripts, &dst_transcripts) {
                warn!("Migration: failed to copy transcripts/: {}", e);
            } else {
                info!("Migration: copied transcripts/");
            }
        }
    }
}

fn copy_file(src: &std::path::Path, dst: &std::path::Path) -> std::io::Result<()> {
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::copy(src, dst)?;
    Ok(())
}

fn copy_dir(src: &std::path::Path, dst: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if file_type.is_file() {
            std::fs::copy(&src_path, &dst_path)?;
        } else if file_type.is_dir() {
            copy_dir(&src_path, &dst_path)?;
        }
    }
    Ok(())
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

fn default_model() -> String {
    "Qwen/Qwen3-ASR-1.7B".to_string()
}

fn default_max_new_tokens() -> u32 {
    2048
}

fn default_require_gpu() -> bool {
    false
}

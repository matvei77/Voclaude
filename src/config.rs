//! Configuration management with sensible defaults.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use directories::ProjectDirs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Hotkey to toggle recording (e.g., "F4", "Ctrl+Alt+V")
    pub hotkey: String,

    /// Language for transcription (None = auto-detect)
    /// Note: Parakeet is English-only, this is for future multi-language support
    pub language: Option<String>,

    /// Add trailing space after pasted text
    pub add_trailing_space: bool,

    /// Capitalize first letter
    pub capitalize_first: bool,

    /// Unload model after N seconds idle (saves VRAM)
    pub idle_unload_seconds: u64,

    /// Show notifications
    pub show_notifications: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            hotkey: "F4".to_string(),
            language: None, // Parakeet is English-only for now
            add_trailing_space: true,
            capitalize_first: true,
            idle_unload_seconds: 300, // 5 minutes
            show_notifications: true,
        }
    }
}

impl Config {
    /// Get the config directory
    pub fn config_dir() -> Option<PathBuf> {
        ProjectDirs::from("com", "voclaude", "Voclaude")
            .map(|dirs| dirs.config_dir().to_path_buf())
    }

    /// Get the models directory
    pub fn models_dir() -> Option<PathBuf> {
        ProjectDirs::from("com", "voclaude", "Voclaude")
            .map(|dirs| dirs.data_dir().join("models"))
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

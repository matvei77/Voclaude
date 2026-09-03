//! Configuration management with sensible defaults.

use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tracing::{info, warn};

pub const MODEL_QWEN3_ASR_0_6B: &str = "Qwen/Qwen3-ASR-0.6B";
pub const MODEL_QWEN3_ASR_1_7B: &str = "Qwen/Qwen3-ASR-1.7B";

/// Return the canonical ProjectDirs for the application.
pub fn project_dirs() -> Option<ProjectDirs> {
    ProjectDirs::from("com", "voclaude", "Voclaude")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Schema version — bump when breaking changes are made to the config format.
    #[serde(default = "default_config_version")]
    pub config_version: u32,

    /// Hotkey to toggle recording (e.g., "F4", "Ctrl+Alt+V")
    #[serde(default = "default_hotkey")]
    pub hotkey: String,

    /// Hotkey to toggle the history window
    #[serde(default = "default_history_hotkey")]
    pub history_hotkey: String,

    /// Language hint for ASR (None = auto-detect)
    #[serde(default)]
    pub language: Option<String>,

    /// Add trailing space after pasted text
    #[serde(default = "default_add_trailing_space")]
    pub add_trailing_space: bool,

    /// Capitalize first letter
    #[serde(default = "default_capitalize_first")]
    pub capitalize_first: bool,

    /// Unload model after N seconds idle (saves VRAM)
    #[serde(default = "default_idle_unload_seconds")]
    pub idle_unload_seconds: u64,

    /// Show notifications
    #[serde(default = "default_show_notifications")]
    pub show_notifications: bool,

    /// Maximum number of history entries to retain
    #[serde(default = "default_history_max_entries")]
    pub history_max_entries: usize,

    /// Enable GPU acceleration when available
    #[serde(default = "default_use_gpu")]
    pub use_gpu: bool,

    /// User-facing model tier. Used when `model` is not explicitly set.
    #[serde(default = "default_model_tier")]
    pub model_tier: String,

    /// Select a conservative model automatically for the selected hardware path.
    #[serde(default = "default_auto_select_model")]
    pub auto_select_model: bool,

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

    /// Reduce the generation cap for short recordings to lower memory use.
    #[serde(default = "default_adaptive_max_new_tokens")]
    pub adaptive_max_new_tokens: bool,

    /// Maximum audio chunk size before splitting at silence boundaries.
    #[serde(default = "default_max_chunk_seconds")]
    pub max_chunk_seconds: u32,

    /// Fail request if CUDA is unavailable.
    #[serde(default = "default_require_gpu", alias = "qwen_require_gpu")]
    pub require_gpu: bool,

    /// Number of audio recordings to retain on disk (0 = delete immediately).
    /// Oldest recordings beyond this count are deleted after successful transcription.
    /// Failed transcriptions always keep their audio file.
    #[serde(default = "default_audio_retention_count")]
    pub audio_retention_count: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            config_version: default_config_version(),
            hotkey: "F4".to_string(),
            history_hotkey: default_history_hotkey(),
            language: None,
            add_trailing_space: true,
            capitalize_first: true,
            idle_unload_seconds: 30,
            show_notifications: true,
            history_max_entries: default_history_max_entries(),
            use_gpu: true,
            model_tier: default_model_tier(),
            auto_select_model: default_auto_select_model(),
            model: default_model(),
            model_path: None,
            max_new_tokens: default_max_new_tokens(),
            adaptive_max_new_tokens: default_adaptive_max_new_tokens(),
            max_chunk_seconds: default_max_chunk_seconds(),
            require_gpu: false,
            audio_retention_count: default_audio_retention_count(),
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

    /// Load config from disk, or create default. Validates all fields.
    pub fn load() -> Result<Self, Box<dyn std::error::Error>> {
        let path = Self::config_path().ok_or("Could not determine config path")?;

        let config = if path.exists() {
            let contents = std::fs::read_to_string(&path)?;
            let mut has_explicit_model = false;
            let mut has_model_tier = false;
            // P-3: Warn about unknown fields by round-tripping through Value
            if let Ok(table) = contents.parse::<toml::Table>() {
                has_explicit_model = table.contains_key("model") || table.contains_key("qwen_model");
                has_model_tier = table.contains_key("model_tier");
                let known_fields = [
                    "config_version", "hotkey", "history_hotkey", "language",
                    "add_trailing_space", "capitalize_first", "idle_unload_seconds",
                    "show_notifications", "history_max_entries", "use_gpu",
                    "model_tier", "auto_select_model", "model", "model_path",
                    "max_new_tokens", "adaptive_max_new_tokens", "max_chunk_seconds",
                    "require_gpu", "audio_retention_count",
                    // Legacy aliases
                    "qwen_model", "qwen_model_path", "qwen_max_new_tokens", "qwen_require_gpu",
                ];
                for key in table.keys() {
                    if !known_fields.contains(&key.as_str()) {
                        warn!("Unknown config field '{}' (possible typo)", key);
                    }
                }
            }
            let mut config: Config = toml::from_str(&contents)?;
            if !has_explicit_model && (has_model_tier || config.auto_select_model) {
                config.model = config.model_for_current_tier();
            }
            config
        } else {
            let config = Config::default();
            config.save()?;
            config
        };

        config.validate()?;
        Ok(config)
    }

    /// Validate all config fields. Returns Err with a descriptive message
    /// on the first invalid value.
    pub fn validate(&self) -> Result<(), Box<dyn std::error::Error>> {
        // P-4: Check config_version for forward compatibility
        if self.config_version > default_config_version() {
            warn!(
                "Config version {} is newer than supported version {}; some fields may be ignored",
                self.config_version, default_config_version()
            );
        }
        if self.hotkey.trim().is_empty() {
            return Err("hotkey cannot be empty".into());
        }
        if self.history_hotkey.trim().is_empty() {
            return Err("history_hotkey cannot be empty".into());
        }
        if self.idle_unload_seconds == 0 {
            return Err("idle_unload_seconds must be > 0".into());
        }
        if self.history_max_entries == 0 {
            return Err("history_max_entries must be > 0".into());
        }
        if self.max_new_tokens == 0 || self.max_new_tokens > 8192 {
            return Err("max_new_tokens must be between 1 and 8192".into());
        }
        if self.max_chunk_seconds < 10 || self.max_chunk_seconds > 300 {
            return Err("max_chunk_seconds must be between 10 and 300".into());
        }
        if model_for_tier(&self.model_tier, self.use_gpu).is_none() {
            return Err(
                "model_tier must be one of: auto, fast, small, medium, balanced, best, large, accurate"
                    .into(),
            );
        }
        if self.model.trim().is_empty() {
            return Err("model cannot be empty".into());
        }
        Ok(())
    }

    /// Save config to disk atomically (write-temp-then-rename).
    pub fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        let path = Self::config_path().ok_or("Could not determine config path")?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let contents = toml::to_string_pretty(self)?;
        // P-2: Use unique temp path with PID+timestamp to avoid collision on crash
        let pid = std::process::id();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let temp_name = format!("config.toml.{}-{}.tmp", pid, ts);
        let temp_path = path.with_file_name(temp_name);
        let mut file = std::fs::File::create(&temp_path)?;
        std::io::Write::write_all(&mut file, contents.as_bytes())?;
        file.sync_all()?;
        drop(file);
        // Retry rename for Windows antivirus/indexer
        let mut last_err = None;
        for attempt in 0..5u32 {
            match std::fs::rename(&temp_path, &path) {
                Ok(()) => return Ok(()),
                Err(err) => {
                    if attempt < 4 {
                        std::thread::sleep(std::time::Duration::from_millis(50 * (1 << attempt)));
                    }
                    last_err = Some(err);
                }
            }
        }
        let _ = std::fs::remove_file(&temp_path);
        Err(last_err.unwrap().into())
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

    pub fn model_for_current_tier(&self) -> String {
        model_for_tier(&self.model_tier, self.use_gpu)
            .unwrap_or(MODEL_QWEN3_ASR_1_7B)
            .to_string()
    }

    pub fn set_model_tier(&mut self, tier: impl Into<String>) {
        self.model_tier = tier.into();
        self.model = self.model_for_current_tier();
    }

    pub fn model_tier_summary(&self) -> &'static str {
        model_tier_summary(&self.model_tier, self.use_gpu)
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
                // Validate the migrated config
                if let Ok(contents) = std::fs::read_to_string(&new_config) {
                    if toml::from_str::<Config>(&contents).is_err() {
                        warn!("Migration: copied config.toml is invalid; removing");
                        let _ = std::fs::remove_file(&new_config);
                    }
                }
            }
        }

        // Data files
        let data_files = ["history.json", "session.json", "audio.f32"];
        for name in &data_files {
            let src = legacy.data_dir().join(name);
            let dst = current.data_dir().join(name);
            if src.exists() && !dst.exists() {
                // P-5: Validate JSON files before copying to avoid propagating corruption
                if *name == "history.json" || *name == "session.json" {
                    if let Ok(contents) = std::fs::read_to_string(&src) {
                        if serde_json::from_str::<serde_json::Value>(&contents).is_err() {
                            warn!("Migration: skipping corrupt {}", name);
                            continue;
                        }
                    }
                }
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

fn default_config_version() -> u32 {
    1
}

fn default_use_gpu() -> bool {
    true
}

fn default_model_tier() -> String {
    "best".to_string()
}

fn default_auto_select_model() -> bool {
    false
}

fn default_history_max_entries() -> usize {
    500
}

fn default_history_hotkey() -> String {
    "Ctrl+Shift+H".to_string()
}

fn default_model() -> String {
    MODEL_QWEN3_ASR_1_7B.to_string()
}

fn default_max_new_tokens() -> u32 {
    2048
}

fn default_adaptive_max_new_tokens() -> bool {
    true
}

fn default_max_chunk_seconds() -> u32 {
    300
}

fn default_require_gpu() -> bool {
    false
}

fn default_hotkey() -> String {
    "F4".to_string()
}

fn default_add_trailing_space() -> bool {
    true
}

fn default_capitalize_first() -> bool {
    true
}

fn default_idle_unload_seconds() -> u64 {
    30
}

fn default_show_notifications() -> bool {
    true
}

fn default_audio_retention_count() -> usize {
    10
}

pub fn model_for_tier(tier: &str, use_gpu: bool) -> Option<&'static str> {
    match tier.trim().to_ascii_lowercase().as_str() {
        "auto" => Some(if use_gpu {
            MODEL_QWEN3_ASR_1_7B
        } else {
            MODEL_QWEN3_ASR_0_6B
        }),
        "fast" | "small" | "light" | "cpu" => Some(MODEL_QWEN3_ASR_0_6B),
        "medium" | "balanced" => Some(MODEL_QWEN3_ASR_0_6B),
        "best" | "large" | "accurate" => Some(MODEL_QWEN3_ASR_1_7B),
        _ => None,
    }
}

pub fn model_tier_summary(tier: &str, use_gpu: bool) -> &'static str {
    match model_for_tier(tier, use_gpu) {
        Some(MODEL_QWEN3_ASR_0_6B) => {
            "Qwen3-ASR-0.6B: smaller download and much lower memory use; best default for CPU and low-end laptop GPUs."
        }
        Some(MODEL_QWEN3_ASR_1_7B) => {
            "Qwen3-ASR-1.7B: highest quality; best for NVIDIA GPUs with comfortable VRAM."
        }
        _ => "Unknown model tier.",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        let config = Config::default();
        config.validate().expect("default config should be valid");
        assert_eq!(config.config_version, 1);
        assert_eq!(config.hotkey, "F4");
    }

    #[test]
    fn parse_minimal_toml() {
        let toml_str = r#"
            hotkey = "F5"
            add_trailing_space = false
            capitalize_first = false
            idle_unload_seconds = 60
            show_notifications = false
        "#;
        let config: Config = toml::from_str(toml_str).expect("should parse");
        assert_eq!(config.hotkey, "F5");
        assert_eq!(config.idle_unload_seconds, 60);
        // Defaults should fill in
        assert_eq!(config.config_version, 1);
        assert_eq!(config.history_max_entries, 500);
        assert_eq!(config.model_tier, "best");
        config.validate().expect("should be valid");
    }

    #[test]
    fn parse_legacy_aliases() {
        let toml_str = r#"
            hotkey = "F4"
            add_trailing_space = true
            capitalize_first = true
            idle_unload_seconds = 30
            show_notifications = true
            qwen_model = "Qwen/Qwen3-ASR-0.6B"
            qwen_model_path = "/some/path"
            qwen_max_new_tokens = 1024
            qwen_require_gpu = true
        "#;
        let config: Config = toml::from_str(toml_str).expect("should parse legacy aliases");
        assert_eq!(config.model, "Qwen/Qwen3-ASR-0.6B");
        assert_eq!(config.model_path.as_deref(), Some("/some/path"));
        assert_eq!(config.max_new_tokens, 1024);
        assert!(config.require_gpu);
    }

    #[test]
    fn validate_rejects_empty_hotkey() {
        let mut config = Config::default();
        config.hotkey = "".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_idle() {
        let mut config = Config::default();
        config.idle_unload_seconds = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_max_tokens() {
        let mut config = Config::default();
        config.max_new_tokens = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_rejects_huge_max_tokens() {
        let mut config = Config::default();
        config.max_new_tokens = 10000;
        assert!(config.validate().is_err());
    }

    #[test]
    fn model_display_name_extracts_basename() {
        let config = Config::default();
        assert_eq!(config.model_display_name(), "qwen3-asr-1.7b");
    }

    #[test]
    fn model_tier_selects_small_model() {
        let mut config = Config::default();
        config.set_model_tier("medium");
        assert_eq!(config.model, MODEL_QWEN3_ASR_0_6B);
        config.validate().expect("medium tier should be valid");
    }

    #[test]
    fn auto_tier_selects_cpu_model_without_gpu() {
        let mut config = Config::default();
        config.use_gpu = false;
        config.set_model_tier("auto");
        assert_eq!(config.model, MODEL_QWEN3_ASR_0_6B);
    }
}

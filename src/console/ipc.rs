//! Messages between the tray app (core) and the console window process.
//!
//! The core starts `voclaude console --ipc` with piped stdin and stdout. Each
//! message is one JSON object on one line: core -> console on the console's
//! stdin, console -> core on its stdout. The console reads everything else
//! (history, settings, logs) from the files the core writes.

use serde::{Deserialize, Serialize};

/// What the core is doing, for the console's status strip.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Status {
    /// "Idle", "Recording", "Transcribing" or "Recovering".
    pub state: String,
    /// Engine label from the worker once a model is loaded, else from config.
    pub model: String,
    pub use_gpu: bool,
    /// False until the worker has reported a loaded model.
    pub model_loaded: bool,
    pub history_count: usize,
    pub input_device: Option<String>,
    pub last_duration_ms: Option<u64>,
    /// Audio seconds per processing second for the last dictation.
    pub last_speed: Option<f32>,
    pub last_message: Option<String>,
    /// Set while `last_message` reports a failure.
    pub last_message_is_error: bool,
    /// Port of the local API when it is listening.
    pub api_port: Option<u16>,
    /// Why the API is not listening although it is enabled.
    pub api_error: Option<String>,
    /// The saved config.toml could not be applied (or loaded at start).
    pub config_error: Option<String>,
    pub record_hotkey: String,
    pub console_hotkey: String,
    /// "Recover Last Recording" has work to do.
    pub recoverable: bool,
}

/// Core -> console.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "msg", rename_all = "snake_case")]
pub enum ToConsole {
    Status(Status),
    /// Microphone level 0..1, only while recording (at most ~20 per second).
    Level { level: f32 },
    /// history.json changed on disk.
    HistoryChanged,
    /// Bring the window to the front.
    Focus,
}

/// Console -> core.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum FromConsole {
    /// Same as the record hotkey.
    ToggleRecording,
    /// Remove one history entry (and its recording).
    DeleteHistory { id: String },
    /// Same as the tray item "Recover Last Recording".
    RecoverLast,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_are_single_json_lines() {
        let line = serde_json::to_string(&ToConsole::Level { level: 0.5 }).unwrap();
        assert_eq!(line, r#"{"msg":"level","level":0.5}"#);
        let line = serde_json::to_string(&FromConsole::DeleteHistory { id: "1-2".into() }).unwrap();
        assert_eq!(line, r#"{"cmd":"delete_history","id":"1-2"}"#);
        let status = ToConsole::Status(Status { last_message: Some("a\nb".into()), ..Default::default() });
        assert!(!serde_json::to_string(&status).unwrap().contains('\n'));
        let back: FromConsole = serde_json::from_str(r#"{"cmd":"toggle_recording"}"#).unwrap();
        assert!(matches!(back, FromConsole::ToggleRecording));
    }
}

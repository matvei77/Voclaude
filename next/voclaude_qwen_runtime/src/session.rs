//! Recording session metadata storage.

use crate::history::AudioMetadata;
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::warn;

static SESSION_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SessionState {
    Recording,
    Transcribing,
    Completed,
    Failed,
    Aborted,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordingSession {
    pub id: String,
    pub started_at_ms: u64,
    pub ended_at_ms: Option<u64>,
    pub state: SessionState,
    pub audio_path: Option<String>,
    pub audio: Option<AudioMetadata>,
    pub transcript: Option<String>,
    pub error: Option<String>,
    pub history_entry_id: Option<String>,
}

impl RecordingSession {
    pub fn is_recoverable(&self) -> bool {
        matches!(self.state, SessionState::Recording | SessionState::Transcribing)
    }

    pub fn audio_path(&self) -> Option<PathBuf> {
        self.audio_path.as_ref().map(PathBuf::from)
    }
}

#[derive(Debug)]
pub struct SessionStore {
    path: PathBuf,
    current: Option<RecordingSession>,
}

impl SessionStore {
    pub fn load() -> Result<Self, Box<dyn std::error::Error>> {
        let path = Self::session_path()?;
        let current = if path.exists() {
            match std::fs::read_to_string(&path) {
                Ok(contents) => match serde_json::from_str::<RecordingSession>(&contents) {
                    Ok(session) => Some(session),
                    Err(err) => {
                        warn!("Failed to parse session file: {}", err);
                        let _ = backup_corrupt(&path, &contents);
                        None
                    }
                },
                Err(err) => {
                    warn!("Failed to read session file: {}", err);
                    None
                }
            }
        } else {
            None
        };

        Ok(Self { path, current })
    }

    pub fn current(&self) -> Option<&RecordingSession> {
        self.current.as_ref()
    }

    pub fn start(&mut self) -> Result<RecordingSession, Box<dyn std::error::Error>> {
        let session = RecordingSession {
            id: new_session_id(),
            started_at_ms: now_ms(),
            ended_at_ms: None,
            state: SessionState::Recording,
            audio_path: Some(Self::audio_path()?.to_string_lossy().to_string()),
            audio: None,
            transcript: None,
            error: None,
            history_entry_id: None,
        };
        self.current = Some(session.clone());
        self.persist()?;
        Ok(session)
    }

    pub fn mark_transcribing(
        &mut self,
        audio: AudioMetadata,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(session) = self.current.as_mut() {
            session.state = SessionState::Transcribing;
            session.audio = Some(audio);
            session.ended_at_ms = Some(now_ms());
            self.persist()?;
        }
        Ok(())
    }

    pub fn mark_completed(
        &mut self,
        transcript: String,
        history_entry_id: Option<String>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(session) = self.current.as_mut() {
            session.state = SessionState::Completed;
            session.transcript = Some(transcript);
            session.history_entry_id = history_entry_id;
            session.ended_at_ms = Some(now_ms());
            self.persist()?;
        }
        Ok(())
    }

    pub fn mark_failed(&mut self, error: String) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(session) = self.current.as_mut() {
            session.state = SessionState::Failed;
            session.error = Some(error);
            session.ended_at_ms = Some(now_ms());
            self.persist()?;
        }
        Ok(())
    }

    pub fn mark_aborted(&mut self, reason: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(session) = self.current.as_mut() {
            session.state = SessionState::Aborted;
            session.error = reason;
            session.ended_at_ms = Some(now_ms());
            self.persist()?;
        }
        Ok(())
    }

    fn session_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
        ProjectDirs::from("com", "voclaude", "VoclaudeQwenRuntime")
            .map(|dirs| dirs.data_dir().join("session.json"))
            .ok_or_else(|| "Could not determine session path".into())
    }

    fn audio_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
        ProjectDirs::from("com", "voclaude", "VoclaudeQwenRuntime")
            .map(|dirs| dirs.data_dir().join("audio.f32"))
            .ok_or_else(|| "Could not determine audio path".into())
    }

    fn persist(&self) -> Result<(), Box<dyn std::error::Error>> {
        let Some(session) = self.current.as_ref() else {
            return Ok(());
        };
        let contents = serde_json::to_string_pretty(session)?;
        write_atomic(&self.path, &contents)?;
        Ok(())
    }
}

fn new_session_id() -> String {
    let counter = SESSION_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{}-{}", now_ms(), counter)
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or_default()
}

fn write_atomic(path: &Path, contents: &str) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temp_path = path.with_extension("json.tmp");
    let mut file = std::fs::File::create(&temp_path)?;
    file.write_all(contents.as_bytes())?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(&temp_path, path)?;
    Ok(())
}

fn backup_corrupt(path: &Path, contents: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let backup_path = path.with_file_name(format!("session.json.corrupt-{}", now_ms()));
    std::fs::write(&backup_path, contents)?;
    Ok(backup_path)
}

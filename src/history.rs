//! Transcription history storage and retention.

use crossbeam_channel::Sender;
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, warn};

static ENTRY_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioMetadata {
    pub sample_rate: u32,
    pub sample_count: usize,
    pub duration_ms: u64,
}

impl AudioMetadata {
    pub fn from_samples(sample_count: usize, sample_rate: u32) -> Self {
        let duration_ms = (sample_count as u64 * 1000) / sample_rate as u64;
        Self {
            sample_rate,
            sample_count,
            duration_ms,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub id: String,
    pub created_at_ms: u64,
    pub text: String,
    pub audio: Option<AudioMetadata>,
}

impl HistoryEntry {
    fn new(text: String, audio: Option<AudioMetadata>) -> Self {
        let created_at_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_millis() as u64)
            .unwrap_or_default();
        let counter = ENTRY_COUNTER.fetch_add(1, Ordering::Relaxed);
        let id = format!("{}-{}", created_at_ms, counter);
        Self {
            id,
            created_at_ms,
            text,
            audio,
        }
    }
}

#[derive(Debug)]
pub struct HistoryStore {
    entries: Vec<HistoryEntry>,
    path: PathBuf,
    max_entries: usize,
    update_tx: Sender<HistoryEntry>,
}

impl HistoryStore {
    pub fn load(
        max_entries: usize,
        update_tx: Sender<HistoryEntry>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let path = Self::history_path()?;
        let entries = if path.exists() {
            match std::fs::read_to_string(&path) {
                Ok(contents) => match serde_json::from_str::<Vec<HistoryEntry>>(&contents) {
                    Ok(entries) => entries,
                    Err(err) => {
                        warn!("Failed to parse history file: {}", err);
                        Vec::new()
                    }
                },
                Err(err) => {
                    warn!("Failed to read history file: {}", err);
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        };

        let mut store = Self {
            entries,
            path,
            max_entries: max_entries.max(1),
            update_tx,
        };
        store.apply_retention();
        store.persist()?;
        Ok(store)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn entries(&self) -> &[HistoryEntry] {
        &self.entries
    }

    pub fn append(
        &mut self,
        text: String,
        audio: Option<AudioMetadata>,
    ) -> Result<HistoryEntry, Box<dyn std::error::Error>> {
        let entry = HistoryEntry::new(text, audio);
        self.entries.push(entry.clone());
        self.apply_retention();
        self.persist()?;
        if let Err(err) = self.update_tx.try_send(entry.clone()) {
            debug!("Dropping history update: {}", err);
        }
        Ok(entry)
    }

    fn apply_retention(&mut self) {
        if self.entries.len() > self.max_entries {
            let excess = self.entries.len() - self.max_entries;
            self.entries.drain(0..excess);
        }
    }

    fn persist(&self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let contents = serde_json::to_string_pretty(&self.entries)?;
        std::fs::write(&self.path, contents)?;
        Ok(())
    }

    fn history_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
        ProjectDirs::from("com", "voclaude", "Voclaude")
            .map(|dirs| dirs.data_dir().join("history.json"))
            .ok_or_else(|| "Could not determine history path".into())
    }
}

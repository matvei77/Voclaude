//! Transcription history storage and retention.

use crossbeam_channel::Sender;
use crate::config::project_dirs;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};
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
        let (entries, needs_persist) = Self::read_entries(&path)?;

        let mut store = Self {
            entries,
            path,
            max_entries: max_entries.max(1),
            update_tx,
        };
        let retention_applied = store.apply_retention();
        if needs_persist || retention_applied {
            store.persist()?;
        }
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

    fn apply_retention(&mut self) -> bool {
        if self.entries.len() > self.max_entries {
            let excess = self.entries.len() - self.max_entries;
            self.entries.drain(0..excess);
            return true;
        }
        false
    }

    fn persist(&self) -> Result<(), Box<dyn std::error::Error>> {
        let contents = serde_json::to_string_pretty(&self.entries)?;
        write_atomic(&self.path, &contents)?;
        Ok(())
    }

    fn history_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
        project_dirs()
            .map(|dirs| dirs.data_dir().join("history.json"))
            .ok_or_else(|| "Could not determine history path".into())
    }

    fn read_entries(
        path: &Path,
    ) -> Result<(Vec<HistoryEntry>, bool), Box<dyn std::error::Error>> {
        if !path.exists() {
            return Ok((Vec::new(), true));
        }

        let contents = match std::fs::read_to_string(path) {
            Ok(contents) => contents,
            Err(err) => {
                warn!("Failed to read history file: {}", err);
                return Ok((Vec::new(), false));
            }
        };

        match serde_json::from_str::<Vec<HistoryEntry>>(&contents) {
            Ok(entries) => Ok((entries, false)),
            Err(err) => {
                warn!("Failed to parse history file: {}", err);
                let _ = backup_corrupt_history(path, &contents);
                let recovered = recover_entries(&contents);
                if !recovered.is_empty() {
                    warn!("Recovered {} entries from corrupt history", recovered.len());
                }
                Ok((recovered, true))
            }
        }
    }
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

    // On Windows, rename can fail with ACCESS_DENIED if the target is
    // held by antivirus or indexing. Retry a few times with backoff.
    let mut last_err = None;
    for attempt in 0..5u32 {
        match std::fs::rename(&temp_path, path) {
            Ok(()) => return Ok(()),
            Err(err) => {
                if attempt < 4 {
                    std::thread::sleep(std::time::Duration::from_millis(50 * (1 << attempt)));
                }
                last_err = Some(err);
            }
        }
    }
    Err(last_err.unwrap().into())
}

fn backup_corrupt_history(
    path: &Path,
    contents: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or_default();
    let backup_path = path.with_file_name(format!("history.json.corrupt-{}", ts));
    std::fs::write(&backup_path, contents)?;
    Ok(backup_path)
}

fn recover_entries(contents: &str) -> Vec<HistoryEntry> {
    let mut entries = Vec::new();
    let mut in_string = false;
    let mut escape = false;
    let mut depth: usize = 0;
    let mut start: Option<usize> = None;

    for (idx, ch) in contents.char_indices() {
        if in_string {
            if escape {
                escape = false;
                continue;
            }
            if ch == '\\' {
                escape = true;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => {
                if depth == 0 {
                    start = Some(idx);
                }
                depth += 1;
            }
            '}' => {
                if depth > 0 {
                    depth -= 1;
                }
                if depth == 0 {
                    if let Some(start_idx) = start.take() {
                        let slice = &contents[start_idx..=idx];
                        if let Ok(entry) = serde_json::from_str::<HistoryEntry>(slice) {
                            entries.push(entry);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    entries
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn history_entry_has_unique_id() {
        let e1 = HistoryEntry::new("hello".to_string(), None);
        let e2 = HistoryEntry::new("world".to_string(), None);
        assert_ne!(e1.id, e2.id);
    }

    #[test]
    fn audio_metadata_duration() {
        let meta = AudioMetadata::from_samples(16000, 16000);
        assert_eq!(meta.duration_ms, 1000);
    }

    #[test]
    fn recover_entries_from_corrupt_json() {
        // Partial/corrupt JSON with two valid entries
        let corrupt = r#"[
            {"id":"1-0","created_at_ms":1000,"text":"hello","audio":null},
            {"id":"2-1","created_at_ms":2000,"text":"world","audio":null}
        GARBAGE"#;
        let recovered = recover_entries(corrupt);
        assert_eq!(recovered.len(), 2);
        assert_eq!(recovered[0].text, "hello");
        assert_eq!(recovered[1].text, "world");
    }

    #[test]
    fn recover_entries_empty_string() {
        let recovered = recover_entries("");
        assert!(recovered.is_empty());
    }

    #[test]
    fn write_atomic_retry_succeeds() {
        let dir = std::env::temp_dir().join("voclaude_test_history");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_history.json");
        write_atomic(&path, "[]").expect("write_atomic should succeed");
        let contents = std::fs::read_to_string(&path).unwrap();
        assert_eq!(contents, "[]");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

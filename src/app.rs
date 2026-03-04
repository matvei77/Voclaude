//! Main application orchestration.

use crate::audio::{AudioCapture, TARGET_SAMPLE_RATE};
use crate::config::Config;
use crate::history::{AudioMetadata, HistoryEntry, HistoryStore};
use crate::hotkey::HotkeyManager;
use crate::inference::{AsrEngine, InferenceProgress, InferenceStage, QwenEngine};
use crate::tray::TrayManager;
use crate::ui::{LogBuffer, UiManager, UiStatus};
use crate::session::SessionStore;

use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use crate::config::project_dirs;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, error, info, warn, trace};

#[cfg(not(target_os = "windows"))]
use notify_rust::Notification;

/// Application events
#[derive(Debug, Clone)]
pub enum AppEvent {
    /// Hotkey pressed - toggle recording
    HotkeyPressed,
    /// Toggle history window
    ToggleHistoryWindow,
    /// Show history window
    ShowHistoryWindow,
    /// Open transcripts folder
    OpenTranscriptsFolder,
    /// Open last transcript
    OpenLastTranscript,
    /// Open config file
    OpenSettings,
    /// Quit requested
    Quit,
    /// Inference progress update
    InferenceProgress(InferenceProgress),
    /// Transcription completed
    TranscriptionComplete(Result<String, String>),
    /// Inference engine info
    InferenceEngineInfo {
        using_gpu: bool,
        model: String,
        model_size_mb: u64,
    },
    /// History updated (for UI listeners)
    HistoryUpdated(HistoryEntry),
    /// E-8/S-7: Config reloaded successfully — apply runtime-updatable fields
    ConfigReloaded(Config),
}

/// Application state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppState {
    Idle,
    Recording,
    Transcribing,
}

const LONG_RECORDING_MS: u64 = 10 * 60 * 1000;
const LONG_TRANSCRIPT_CHARS: usize = 20000;

#[derive(Debug)]
enum InferenceCommand {
    TranscribeFile(PathBuf),
    Unload,
    Shutdown,
}

struct NotificationManager {
    enabled: bool,
    last_message: Option<String>,
    last_sent: Instant,
}

impl NotificationManager {
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            last_message: None,
            last_sent: Instant::now() - Duration::from_secs(60),
        }
    }

    fn notify(&mut self, message: &str) {
        if !self.enabled {
            return;
        }

        let min_interval = Duration::from_secs(10);
        if self.last_message.as_deref() == Some(message) && self.last_sent.elapsed() < min_interval {
            return;
        }

        #[cfg(not(target_os = "windows"))]
        {
            if let Err(e) = Notification::new()
                .summary("Voclaude")
                .body(message)
                .show()
            {
                warn!("Failed to show notification: {}", e);
                return;
            }
        }

        #[cfg(target_os = "windows")]
        {
            debug!("Notification: {}", message);
        }

        self.last_message = Some(message.to_string());
        self.last_sent = Instant::now();
    }
}

/// Main application
pub struct App {
    config: Config,
    state: AppState,
    event_tx: Sender<AppEvent>,
    event_rx: Receiver<AppEvent>,
    is_running: bool,
    ui: UiManager,
    #[allow(dead_code)]
    log_buffer: LogBuffer,
}

impl App {
    /// Run the application
    pub fn run(config: Config, log_buffer: LogBuffer) -> Result<(), Box<dyn std::error::Error>> {
        let (event_tx, event_rx) = unbounded::<AppEvent>();
        let ui = UiManager::new(log_buffer.clone(), config.use_gpu)?;

        let mut app = App {
            config,
            state: AppState::Idle,
            event_tx,
            event_rx,
            is_running: true,
            ui,
            log_buffer,
        };

        app.run_event_loop()
    }

    fn run_event_loop(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        info!("Initializing components...");

        // Initialize tray icon - MUST keep alive!
        let _tray = TrayManager::new(self.event_tx.clone())?;
        info!("Tray icon ready");

        // Initialize hotkey listener - single unified thread for all hotkeys
        // to avoid GlobalHotKeyEvent receiver contention.
        let _hotkey = HotkeyManager::new_multi(
            &[
                (&self.config.hotkey, AppEvent::HotkeyPressed),
                (&self.config.history_hotkey, AppEvent::ToggleHistoryWindow),
            ],
            self.event_tx.clone(),
        )?;
        info!("Hotkeys registered: {} (record), {} (history)", self.config.hotkey, self.config.history_hotkey);

        // Initialize audio capture
        let (level_tx, level_rx) = bounded::<f32>(64);
        let audio = AudioCapture::new(Some(level_tx))?;
        info!("Audio capture ready");

        // No preflight — model loads lazily on first transcription.
        // If it fails, the user gets immediate feedback via the HUD.

        // Initialize inference worker (lazy - loads model on demand)
        let (mut inference_tx, mut inference_handle) = Self::spawn_inference_worker(self.event_tx.clone(), self.config.clone());
        info!("Inference worker ready (model will load on first use)");

        // Clipboard
        let mut clipboard = arboard::Clipboard::new()?;

        let mut notifications = NotificationManager::new(self.config.show_notifications);

        // History storage
        let (history_update_tx, history_update_rx) = bounded::<HistoryEntry>(32);
        let mut history = HistoryStore::load(self.config.history_max_entries, history_update_tx)?;
        info!("History loaded: {} entries", history.len());
        for entry in history.entries() {
            self.ui.push_history(entry.text.clone());
        }

        let mut ui_status = UiStatus::new(
            self.config.hotkey.clone(),
            self.config.use_gpu,
            self.config.model_display_name(),
            None,
        );
        ui_status.history_count = history.len();
        ui_status.input_device = Some(audio.device_name());
        ui_status.input_level = Some(0.0);
        self.ui.set_status(ui_status.clone());

        let mut session_store = SessionStore::load()?;

        // Track last activity for idle unload
        let mut last_activity = Instant::now();
        let mut idle_unload_requested = false;
        let mut pending_audio_metadata: Option<AudioMetadata> = None;
        let mut last_progress_stage: Option<InferenceStage> = None;
        let mut transcribe_started_at: Option<Instant> = None;
        let mut last_transcript_path: Option<PathBuf> = None;

        if let Some(session) = session_store.current().cloned() {
            if session.is_recoverable() {
                if let Some(path) = session.audio_path() {
                    if path.exists() {
                        info!("Recovering previous session: {}", session.id);
                        if let Some(audio) = session.audio.clone() {
                            pending_audio_metadata = Some(audio);
                        } else if let Ok(metadata) = std::fs::metadata(&path) {
                            let sample_count = (metadata.len() / 4) as usize;
                            pending_audio_metadata = Some(AudioMetadata::from_samples(
                                sample_count,
                                TARGET_SAMPLE_RATE,
                            ));
                        }

                        if let Some(audio) = pending_audio_metadata.clone() {
                            let _ = session_store.mark_transcribing(audio);
                        }

                        self.state = AppState::Transcribing;
                        _tray.set_state(AppState::Transcribing);
                        transcribe_started_at = Some(Instant::now());
                        ui_status.state = "Recovering".to_string();
                        ui_status.last_message = Some("Recovering recording...".to_string());
                        self.ui.set_status(ui_status.clone());

                        // E-2: Use try_send to avoid blocking main thread
                        if let Err(err) = inference_tx.try_send(InferenceCommand::TranscribeFile(path)) {
                            warn!("Failed to start recovery transcription: {}", err);
                            let _ = session_store.mark_failed(
                                "Failed to start recovery transcription".to_string(),
                            );
                            self.state = AppState::Idle;
                            _tray.set_state(AppState::Idle);
                            ui_status.state = "Idle".to_string();
                            ui_status.last_message = Some("Recovery failed".to_string());
                            self.ui.set_status(ui_status.clone());
                        }
                    } else {
                        warn!("Recovery audio file missing; marking session failed");
                        let _ = session_store.mark_failed("Recovery audio file missing".to_string());
                    }
                }
            }
        }

        info!("=== VOCLAUDE READY ===");
        info!("Press {} to start recording", self.config.hotkey);
        info!("Press {} to toggle history window", self.config.history_hotkey);
        info!("Main thread ID: {:?}", std::thread::current().id());

        // Log Windows thread ID for debugging
        #[cfg(target_os = "windows")]
        {
            #[link(name = "kernel32")]
            extern "system" {
                fn GetCurrentThreadId() -> u32;
            }
            let win_thread_id = unsafe { GetCurrentThreadId() };
            info!("Windows Thread ID: {}", win_thread_id);
        }

        info!("Entering main event loop...");

        let mut loop_count: u64 = 0;
        let mut last_status_log = Instant::now();
        let mut messages_pumped: u64 = 0;

        // Watchdog respawn counter and backoff state
        let mut watchdog_respawn_count: u32 = 0;
        // E-3: Deferred respawn (avoid blocking main thread with sleep)
        let mut deferred_respawn_at: Option<Instant> = None;
        // E-4: Keep sentinel receiver alive so send() doesn't immediately error
        let mut _sentinel_rx: Option<Receiver<InferenceCommand>> = None;
        // E-7: Track audio file path for cleanup after transcription
        let mut pending_audio_path: Option<PathBuf> = None;

        // Settings watcher cancellation flag
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        let mut settings_watcher_cancel: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));

        // Main event loop - runs on main thread with Windows message pump
        while self.is_running {
            loop_count += 1;

            // Pump Windows messages (required for hotkeys and tray to work)
            let (pumped, os_quit) = Self::pump_messages();
            messages_pumped += pumped as u64;
            // E-1: Handle OS shutdown/logoff via WM_QUIT
            if os_quit {
                info!("OS requested shutdown via WM_QUIT");
                let _ = inference_tx.try_send(InferenceCommand::Shutdown);
                settings_watcher_cancel.store(true, Ordering::Relaxed);
                self.is_running = false;
            }

            // Log status every 10 seconds
            if last_status_log.elapsed() > Duration::from_secs(10) {
                info!("=== MAIN LOOP STATUS ===");
                info!("Loop iterations: {}", loop_count);
                info!("Messages pumped total: {}", messages_pumped);
                info!("Current state: {:?}", self.state);
                info!("Event channel len: {}", self.event_rx.len());
                last_status_log = Instant::now();
            }

            // E-3: Check deferred respawn (non-blocking, no sleep on main thread)
            if let Some(respawn_at) = deferred_respawn_at {
                if Instant::now() >= respawn_at {
                    deferred_respawn_at = None;
                    let (new_tx, new_handle) = Self::spawn_inference_worker(self.event_tx.clone(), self.config.clone());
                    inference_tx = new_tx;
                    inference_handle = new_handle;
                    info!("Inference worker respawned (attempt {})", watchdog_respawn_count);
                    notifications.notify("Inference engine restarted after error");
                }
            }

            // Watchdog: detect inference worker panic and respawn
            if deferred_respawn_at.is_none() && inference_handle.is_finished() {
                error!("Inference worker thread exited unexpectedly — scheduling respawn");
                let _ = inference_handle.join();
                watchdog_respawn_count += 1;
                if watchdog_respawn_count > 5 {
                    error!("Inference worker failed {} times; disabling", watchdog_respawn_count);
                    notifications.notify("Inference engine failed repeatedly — restart app");
                    // E-4: Keep receiver alive so send() returns SendError, not panic
                    let (dead_tx, dead_rx) = bounded::<InferenceCommand>(1);
                    _sentinel_rx = Some(dead_rx);
                    // E-5: Use a flag-based sentinel that can be joined on shutdown
                    let sentinel_shutdown = Arc::new(AtomicBool::new(false));
                    let sentinel_flag = sentinel_shutdown.clone();
                    inference_handle = thread::spawn(move || {
                        while !sentinel_flag.load(Ordering::Relaxed) {
                            thread::sleep(Duration::from_millis(100));
                        }
                    });
                    inference_tx = dead_tx;
                } else {
                    // E-15: Use saturating_sub(1) to fix off-by-one backoff
                    let backoff = Duration::from_millis(500 * 2u64.pow(watchdog_respawn_count.saturating_sub(1).min(4)));
                    deferred_respawn_at = Some(Instant::now() + backoff);
                    // Spawn a no-op placeholder so is_finished() doesn't fire again
                    inference_handle = thread::spawn(|| {});
                }
                if self.state == AppState::Transcribing {
                    self.state = AppState::Idle;
                    pending_audio_metadata = None;
                    _tray.set_state(AppState::Idle);
                    ui_status.state = "Idle".to_string();
                    ui_status.last_message = Some("Inference engine restarted after error".to_string());
                    self.ui.set_status(ui_status.clone());
                }
                // E-14: Reset idle_unload_requested on watchdog respawn
                idle_unload_requested = false;
            }

            // Check for idle unload
            if self.state == AppState::Idle {
                let idle_duration = last_activity.elapsed();
                if idle_duration > Duration::from_secs(self.config.idle_unload_seconds)
                    && !idle_unload_requested
                {
                    info!("Unloading model after {} seconds idle", idle_duration.as_secs());
                    // E-2: Use try_send to avoid blocking main thread
                    if let Err(err) = inference_tx.try_send(InferenceCommand::Unload) {
                        warn!("Failed to request model unload: {}", err);
                    }
                    idle_unload_requested = true;
                }
            }

            while let Ok(entry) = history_update_rx.try_recv() {
                if let Err(err) = self.event_tx.try_send(AppEvent::HistoryUpdated(entry)) {
                    debug!("Dropping history update event: {}", err);
                }
            }

            let mut latest_level: Option<f32> = None;
            while let Ok(level) = level_rx.try_recv() {
                latest_level = Some(level.clamp(0.0, 1.0));
            }
            if let Some(level) = latest_level {
                ui_status.input_level = Some(level);
                self.ui.set_status(ui_status.clone());
            }

            // Process events (non-blocking)
            match self.event_rx.try_recv() {
                Ok(event) => {
                    info!("=== APP EVENT RECEIVED ===");
                    info!("Event: {:?}", event);
                    info!("Time since last activity: {:?}", last_activity.elapsed());

                    match event {
                        AppEvent::HotkeyPressed => {
                            last_activity = Instant::now();
                            idle_unload_requested = false;
                            info!("Processing HotkeyPressed, current state: {:?}", self.state);
                            match self.state {
                                AppState::Idle => {
                                    info!("Starting recording...");
                                    if let Err(e) = audio.start() {
                                        error!("Failed to start recording: {}", e);
                                        _tray.set_state(AppState::Idle);
                                        notifications.notify("Failed to start recording");
                                        ui_status.state = "Idle".to_string();
                                        ui_status.last_message = Some(format!("Recording failed: {}", e));
                                        self.ui.set_status(ui_status.clone());
                                        continue;
                                    }
                                    // Update displayed device name (may have changed since last recording)
                                    ui_status.input_device = Some(audio.device_name());
                                    if let Err(err) = session_store.start() {
                                        warn!("Failed to start session metadata: {}", err);
                                    }
                                    self.state = AppState::Recording;
                                    pending_audio_metadata = None;
                                    last_progress_stage = None;
                                    _tray.set_state(AppState::Recording);
                                    ui_status.state = "Recording".to_string();
                                    ui_status.last_message = Some("Recording...".to_string());
                                    self.ui.set_status(ui_status.clone());
                                }
                                AppState::Recording => {
                                    info!("Stopping recording...");
                                    match audio.stop() {
                                        Ok(recording) => {
                                            if recording.is_empty() {
                                                warn!("No audio recorded");
                                                let _ = session_store.mark_aborted(Some("no_audio".to_string()));
                                                self.state = AppState::Idle;
                                                _tray.set_state(AppState::Idle);
                                                ui_status.state = "Idle".to_string();
                                                ui_status.last_message = Some("No audio captured".to_string());
                                                ui_status.input_level = Some(0.0);
                                                self.ui.set_status(ui_status.clone());
                                                continue;
                                            }

                                            let sample_count = recording.sample_count;
                                            info!("Got {} samples, transcribing...", sample_count);
                                            self.state = AppState::Transcribing;
                                            let metadata = AudioMetadata::from_samples(
                                                sample_count,
                                                recording.sample_rate,
                                            );
                                            pending_audio_metadata = Some(metadata.clone());
                                            let _ = session_store.mark_transcribing(metadata);
                                            _tray.set_state(AppState::Transcribing);
                                            transcribe_started_at = Some(Instant::now());
                                            ui_status.state = "Transcribing".to_string();
                                            ui_status.last_message = Some("Transcribing audio...".to_string());
                                            self.ui.set_status(ui_status.clone());

                                            // E-7: Track audio file path for cleanup
                                            pending_audio_path = Some(recording.path.clone());

                                            // E-2: Use try_send to avoid blocking main thread
                                            if let Err(err) = inference_tx.try_send(
                                                InferenceCommand::TranscribeFile(recording.path),
                                            ) {
                                                error!("Failed to start transcription: {}", err);
                                                let _ = session_store.mark_failed(
                                                    "Failed to start transcription".to_string(),
                                                );
                                                self.state = AppState::Idle;
                                                pending_audio_metadata = None;
                                                _tray.set_state(AppState::Idle);
                                                ui_status.state = "Idle".to_string();
                                                ui_status.last_message = Some("Transcription failed to start".to_string());
                                                ui_status.input_level = Some(0.0);
                                                self.ui.set_status(ui_status.clone());
                                            }
                                        }
                                        Err(e) => {
                                            error!("Failed to stop recording: {}", e);
                                            let _ = session_store
                                                .mark_failed("Failed to stop recording".to_string());
                                            self.state = AppState::Idle;
                                            _tray.set_state(AppState::Idle);
                                            ui_status.state = "Idle".to_string();
                                            ui_status.last_message = Some("Failed to stop recording".to_string());
                                            ui_status.input_level = Some(0.0);
                                            self.ui.set_status(ui_status.clone());
                                        }
                                    }
                                }
                                AppState::Transcribing => {
                                    // Ignore hotkey while transcribing
                                    debug!("Ignoring hotkey while transcribing");
                                }
                            }
                        }
                        AppEvent::InferenceProgress(progress) => {
                            // E-9: Only process progress when actively transcribing
                            if self.state == AppState::Transcribing {
                                _tray.set_progress(&progress.message);
                                ui_status.last_message = Some(progress.message.clone());
                                self.ui.set_status(ui_status.clone());
                                if last_progress_stage != Some(progress.stage) {
                                    notifications.notify(&progress.message);
                                    last_progress_stage = Some(progress.stage);
                                }
                            }
                        }
                        AppEvent::InferenceEngineInfo {
                            using_gpu,
                            model,
                            model_size_mb,
                        } => {
                            ui_status.use_gpu = using_gpu;
                            ui_status.model = model;
                            ui_status.model_size_mb = Some(model_size_mb);
                            if self.config.use_gpu && !using_gpu {
                                ui_status.last_message =
                                    Some("GPU init failed; using CPU".to_string());
                            }
                            self.ui.set_status(ui_status.clone());
                        }
                        AppEvent::TranscriptionComplete(result) => {
                            // E-6: Guard against stale results from respawned worker
                            if self.state != AppState::Transcribing {
                                warn!("Ignoring stale TranscriptionComplete (state={:?})", self.state);
                                continue;
                            }
                            last_activity = Instant::now();
                            idle_unload_requested = false;
                            // Reset respawn counter on successful transcription
                            if matches!(result, Ok(_)) {
                                watchdog_respawn_count = 0;
                            }
                            // E-7: Clean up audio file after transcription
                            if let Some(audio_path) = pending_audio_path.take() {
                                if audio_path.exists() {
                                    if let Err(err) = std::fs::remove_file(&audio_path) {
                                        warn!("Failed to delete audio file {:?}: {}", audio_path, err);
                                    } else {
                                        debug!("Cleaned up audio file: {:?}", audio_path);
                                    }
                                }
                            }
                            match result {
                                Ok(text) => {
                                    // S-1: Don't log full transcript (may contain sensitive dictation)
                                    info!("Transcription complete: {} chars", text.len());

                                    // Format text
                                    let mut formatted = text.trim().to_string();
                                    // E-11: Use byte offset instead of char split to avoid
                                    // corrupting multi-codepoint grapheme clusters
                                    if self.config.capitalize_first && !formatted.is_empty() {
                                        if let Some(first_char) = formatted.chars().next() {
                                            let upper: String = first_char.to_uppercase().collect();
                                            formatted = upper + &formatted[first_char.len_utf8()..];
                                        }
                                    }
                                    if self.config.add_trailing_space {
                                        formatted.push(' ');
                                    }

                                    let history_text = formatted.trim_end().to_string();
                                    let metadata_snapshot = pending_audio_metadata.clone();
                                    let is_long = metadata_snapshot
                                        .as_ref()
                                        .map(|meta| meta.duration_ms >= LONG_RECORDING_MS)
                                        .unwrap_or(false)
                                        || history_text.chars().count() > LONG_TRANSCRIPT_CHARS;
                                    let session_id = session_store
                                        .current()
                                        .map(|session| session.id.clone())
                                        .unwrap_or_else(fallback_session_id);
                                    let mut saved_transcript_path = None;
                                    if is_long && !history_text.is_empty() {
                                        if let Some(path) = transcript_output_path(&session_id) {
                                            match write_transcript_file(&path, &history_text) {
                                                Ok(()) => {
                                                    saved_transcript_path = Some(path);
                                                }
                                                Err(err) => {
                                                    warn!("Failed to save transcript: {}", err);
                                                }
                                            }
                                        }
                                    }

                                    if let Some(path) = saved_transcript_path.as_ref() {
                                        last_transcript_path = Some(path.clone());
                                    } else if !history_text.is_empty() {
                                        if let Some(path) = last_transcript_output_path() {
                                            if let Err(err) = write_transcript_file(&path, &history_text) {
                                                warn!("Failed to save last transcript: {}", err);
                                            } else {
                                                last_transcript_path = Some(path);
                                            }
                                        }
                                    }
                                    let mut history_entry_id = None;
                                    let mut history_error = None;
                                    if !history_text.is_empty() {
                                        let metadata = pending_audio_metadata.take();
                                        match history.append(history_text.clone(), metadata) {
                                            Ok(entry) => {
                                                history_entry_id = Some(entry.id);
                                                ui_status.history_count = history.len();
                                            }
                                            Err(e) => {
                                                error!("Failed to append history: {}", e);
                                                history_error = Some(e.to_string());
                                            }
                                        }
                                    }

                                    if let Some(err) = history_error {
                                        let _ = session_store
                                            .mark_failed(format!("Failed to append history: {}", err));
                                    } else {
                                        let _ = session_store
                                            .mark_completed(history_text.clone(), history_entry_id);
                                    }

                                    if let (Some(started_at), Some(audio)) =
                                        (transcribe_started_at.take(), metadata_snapshot)
                                    {
                                        let processing_ms = started_at.elapsed().as_millis() as u64;
                                        let audio_secs = audio.duration_ms as f32 / 1000.0;
                                        let processing_secs = (processing_ms as f32 / 1000.0).max(0.01);
                                        ui_status.last_duration_ms = Some(audio.duration_ms);
                                        ui_status.last_speed = Some(audio_secs / processing_secs);
                                    }

                                    let clipboard_text = saved_transcript_path
                                        .as_ref()
                                        .map(|path| path.display().to_string())
                                        .unwrap_or_else(|| formatted.clone());
                                    let notify_message = if saved_transcript_path.is_some() {
                                        "Transcript saved; path copied to clipboard"
                                    } else {
                                        "Transcription copied to clipboard"
                                    };

                                    // Copy to clipboard
                                    if let Err(e) = clipboard.set_text(&clipboard_text) {
                                        error!("Failed to copy to clipboard: {}", e);
                                        ui_status.state = "Idle".to_string();
                                        ui_status.last_message = Some("Clipboard copy failed".to_string());
                                        self.ui.set_status(ui_status.clone());
                                    } else {
                                        info!("Copied to clipboard!");
                                        notifications.notify(notify_message);
                                        ui_status.state = "Idle".to_string();
                                        ui_status.last_message = Some(notify_message.to_string());
                                        self.ui.set_status(ui_status.clone());
                                    }
                                }
                                Err(e) => {
                                    error!("Transcription failed: {}", e);
                                    let _ = session_store
                                        .mark_failed(format!("Transcription failed: {}", e));
                                    let short = shorten_error_message(&e, 140);
                                    notifications.notify(&format!("Transcription failed: {}", short));
                                    ui_status.state = "Idle".to_string();
                                    ui_status.last_message =
                                        Some(format!("Transcription failed: {}", short));
                                    self.ui.set_status(ui_status.clone());
                                }
                            }

                            self.state = AppState::Idle;
                            pending_audio_metadata = None;
                            last_progress_stage = None;
                            _tray.set_state(AppState::Idle);
                            ui_status.input_level = Some(0.0);
                            self.ui.set_status(ui_status.clone());

                            // Model stays resident for fast re-use.
                            // The idle_unload_seconds timer (default 30s) handles
                            // VRAM reclamation when the user stops using the app.
                        }
                        AppEvent::HistoryUpdated(entry) => {
                            debug!("History updated: {}", entry.id);
                            self.ui.push_history(entry.text);
                            ui_status.history_count = history.len();
                            self.ui.set_status(ui_status.clone());
                        }
                        AppEvent::ToggleHistoryWindow => {
                            last_activity = Instant::now();
                            idle_unload_requested = false;
                            // Reload history from disk before showing so entries
                            // are always fresh, even across hide/show cycles.
                            let entries: Vec<String> = history
                                .entries()
                                .iter()
                                .map(|e| e.text.clone())
                                .collect();
                            self.ui.reload_history(entries);
                            if !self.ui.toggle() {
                                warn!("History window toggle failed (UI thread may have exited)");
                                notifications.notify("History window is unavailable");
                            }
                        }
                        AppEvent::ShowHistoryWindow => {
                            last_activity = Instant::now();
                            idle_unload_requested = false;
                            let entries: Vec<String> = history
                                .entries()
                                .iter()
                                .map(|e| e.text.clone())
                                .collect();
                            info!("ShowHistoryWindow: {} entries to display", entries.len());
                            self.ui.reload_history(entries);
                            if !self.ui.show() {
                                warn!("History window show failed (UI thread may have exited)");
                                notifications.notify("History window is unavailable");
                            }
                        }
                        AppEvent::OpenLastTranscript => {
                            last_activity = Instant::now();
                            idle_unload_requested = false;
                            match last_transcript_path.as_ref() {
                                Some(path) if path.exists() => {
                                    if let Err(err) = open_path(path) {
                                        warn!("Failed to open transcript: {}", err);
                                    }
                                }
                                _ => {
                                    notifications.notify("No saved transcript yet");
                                }
                            }
                        }
                        AppEvent::OpenSettings => {
                            last_activity = Instant::now();
                            idle_unload_requested = false;
                            match ensure_config_file() {
                                Ok(path) => {
                                    if let Err(err) = open_path(&path) {
                                        warn!("Failed to open config file: {}", err);
                                    } else {
                                        // Validate config after user potentially edits it.
                                        // Spawn a background thread to poll for file changes
                                        // and validate when the modification time updates.
                                        let event_tx_clone = self.event_tx.clone();
                                        let config_path = path.clone();
                                        let original_mtime = std::fs::metadata(&config_path)
                                            .and_then(|m| m.modified())
                                            .ok();
                                        // BUG 16: cancel any previous watcher, then arm a fresh flag
                                        settings_watcher_cancel.store(true, Ordering::Relaxed);
                                        settings_watcher_cancel = Arc::new(AtomicBool::new(false));
                                        let cancel = Arc::clone(&settings_watcher_cancel);
                                        thread::spawn(move || {
                                            // Wait up to 5 minutes for the user to save
                                            let deadline = Instant::now() + Duration::from_secs(300);
                                            while Instant::now() < deadline {
                                                // BUG 16: check cancellation at top of each iteration
                                                if cancel.load(Ordering::Relaxed) {
                                                    break;
                                                }
                                                thread::sleep(Duration::from_secs(2));
                                                if cancel.load(Ordering::Relaxed) {
                                                    break;
                                                }
                                                let current_mtime = std::fs::metadata(&config_path)
                                                    .and_then(|m| m.modified())
                                                    .ok();
                                                // BUG 46: removed last_check dead code — just check mtime
                                                if current_mtime != original_mtime {
                                                    // File was modified — validate
                                                    match Config::load() {
                                                        Ok(new_config) => {
                                                            // E-8/S-7: Send the new config so it's actually applied
                                                            info!("Config reloaded successfully after edit");
                                                            let _ = event_tx_clone.send(AppEvent::ConfigReloaded(new_config));
                                                            break;
                                                        }
                                                        Err(err) => {
                                                            warn!("Config validation failed: {}", err);
                                                            // Don't break — user might fix and re-save
                                                        }
                                                    }
                                                }
                                            }
                                        });
                                    }
                                }
                                Err(err) => {
                                    warn!("Failed to resolve config file: {}", err);
                                    notifications.notify("Failed to open config file");
                                }
                            }
                        }
                        AppEvent::OpenTranscriptsFolder => {
                            last_activity = Instant::now();
                            idle_unload_requested = false;
                            if let Err(err) = open_transcripts_folder() {
                                warn!("Failed to open transcripts folder: {}", err);
                            }
                        }
                        AppEvent::Quit => {
                            last_activity = Instant::now();
                            idle_unload_requested = false;
                            info!("Quit requested");
                            // E-2: Use try_send to avoid blocking main thread
                            let _ = inference_tx.try_send(InferenceCommand::Shutdown);
                            // E-13: Cancel settings watcher on quit
                            settings_watcher_cancel.store(true, Ordering::Relaxed);
                            self.is_running = false;
                        }
                        // E-8/S-7: Apply runtime-updatable config fields
                        AppEvent::ConfigReloaded(new_config) => {
                            let mut needs_restart = Vec::new();
                            if new_config.hotkey != self.config.hotkey {
                                needs_restart.push("hotkey");
                            }
                            if new_config.history_hotkey != self.config.history_hotkey {
                                needs_restart.push("history_hotkey");
                            }
                            if new_config.model != self.config.model {
                                needs_restart.push("model");
                            }
                            if new_config.use_gpu != self.config.use_gpu {
                                needs_restart.push("use_gpu");
                            }
                            // Apply runtime-updatable fields
                            self.config.idle_unload_seconds = new_config.idle_unload_seconds;
                            self.config.show_notifications = new_config.show_notifications;
                            self.config.capitalize_first = new_config.capitalize_first;
                            self.config.add_trailing_space = new_config.add_trailing_space;
                            self.config.max_new_tokens = new_config.max_new_tokens;
                            self.config.language = new_config.language.clone();
                            self.config.history_max_entries = new_config.history_max_entries;
                            self.config.require_gpu = new_config.require_gpu;
                            notifications.enabled = new_config.show_notifications;
                            let msg = if needs_restart.is_empty() {
                                "Settings applied successfully".to_string()
                            } else {
                                format!("Settings saved (restart required for: {})", needs_restart.join(", "))
                            };
                            info!("{}", msg);
                            notifications.notify(&msg);
                            ui_status.last_message = Some(msg);
                            self.ui.set_status(ui_status.clone());
                        }
                    }
                }
                Err(crossbeam_channel::TryRecvError::Empty) => {
                    // No events, sleep briefly to avoid busy-spinning
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    error!("Event channel disconnected");
                    break;
                }
            }
        }

        drop(inference_tx); // Close channel so worker exits iter() loop
        info!("Waiting for inference worker to shut down...");
        if let Err(e) = inference_handle.join() {
            error!("Inference worker thread panicked: {:?}", e);
        }
        info!("Shutdown complete.");
        Ok(())
    }

    fn spawn_inference_worker(
        event_tx: Sender<AppEvent>,
        config: Config,
    ) -> (Sender<InferenceCommand>, thread::JoinHandle<()>) {
        let (inference_tx, inference_rx) = bounded::<InferenceCommand>(8);
        let handle = thread::spawn(move || Self::inference_worker(inference_rx, event_tx, config));
        (inference_tx, handle)
    }

    fn inference_worker(
        inference_rx: Receiver<InferenceCommand>,
        event_tx: Sender<AppEvent>,
        config: Config,
    ) {
        let mut engine = match QwenEngine::new_with_config(&config) {
            Ok(engine) => engine,
            Err(err) => {
                let _ = event_tx.send(AppEvent::TranscriptionComplete(Err(format!(
                    "Failed to initialize inference: {}",
                    err
                ))));
                return;
            }
        };

        for command in inference_rx.iter() {
            match command {
                InferenceCommand::TranscribeFile(path) => {
                    let mut callback = |progress: InferenceProgress| {
                        let _ = event_tx.send(AppEvent::InferenceProgress(progress));
                    };
                    if let Err(err) = engine.prepare(Some(&mut callback)) {
                        let _ = event_tx.send(AppEvent::TranscriptionComplete(Err(
                            format!("Failed to load model: {}", err),
                        )));
                        continue;
                    }
                    let _ = event_tx.send(AppEvent::InferenceEngineInfo {
                        using_gpu: engine.active_gpu(),
                        model: engine.model_label(),
                        model_size_mb: engine.model_size_mb(),
                    });
                    let result =
                        engine.transcribe_file_with_progress(&path, Some(&mut callback));
                    let result = result.map_err(|err| err.to_string());
                    let _ = event_tx.send(AppEvent::TranscriptionComplete(result));
                }
                InferenceCommand::Unload => {
                    engine.unload();
                }
                InferenceCommand::Shutdown => {
                    break;
                }
            }
        }
    }

    /// Pump Windows messages - MUST be called from main thread
    /// Returns (messages_processed, quit_received)
    #[cfg(target_os = "windows")]
    fn pump_messages() -> (u32, bool) {
        use windows_sys::Win32::UI::WindowsAndMessaging::{
            DispatchMessageW, PeekMessageW, TranslateMessage, MSG, PM_REMOVE, PM_NOREMOVE,
            WM_HOTKEY, WM_QUIT, WM_TIMER, WM_NULL,
        };

        let mut count = 0u32;
        let mut quit = false;
        unsafe {
            let mut msg: MSG = std::mem::zeroed();

            let has_messages = PeekMessageW(&mut msg, 0 as _, 0, 0, PM_NOREMOVE);
            if has_messages != 0 {
                trace!("PeekMessage found messages waiting");
            }

            while PeekMessageW(&mut msg, 0 as _, 0, 0, PM_REMOVE) != 0 {
                count += 1;

                match msg.message {
                    WM_HOTKEY => {
                        info!("=== WM_HOTKEY MESSAGE RECEIVED ===");
                        info!("hwnd: {:?}", msg.hwnd);
                        info!("wParam (hotkey id): {}", msg.wParam);
                        info!("lParam: {:#x}", msg.lParam);
                    }
                    WM_QUIT => {
                        // E-1: Handle WM_QUIT properly — don't dispatch, signal exit
                        info!("WM_QUIT received — triggering shutdown");
                        quit = true;
                        continue; // Don't dispatch WM_QUIT
                    }
                    WM_TIMER | WM_NULL => {}
                    _ => {
                        debug!("Windows message: {} (hwnd: {:?}, wParam: {}, lParam: {})",
                               msg.message, msg.hwnd, msg.wParam, msg.lParam);
                    }
                }

                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }
        (count, quit)
    }

    #[cfg(not(target_os = "windows"))]
    fn pump_messages() -> (u32, bool) {
        (0, false)
    }
}

fn fallback_session_id() -> String {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or_default();
    format!("session-{}", ts)
}

fn transcripts_dir() -> Option<PathBuf> {
    project_dirs().map(|dirs| dirs.data_dir().join("transcripts"))
}

fn transcript_output_path(session_id: &str) -> Option<PathBuf> {
    transcripts_dir().map(|dir| dir.join(format!("{}.txt", session_id)))
}

fn last_transcript_output_path() -> Option<PathBuf> {
    transcripts_dir().map(|dir| dir.join("last_transcript.txt"))
}

fn write_transcript_file(path: &Path, text: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::File::create(path)?;
    file.write_all(text.as_bytes())?;
    file.flush()?;
    Ok(())
}

fn open_transcripts_folder() -> Result<(), Box<dyn std::error::Error>> {
    let Some(dir) = transcripts_dir() else {
        return Err("Could not determine transcripts directory".into());
    };
    fs::create_dir_all(&dir)?;
    open_path(&dir)
}

fn open_path(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        // "start" opens files with default app and folders in Explorer.
        // First "" arg is the window title (required by start when path is quoted).
        Command::new("cmd")
            .args(["/c", "start", ""])
            .arg(path)
            .creation_flags(0x08000000) // CREATE_NO_WINDOW
            .spawn()?;
    }

    #[cfg(target_os = "macos")]
    {
        Command::new("open").arg(path).spawn()?;
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        Command::new("xdg-open").arg(path).spawn()?;
    }

    Ok(())
}

fn ensure_config_file() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let path = Config::config_path().ok_or("Could not determine config path")?;
    if !path.exists() {
        Config::default().save()?;
    }
    Ok(path)
}

fn shorten_error_message(input: &str, max_chars: usize) -> String {
    if input.chars().count() <= max_chars {
        return input.to_string();
    }
    let mut out = String::new();
    for ch in input.chars().take(max_chars) {
        out.push(ch);
    }
    out.push_str("...");
    out
}

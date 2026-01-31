//! Main application orchestration.

use crate::audio::{AudioCapture, WHISPER_SAMPLE_RATE};
use crate::config::Config;
use crate::history::{AudioMetadata, HistoryEntry, HistoryStore};
use crate::hotkey::HotkeyManager;
use crate::inference::{InferenceProgress, InferenceStage, WhisperEngine};
use crate::tray::TrayManager;
use crate::ui::{HudManager, HudState, LogBuffer, UiManager, UiStatus};
use crate::session::SessionStore;

use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use directories::ProjectDirs;
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
}

impl App {
    /// Run the application
    pub fn run(config: Config, log_buffer: LogBuffer) -> Result<(), Box<dyn std::error::Error>> {
        let (event_tx, event_rx) = unbounded::<AppEvent>();
        let ui = UiManager::new(log_buffer)?;

        let mut app = App {
            config,
            state: AppState::Idle,
            event_tx,
            event_rx,
            is_running: true,
            ui,
        };

        app.run_event_loop()
    }

    fn run_event_loop(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        info!("Initializing components...");

        // Initialize tray icon - MUST keep alive!
        let _tray = TrayManager::new(self.event_tx.clone())?;
        info!("Tray icon ready");

        // Initialize hotkey listener - MUST keep alive!
        let _hotkey = HotkeyManager::new(
            &self.config.hotkey,
            AppEvent::HotkeyPressed,
            self.event_tx.clone(),
        )?;
        let _history_hotkey = HotkeyManager::new(
            &self.config.history_hotkey,
            AppEvent::ToggleHistoryWindow,
            self.event_tx.clone(),
        )?;
        info!("Hotkey registered: {}", self.config.hotkey);
        info!("History hotkey registered: {}", self.config.history_hotkey);

        // Initialize audio capture
        let audio = AudioCapture::new()?;
        info!("Audio capture ready");

        // Initialize inference worker (lazy - loads model on demand)
        let inference_tx = Self::spawn_inference_worker(self.event_tx.clone(), self.config.clone());
        info!("Inference worker ready (model will load on first use)");

        // Clipboard
        let mut clipboard = arboard::Clipboard::new()?;

        let mut notifications = NotificationManager::new(self.config.show_notifications);

        let hud = HudManager::new(self.config.use_gpu && cfg!(feature = "cuda"))?;

        // History storage
        let (history_update_tx, history_update_rx) = bounded::<HistoryEntry>(32);
        let mut history = HistoryStore::load(self.config.history_max_entries, history_update_tx)?;
        info!("History loaded: {} entries", history.len());
        for entry in history.entries() {
            self.ui.push_history(entry.text.clone());
        }

        let mut ui_status = UiStatus::new(
            self.config.hotkey.clone(),
            self.config.use_gpu && cfg!(feature = "cuda"),
            "whisper-medium".to_string(),
            Some(1533),
        );
        ui_status.history_count = history.len();
        if self.config.use_gpu && !cfg!(feature = "cuda") {
            ui_status.use_gpu = false;
            ui_status.last_message = Some("GPU requested but CUDA build is disabled".to_string());
        }
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
                                WHISPER_SAMPLE_RATE,
                            ));
                        }

                        if let Some(audio) = pending_audio_metadata.clone() {
                            let _ = session_store.mark_transcribing(audio);
                        }

                        self.state = AppState::Transcribing;
                        _tray.set_state(AppState::Transcribing);
                        hud.set_state(HudState::Transcribing {
                            message: "Recovering recording...".to_string(),
                            percent: None,
                        });
                        transcribe_started_at = Some(Instant::now());
                        ui_status.state = "Recovering".to_string();
                        ui_status.last_message = Some("Recovering recording...".to_string());
                        self.ui.set_status(ui_status.clone());

                        if let Err(err) = inference_tx.send(InferenceCommand::TranscribeFile(path)) {
                            warn!("Failed to start recovery transcription: {}", err);
                            let _ = session_store.mark_failed(
                                "Failed to start recovery transcription".to_string(),
                            );
                            self.state = AppState::Idle;
                            _tray.set_state(AppState::Idle);
                            hud.set_state(HudState::Idle);
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

        // Main event loop - runs on main thread with Windows message pump
        while self.is_running {
            loop_count += 1;

            // Pump Windows messages (required for hotkeys and tray to work)
            let pumped = Self::pump_messages();
            messages_pumped += pumped as u64;

            // Log status every 10 seconds
            if last_status_log.elapsed() > Duration::from_secs(10) {
                info!("=== MAIN LOOP STATUS ===");
                info!("Loop iterations: {}", loop_count);
                info!("Messages pumped total: {}", messages_pumped);
                info!("Current state: {:?}", self.state);
                info!("Event channel len: {}", self.event_rx.len());
                last_status_log = Instant::now();
            }

            // Check for idle unload
            if self.state == AppState::Idle {
                let idle_duration = last_activity.elapsed();
                if idle_duration > Duration::from_secs(self.config.idle_unload_seconds)
                    && !idle_unload_requested
                {
                    info!("Unloading model after {} seconds idle", idle_duration.as_secs());
                    if let Err(err) = inference_tx.send(InferenceCommand::Unload) {
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

            // Process events (non-blocking)
            match self.event_rx.try_recv() {
                Ok(event) => {
                    info!("=== APP EVENT RECEIVED ===");
                    info!("Event: {:?}", event);
                    info!("Time since last activity: {:?}", last_activity.elapsed());
                    last_activity = Instant::now();
                    idle_unload_requested = false;

                    match event {
                        AppEvent::HotkeyPressed => {
                            info!("Processing HotkeyPressed, current state: {:?}", self.state);
                            match self.state {
                                AppState::Idle => {
                                    info!("Starting recording...");
                                    if let Err(e) = audio.start() {
                                        error!("Failed to start recording: {}", e);
                                        _tray.set_state(AppState::Idle);
                                        notifications.notify("Failed to start recording");
                                        ui_status.state = "Idle".to_string();
                                        ui_status.last_message = Some("Failed to start recording".to_string());
                                        self.ui.set_status(ui_status.clone());
                                        continue;
                                    }
                                    if let Err(err) = session_store.start() {
                                        warn!("Failed to start session metadata: {}", err);
                                    }
                                    self.state = AppState::Recording;
                                    pending_audio_metadata = None;
                                    last_progress_stage = None;
                                    _tray.set_state(AppState::Recording);
                                    hud.set_state(HudState::Recording);
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
                                                hud.set_state(HudState::Idle);
                                                ui_status.state = "Idle".to_string();
                                                ui_status.last_message = Some("No audio captured".to_string());
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
                                            hud.set_state(HudState::Transcribing {
                                                message: "Transcribing audio...".to_string(),
                                                percent: None,
                                            });
                                            transcribe_started_at = Some(Instant::now());
                                            ui_status.state = "Transcribing".to_string();
                                            ui_status.last_message = Some("Transcribing audio...".to_string());
                                            self.ui.set_status(ui_status.clone());

                                            if let Err(err) = inference_tx.send(
                                                InferenceCommand::TranscribeFile(recording.path),
                                            ) {
                                                error!("Failed to start transcription: {}", err);
                                                let _ = session_store.mark_failed(
                                                    "Failed to start transcription".to_string(),
                                                );
                                                self.state = AppState::Idle;
                                                pending_audio_metadata = None;
                                                _tray.set_state(AppState::Idle);
                                                hud.set_state(HudState::Idle);
                                                ui_status.state = "Idle".to_string();
                                                ui_status.last_message = Some("Transcription failed to start".to_string());
                                                self.ui.set_status(ui_status.clone());
                                            }
                                        }
                                        Err(e) => {
                                            error!("Failed to stop recording: {}", e);
                                            let _ = session_store
                                                .mark_failed("Failed to stop recording".to_string());
                                            self.state = AppState::Idle;
                                            _tray.set_state(AppState::Idle);
                                            hud.set_state(HudState::Idle);
                                            ui_status.state = "Idle".to_string();
                                            ui_status.last_message = Some("Failed to stop recording".to_string());
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
                            if self.state == AppState::Transcribing {
                                _tray.set_progress(&progress.message);
                            }

                            if self.state == AppState::Transcribing {
                                let _ = hud.set_state(HudState::Transcribing {
                                    message: progress.message.clone(),
                                    percent: progress.percent,
                                });
                                ui_status.last_message = Some(progress.message.clone());
                                self.ui.set_status(ui_status.clone());
                            }

                            if last_progress_stage != Some(progress.stage) {
                                notifications.notify(&progress.message);
                                last_progress_stage = Some(progress.stage);
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
                            hud.set_accel(using_gpu);
                        }
                        AppEvent::TranscriptionComplete(result) => {
                            match result {
                                Ok(text) => {
                                    info!("Transcribed: {}", text);

                                    // Format text
                                    let mut formatted = text.trim().to_string();
                                    if self.config.capitalize_first && !formatted.is_empty() {
                                        let mut chars = formatted.chars();
                                        if let Some(first) = chars.next() {
                                            formatted = first.to_uppercase().collect::<String>() + chars.as_str();
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
                                        hud.set_state(HudState::Ready {
                                            message: "Clipboard copy failed".to_string(),
                                        });
                                        ui_status.state = "Idle".to_string();
                                        ui_status.last_message = Some("Clipboard copy failed".to_string());
                                        self.ui.set_status(ui_status.clone());
                                    } else {
                                        info!("Copied to clipboard!");
                                        notifications.notify(notify_message);
                                        hud.set_state(HudState::Ready {
                                            message: notify_message.to_string(),
                                        });
                                        ui_status.state = "Idle".to_string();
                                        ui_status.last_message = Some(notify_message.to_string());
                                        self.ui.set_status(ui_status.clone());
                                    }
                                }
                                Err(e) => {
                                    error!("Transcription failed: {}", e);
                                    let _ = session_store
                                        .mark_failed(format!("Transcription failed: {}", e));
                                    notifications.notify("Transcription failed");
                                    hud.set_state(HudState::Ready {
                                        message: "Transcription failed".to_string(),
                                    });
                                    ui_status.state = "Idle".to_string();
                                    ui_status.last_message = Some("Transcription failed".to_string());
                                    self.ui.set_status(ui_status.clone());
                                }
                            }

                            self.state = AppState::Idle;
                            pending_audio_metadata = None;
                            last_progress_stage = None;
                            _tray.set_state(AppState::Idle);
                        }
                        AppEvent::HistoryUpdated(entry) => {
                            debug!("History updated: {}", entry.id);
                            self.ui.push_history(entry.text);
                            ui_status.history_count = history.len();
                            self.ui.set_status(ui_status.clone());
                        }
                        AppEvent::ToggleHistoryWindow => {
                            if !self.ui.toggle() {
                                notifications.notify("History window is unavailable");
                            }
                        }
                        AppEvent::ShowHistoryWindow => {
                            if !self.ui.show() {
                                notifications.notify("History window is unavailable");
                            }
                        }
                        AppEvent::OpenLastTranscript => {
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
                        AppEvent::OpenTranscriptsFolder => {
                            if let Err(err) = open_transcripts_folder() {
                                warn!("Failed to open transcripts folder: {}", err);
                            }
                        }
                        AppEvent::Quit => {
                            info!("Quit requested");
                            let _ = inference_tx.send(InferenceCommand::Shutdown);
                            self.is_running = false;
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

        let _ = inference_tx.send(InferenceCommand::Shutdown);
        info!("Shutting down...");
        Ok(())
    }

    fn spawn_inference_worker(
        event_tx: Sender<AppEvent>,
        config: Config,
    ) -> Sender<InferenceCommand> {
        let (inference_tx, inference_rx) = bounded::<InferenceCommand>(2);
        thread::spawn(move || Self::inference_worker(inference_rx, event_tx, config));
        inference_tx
    }

    fn inference_worker(
        inference_rx: Receiver<InferenceCommand>,
        event_tx: Sender<AppEvent>,
        config: Config,
    ) {
        let mut engine = match WhisperEngine::new_with_config(&config) {
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
    /// Returns the number of messages processed
    #[cfg(target_os = "windows")]
    fn pump_messages() -> u32 {
        use windows_sys::Win32::UI::WindowsAndMessaging::{
            DispatchMessageW, PeekMessageW, TranslateMessage, MSG, PM_REMOVE, PM_NOREMOVE,
            WM_HOTKEY, WM_QUIT, WM_TIMER, WM_NULL,
        };

        let mut count = 0u32;
        unsafe {
            let mut msg: MSG = std::mem::zeroed();

            // First check if there are ANY messages (diagnostic)
            let has_messages = PeekMessageW(&mut msg, 0 as _, 0, 0, PM_NOREMOVE);
            if has_messages != 0 {
                trace!("PeekMessage found messages waiting");
            }

            // Process all pending messages (non-blocking)
            // Use -1 as hwnd to get thread messages (not bound to a window)
            // Actually, 0 should work for all thread messages including window messages
            while PeekMessageW(&mut msg, 0 as _, 0, 0, PM_REMOVE) != 0 {
                count += 1;

                // Log interesting messages (skip noisy ones)
                match msg.message {
                    WM_HOTKEY => {
                        info!("=== WM_HOTKEY MESSAGE RECEIVED ===");
                        info!("hwnd: {:?}", msg.hwnd);
                        info!("wParam (hotkey id): {}", msg.wParam);
                        info!("lParam: {:#x}", msg.lParam);
                    }
                    WM_QUIT => {
                        info!("WM_QUIT received");
                    }
                    WM_TIMER | WM_NULL => {
                        // Skip noisy messages
                    }
                    _ => {
                        // Log other messages at debug level to see what we're getting
                        debug!("Windows message: {} (hwnd: {:?}, wParam: {}, lParam: {})",
                               msg.message, msg.hwnd, msg.wParam, msg.lParam);
                    }
                }

                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }
        count
    }

    #[cfg(not(target_os = "windows"))]
    fn pump_messages() -> u32 {
        // No-op on non-Windows platforms
        0
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
    ProjectDirs::from("com", "voclaude", "Voclaude")
        .map(|dirs| dirs.data_dir().join("transcripts"))
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
        Command::new("explorer").arg(path).spawn()?;
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

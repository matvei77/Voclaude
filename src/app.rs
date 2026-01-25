//! Main application orchestration.

use crate::audio::AudioCapture;
use crate::audio::WHISPER_SAMPLE_RATE;
use crate::config::Config;
use crate::history::{AudioMetadata, HistoryEntry, HistoryStore};
use crate::hotkey::HotkeyManager;
use crate::inference::{InferenceProgress, InferenceStage, WhisperEngine};
use crate::tray::TrayManager;
use crate::ui::{LogBuffer, UiManager};

use crossbeam_channel::{bounded, Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};
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
    /// Quit requested
    Quit,
    /// Inference progress update
    InferenceProgress(InferenceProgress),
    /// Transcription completed
    TranscriptionComplete(Result<String, String>),
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

#[derive(Debug)]
enum InferenceCommand {
    Transcribe(Vec<f32>),
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
        let (event_tx, event_rx) = bounded::<AppEvent>(32);
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

        // History storage
        let (history_update_tx, history_update_rx) = bounded::<HistoryEntry>(32);
        let mut history = HistoryStore::load(self.config.history_max_entries, history_update_tx)?;
        info!("History loaded: {} entries", history.len());
        for entry in history.entries() {
            self.ui.push_history(entry.text.clone());
        }

        // Track last activity for idle unload
        let mut last_activity = Instant::now();
        let mut idle_unload_requested = false;
        let mut pending_audio_metadata: Option<AudioMetadata> = None;
        let mut last_progress_stage: Option<InferenceStage> = None;

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
                                        continue;
                                    }
                                    self.state = AppState::Recording;
                                    pending_audio_metadata = None;
                                    last_progress_stage = None;
                                    _tray.set_state(AppState::Recording);
                                }
                                AppState::Recording => {
                                    info!("Stopping recording...");
                                    match audio.stop() {
                                        Ok(samples) => {
                                            if samples.is_empty() {
                                                warn!("No audio recorded");
                                                self.state = AppState::Idle;
                                                _tray.set_state(AppState::Idle);
                                                continue;
                                            }

                                            let sample_count = samples.len();
                                            info!("Got {} samples, transcribing...", sample_count);
                                            self.state = AppState::Transcribing;
                                            pending_audio_metadata = Some(AudioMetadata::from_samples(
                                                sample_count,
                                                WHISPER_SAMPLE_RATE,
                                            ));
                                            _tray.set_state(AppState::Transcribing);

                                            if let Err(err) = inference_tx.send(InferenceCommand::Transcribe(samples)) {
                                                error!("Failed to start transcription: {}", err);
                                                self.state = AppState::Idle;
                                                pending_audio_metadata = None;
                                                _tray.set_state(AppState::Idle);
                                            }
                                        }
                                        Err(e) => {
                                            error!("Failed to stop recording: {}", e);
                                            self.state = AppState::Idle;
                                            _tray.set_state(AppState::Idle);
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

                            if last_progress_stage != Some(progress.stage) {
                                notifications.notify(&progress.message);
                                last_progress_stage = Some(progress.stage);
                            }
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
                                    if !history_text.is_empty() {
                                        let metadata = pending_audio_metadata.take();
                                        if let Err(e) = history.append(history_text, metadata) {
                                            error!("Failed to append history: {}", e);
                                        }
                                    }

                                    // Copy to clipboard
                                    if let Err(e) = clipboard.set_text(&formatted) {
                                        error!("Failed to copy to clipboard: {}", e);
                                    } else {
                                        info!("Copied to clipboard!");
                                        notifications.notify("Transcription copied to clipboard");
                                    }
                                }
                                Err(e) => {
                                    error!("Transcription failed: {}", e);
                                    notifications.notify("Transcription failed");
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
                        }
                        AppEvent::ToggleHistoryWindow => {
                            self.ui.toggle();
                        }
                        AppEvent::ShowHistoryWindow => {
                            self.ui.show();
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
                InferenceCommand::Transcribe(samples) => {
                    let mut callback = |progress: InferenceProgress| {
                        let _ = event_tx.send(AppEvent::InferenceProgress(progress));
                    };
                    let result =
                        engine.transcribe_with_progress(&samples, Some(&mut callback));
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

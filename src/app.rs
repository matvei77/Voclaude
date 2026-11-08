//! Main application orchestration.

use crate::audio::segmenter::segment_all;
use crate::audio::{AudioCapture, SegmentBounds, SegmentReady, SegmenterConfig, TARGET_SAMPLE_RATE};
use crate::config::Config;
use crate::history::{AudioMetadata, HistoryEntry, HistoryStore};
use crate::hotkey::HotkeyManager;
use crate::inference::{InferenceProgress, InferenceStage};
use crate::tray::TrayManager;
use crate::console::ipc::{FromConsole, Status as UiStatus};
use crate::console_host::ConsoleHost;
use crate::session::{RecordingSession, SessionStore};
use crate::worker::{read_f32_range, InferenceCommand};

use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use crate::config::project_dirs;
use std::collections::VecDeque;
use std::fs;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::io::Write;
use std::path::{Path, PathBuf};
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
    /// Show the console window (hotkey, tray click, a second launch)
    OpenConsole,
    /// A command from the console window
    ConsoleCommand(FromConsole),
    /// config.toml changed on disk but could not be applied
    ConfigError(String),
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
    /// A live segment closed on the audio writer thread
    SegmentReady(SegmentReady),
    /// One segment finished transcribing (or failed)
    SegmentTranscribed {
        session_id: String,
        index: usize,
        result: Result<String, String>,
    },
    /// Tray: re-run whatever the last recording is still missing
    RecoverLastRecording,
    /// API: transcribe samples `[start, end)` of a 16 kHz f32 file; reply on the channel
    ApiTranscribeRange {
        path: PathBuf,
        start: usize,
        end: usize,
        reply: Sender<Result<String, String>>,
    },
    /// API: a file finished; record it in history
    ApiTranscribed {
        source: String,
        text: String,
        sample_count: usize,
    },
}

impl AppEvent {
    /// The variant name, for logs. S-1: payloads hold dictated text (and the
    /// config holds the API token), so the log never gets `{:?}` of an event.
    fn kind(&self) -> &'static str {
        match self {
            AppEvent::HotkeyPressed => "HotkeyPressed",
            AppEvent::OpenConsole => "OpenConsole",
            AppEvent::ConsoleCommand(_) => "ConsoleCommand",
            AppEvent::ConfigError(_) => "ConfigError",
            AppEvent::Quit => "Quit",
            AppEvent::InferenceProgress(_) => "InferenceProgress",
            AppEvent::TranscriptionComplete(Ok(_)) => "TranscriptionComplete(ok)",
            AppEvent::TranscriptionComplete(Err(_)) => "TranscriptionComplete(error)",
            AppEvent::InferenceEngineInfo { .. } => "InferenceEngineInfo",
            AppEvent::HistoryUpdated(_) => "HistoryUpdated",
            AppEvent::ConfigReloaded(_) => "ConfigReloaded",
            AppEvent::SegmentReady(_) => "SegmentReady",
            AppEvent::SegmentTranscribed { result: Ok(_), .. } => "SegmentTranscribed(ok)",
            AppEvent::SegmentTranscribed { result: Err(_), .. } => "SegmentTranscribed(error)",
            AppEvent::RecoverLastRecording => "RecoverLastRecording",
            AppEvent::ApiTranscribeRange { .. } => "ApiTranscribeRange",
            AppEvent::ApiTranscribed { .. } => "ApiTranscribed",
        }
    }
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

/// Attempts per segment before it is declared failed.
const SEGMENT_MAX_ATTEMPTS: u8 = 2;

/// App-side view of one live segment.
#[derive(Debug, Clone)]
struct LiveSegment {
    bounds: SegmentBounds,
    text: Option<String>,
    failed: Option<String>,
    attempts: u8,
    in_flight: bool,
}

/// The recording currently being captured or finished.
#[derive(Debug)]
struct LiveSession {
    id: String,
    audio_path: PathBuf,
    segments: Vec<LiveSegment>,
    /// True once the recorder has stopped and every segment is known.
    recording_done: bool,
    started_at: Instant,
}

impl LiveSession {
    fn new(id: String, audio_path: PathBuf) -> Self {
        Self {
            id,
            audio_path,
            segments: Vec::new(),
            recording_done: false,
            started_at: Instant::now(),
        }
    }

    fn from_session(session: &RecordingSession, audio_path: PathBuf) -> Self {
        let mut live = Self::new(session.id.clone(), audio_path);
        for rec in &session.segments {
            while live.segments.len() < rec.index {
                // Gap in the journal — should not happen, keep indices aligned.
                live.segments.push(LiveSegment {
                    bounds: SegmentBounds { start: 0, end: 0 },
                    text: Some(String::new()),
                    failed: None,
                    attempts: 0,
                    in_flight: false,
                });
            }
            live.segments.push(LiveSegment {
                bounds: SegmentBounds { start: rec.start_sample, end: rec.end_sample },
                text: rec.text.clone(),
                failed: None,
                attempts: 0,
                in_flight: false,
            });
        }
        live
    }

    /// Add a segment if its index is new. Returns true when added.
    fn add_segment(&mut self, index: usize, bounds: SegmentBounds) -> bool {
        if index == self.segments.len() {
            self.segments.push(LiveSegment { bounds, text: None, failed: None, attempts: 0, in_flight: false });
            true
        } else {
            false
        }
    }

    fn done_count(&self) -> usize {
        self.segments.iter().filter(|s| s.text.is_some()).count()
    }

    fn all_done(&self) -> bool {
        self.segments.iter().all(|s| s.text.is_some())
    }

    /// Every segment has either text or a final failure.
    fn all_settled(&self) -> bool {
        self.segments.iter().all(|s| s.text.is_some() || s.failed.is_some())
    }

    fn any_failed(&self) -> bool {
        self.segments.iter().any(|s| s.failed.is_some())
    }

    fn joined_text(&self) -> String {
        let parts: Vec<&str> = self
            .segments
            .iter()
            .filter_map(|s| s.text.as_deref())
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .collect();
        parts.join(" ")
    }

    fn covered_end(&self) -> usize {
        self.segments.iter().map(|s| s.bounds.end).max().unwrap_or(0)
    }

    fn pending_indices(&self) -> Vec<usize> {
        self.segments
            .iter()
            .enumerate()
            .filter(|(_, s)| s.text.is_none() && s.failed.is_none() && !s.in_flight)
            .map(|(i, _)| i)
            .collect()
    }
}

fn partial_transcript_path(session_id: &str) -> Option<PathBuf> {
    transcripts_dir().map(|dir| dir.join(format!("{}.partial.txt", session_id)))
}

/// Build a `LiveSession` for a journaled session whose audio is on disk,
/// segmenting any audio the journal does not cover. Returns the session and
/// the bounds of segments that were newly created (so the caller can journal them).
fn recover_live_session(
    session: &RecordingSession,
    audio_path: &Path,
    seg_cfg: SegmenterConfig,
) -> Result<(LiveSession, Vec<(usize, SegmentBounds)>, usize), String> {
    let total = fs::metadata(audio_path)
        .map_err(|e| format!("audio file: {}", e))?
        .len() as usize
        / 4;
    let mut live = LiveSession::from_session(session, audio_path.to_path_buf());
    let covered = live.covered_end().min(total);
    let mut new_bounds = Vec::new();
    if total > covered {
        let rest = read_f32_range(audio_path, covered, total)?;
        for b in segment_all(&rest, seg_cfg) {
            let bounds = SegmentBounds { start: b.start + covered, end: b.end + covered };
            let index = live.segments.len();
            live.add_segment(index, bounds);
            new_bounds.push((index, bounds));
        }
    }
    live.recording_done = true;
    Ok((live, new_bounds, total))
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
    console: ConsoleHost,
    /// config.toml failed to load at startup (the defaults are in use).
    startup_config_error: Option<String>,
}

impl App {
    /// Run the application
    pub fn run(config: Config, startup_config_error: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
        let (event_tx, event_rx) = unbounded::<AppEvent>();
        let console = ConsoleHost::new(event_tx.clone());

        let mut app = App {
            config,
            state: AppState::Idle,
            event_tx,
            event_rx,
            is_running: true,
            console,
            startup_config_error,
        };

        app.run_event_loop()
    }

    fn run_event_loop(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        use std::sync::atomic::Ordering;
        info!("Initializing components...");

        // Initialize tray icon - MUST keep alive!
        let _tray = TrayManager::new(
            self.event_tx.clone(),
            &self.config.hotkey,
            &self.config.history_hotkey,
        )?;
        _tray.set_engine_status(&format!(
            "{} · {} (loads on first use)",
            self.config.model_display_name(),
            if self.config.use_gpu { "GPU" } else { "CPU" }
        ));
        info!("Tray icon ready");

        // Initialize hotkey listener - single unified thread for all _hotkeys
        // to avoid GlobalHotKeyEvent receiver contention.
        let mut _hotkeys = Some(HotkeyManager::new_multi(
            &[
                (&self.config.hotkey, AppEvent::HotkeyPressed),
                (&self.config.history_hotkey, AppEvent::OpenConsole),
            ],
            self.event_tx.clone(),
        )?);
        // A second launch (Start menu, `voclaude console`) opens the window.
        crate::instance::listen_for_open(self.event_tx.clone(), AppEvent::OpenConsole);
        info!("Hotkeys registered: {} (record), {} (window)", self.config.hotkey, self.config.history_hotkey);

        // Initialize audio capture (segments are announced through event_tx)
        let (level_tx, level_rx) = bounded::<f32>(64);
        let (segment_tx, segment_rx) = unbounded::<SegmentReady>();
        let audio = AudioCapture::new(Some(level_tx), Some(segment_tx), self.config.segmenter_config())?;
        info!("Audio capture ready");

        // No preflight: the model loads when the first recording starts. A
        // failure shows in the tray tooltip and the History window status bar.

        // Initialize inference worker (lazy - loads model on demand)
        let inference_shutdown = Arc::new(AtomicBool::new(false));
        let (mut inference_tx, mut inference_handle) =
            Self::spawn_inference_worker(self.event_tx.clone(), self.config.clone(), inference_shutdown.clone());
        info!("Inference worker ready (model will load on first use)");

        // Local HTTP API for other programs (127.0.0.1 only).
        let mut api_error: Option<String> = None;
        let mut api_server = start_api(&self.config, &self.event_tx, &mut api_error);

        // Clipboard
        let mut clipboard = arboard::Clipboard::new()?;

        let mut notifications = NotificationManager::new(self.config.show_notifications);

        // History storage
        let (history_update_tx, history_update_rx) = bounded::<HistoryEntry>(32);
        let mut history = HistoryStore::load(self.config.history_max_entries, self.config.audio_retention_count, history_update_tx)?;
        info!("History loaded: {} entries", history.len());

        let mut ui_status = UiStatus {
            state: "Idle".to_string(),
            model: self.config.model_display_name(),
            use_gpu: self.config.use_gpu,
            history_count: history.len(),
            input_device: Some(audio.device_name()),
            api_port: api_server.as_ref().map(|_| self.config.api_port),
            api_error: api_error.clone(),
            config_error: self.startup_config_error.clone(),
            record_hotkey: self.config.hotkey.clone(),
            console_hotkey: self.config.history_hotkey.clone(),
            ..Default::default()
        };
        if let Some(err) = &self.startup_config_error {
            _tray.set_tooltip_message(&shorten_error_message(&format!("config.toml has an error, using defaults: {}", err), 120));
        }
        self.console.set_status(&ui_status);

        let mut session_store = SessionStore::load()?;

        // Track last activity for idle unload
        let mut last_activity = Instant::now();
        let mut idle_unload_requested = false;
        let mut pending_audio_metadata: Option<AudioMetadata> = None;
        let mut last_progress_stage: Option<InferenceStage> = None;
        let mut transcribe_started_at: Option<Instant> = None;
        // last_transcript.txt is rewritten after every transcription, so the
        // tray's "Open Last Transcript" works after a restart too.

        // Live (streaming) session bookkeeping
        let mut live: Option<LiveSession> = None;
        let mut dispatch_queue: VecDeque<usize> = VecDeque::new();
        let mut pending_audio_path: Option<PathBuf> = None;
        let mut last_recording_tick = Instant::now();
        let mut recover_request_at_start = false;

        if let Some(session) = session_store.current().cloned() {
            if session.is_recoverable() {
                match session.audio_path() {
                    Some(path) if path.exists() => {
                        info!("Recovering previous session: {}", session.id);
                        recover_request_at_start = true;
                    }
                    _ => {
                        warn!("Recovery audio file missing; marking session failed");
                        let _ = session_store.mark_failed("Recovery audio file missing".to_string());
                    }
                }
            }
        }
        _tray.set_recoverable(recoverable_now(&session_store));
        ui_status.recoverable = recoverable_now(&session_store);
        self.console.set_status(&ui_status);
        if recover_request_at_start {
            let _ = self.event_tx.send(AppEvent::RecoverLastRecording);
        }

        info!("=== VOCLAUDE READY ===");
        info!("Press {} to start recording", self.config.hotkey);
        info!("Press {} to open the window", self.config.history_hotkey);
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
        let mut last_console_poll = Instant::now();
        let mut messages_pumped: u64 = 0;

        // Watchdog respawn counter and backoff state
        let mut watchdog_respawn_count: u32 = 0;
        // E-3: Deferred respawn (avoid blocking main thread with sleep)
        let mut deferred_respawn_at: Option<Instant> = None;
        // E-4: Keep sentinel receiver alive so send() doesn't immediately error
        let mut _sentinel_rx: Option<Receiver<InferenceCommand>> = None;

        // Settings watcher: config.toml is reloaded whenever it changes on disk
        // (Edit Settings, `voclaude phrases add`, any editor), for the whole run.
        let settings_watcher_cancel: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
        if let Some(path) = Config::config_path() {
            spawn_settings_watcher(path, self.event_tx.clone(), Arc::clone(&settings_watcher_cancel));
        }

        // Main event loop - runs on main thread with Windows message pump
        while self.is_running {
            loop_count += 1;

            // Pump Windows messages (required for _hotkeys and tray to work)
            let (pumped, os_quit) = Self::pump_messages();
            messages_pumped += pumped as u64;
            // E-1: Handle OS shutdown/logoff via WM_QUIT
            if os_quit {
                info!("OS requested shutdown via WM_QUIT");
                inference_shutdown.store(true, Ordering::SeqCst);
                let _ = inference_tx.try_send(InferenceCommand::Shutdown);
                settings_watcher_cancel.store(true, Ordering::Relaxed);
                self.is_running = false;
            }

            // Log status every 10 seconds
            if last_status_log.elapsed() > Duration::from_secs(10) {
                debug!("=== MAIN LOOP STATUS ===");
                debug!("Loop iterations: {}", loop_count);
                debug!("Messages pumped total: {}", messages_pumped);
                debug!("Current state: {:?}", self.state);
                debug!("Event channel len: {}", self.event_rx.len());
                last_status_log = Instant::now();
            }

            if last_console_poll.elapsed() >= Duration::from_millis(500) {
                last_console_poll = Instant::now();
                self.console.poll();
            }

            // E-3: Check deferred respawn (non-blocking, no sleep on main thread)
            if let Some(respawn_at) = deferred_respawn_at {
                if Instant::now() >= respawn_at {
                    deferred_respawn_at = None;
                    let (new_tx, new_handle) =
                        Self::spawn_inference_worker(self.event_tx.clone(), self.config.clone(), inference_shutdown.clone());
                    inference_tx = new_tx;
                    inference_handle = new_handle;
                    info!("Inference worker respawned (attempt {})", watchdog_respawn_count);
                    notifications.notify("Inference engine restarted after error");
                    // Anything that was in flight died with the old worker: re-queue it.
                    if let Some(live) = live.as_mut() {
                        for seg in live.segments.iter_mut() {
                            if seg.in_flight {
                                seg.in_flight = false;
                                seg.attempts = seg.attempts.saturating_add(1);
                            }
                        }
                        dispatch_queue.clear();
                        dispatch_queue.extend(live.pending_indices());
                    }
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
                if self.state == AppState::Transcribing && live.is_none() {
                    self.state = AppState::Idle;
                    pending_audio_metadata = None;
                    _tray.set_state(AppState::Idle);
                    ui_status.state = "Idle".to_string();
                    say_error(&mut ui_status, "Inference engine restarted after error".to_string());
                    self.console.set_status(&ui_status);
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
                    ui_status.model_loaded = false;
                    self.console.set_status(&ui_status);
                    idle_unload_requested = true;
                }
            }

            while let Ok(entry) = history_update_rx.try_recv() {
                if let Err(err) = self.event_tx.try_send(AppEvent::HistoryUpdated(entry)) {
                    debug!("Dropping history update event: {}", err);
                }
            }

            // Live segments closed by the writer thread
            while let Ok(seg) = segment_rx.try_recv() {
                let _ = self.event_tx.send(AppEvent::SegmentReady(seg));
            }

            // Feed queued segments to the inference worker (bounded channel: keep
            // what does not fit for the next iteration).
            if let Some(live_ref) = live.as_mut() {
                while let Some(&index) = dispatch_queue.front() {
                    let Some(seg) = live_ref.segments.get(index) else {
                        dispatch_queue.pop_front();
                        continue;
                    };
                    if seg.text.is_some() || seg.failed.is_some() || seg.in_flight {
                        dispatch_queue.pop_front();
                        continue;
                    }
                    let cmd = InferenceCommand::TranscribeSegment {
                        session_id: live_ref.id.clone(),
                        index,
                        path: live_ref.audio_path.clone(),
                        start: seg.bounds.start,
                        end: seg.bounds.end,
                        use_context: self.config.segment_context,
                    };
                    match inference_tx.try_send(cmd) {
                        Ok(()) => {
                            dispatch_queue.pop_front();
                            live_ref.segments[index].in_flight = true;
                            live_ref.segments[index].attempts += 1;
                        }
                        Err(crossbeam_channel::TrySendError::Full(_)) => break,
                        Err(crossbeam_channel::TrySendError::Disconnected(_)) => {
                            warn!("Inference worker channel closed; segment {} waits for respawn", index);
                            break;
                        }
                    }
                }
            }

            // Recording status tick (elapsed time + segment progress)
            if self.state == AppState::Recording && last_recording_tick.elapsed() >= Duration::from_secs(1) {
                last_recording_tick = Instant::now();
                if let Some(live_ref) = live.as_ref() {
                    let secs = live_ref.started_at.elapsed().as_secs();
                    let total = live_ref.segments.len();
                    let msg = if total == 0 {
                        format!("Recording {}:{:02}", secs / 60, secs % 60)
                    } else {
                        format!(
                            "Recording {}:{:02} · {}/{} segments transcribed",
                            secs / 60,
                            secs % 60,
                            live_ref.done_count(),
                            total
                        )
                    };
                    say(&mut ui_status, msg);
                    self.console.set_status(&ui_status);
                }
            }

            let mut latest_level: Option<f32> = None;
            while let Ok(level) = level_rx.try_recv() {
                latest_level = Some(level.clamp(0.0, 1.0));
            }
            if let Some(level) = latest_level {
                if self.state == AppState::Recording {
                    self.console.set_level(level);
                }
            }

            // Process events (non-blocking)
            match self.event_rx.try_recv() {
                Ok(event) => {
                    debug!("Event: {} ({:?} since last activity)", event.kind(), last_activity.elapsed());

                    match event {
                        AppEvent::HotkeyPressed => {
                            last_activity = Instant::now();
                            idle_unload_requested = false;
                            info!("Processing HotkeyPressed, current state: {:?}", self.state);
                            match self.state {
                                AppState::Idle => {
                                    info!("Starting recording...");
                                    let audio_path = match audio.start() {
                                        Ok(path) => path,
                                        Err(e) => {
                                            error!("Failed to start recording: {}", e);
                                            _tray.set_state(AppState::Idle);
                                            _tray.set_tooltip_message(&shorten_error_message(&e.to_string(), 100));
                                            notifications.notify("Failed to start recording");
                                            ui_status.state = "Idle".to_string();
                                            say_error(&mut ui_status, format!("Recording failed: {}", e));
                                            self.console.set_status(&ui_status);
                                            continue;
                                        }
                                    };
                                    // Update displayed device name (may have changed since last recording)
                                    ui_status.input_device = Some(audio.device_name());
                                    let session_id = match session_store.start(&audio_path) {
                                        Ok(session) => session.id,
                                        Err(err) => {
                                            warn!("Failed to start session metadata: {}", err);
                                            fallback_session_id()
                                        }
                                    };
                                    live = Some(LiveSession::new(session_id, audio_path));
                                    dispatch_queue.clear();
                                    // Load the model while the user is still speaking.
                                    if let Err(err) = inference_tx.try_send(InferenceCommand::Preload) {
                                        debug!("Preload not queued: {}", err);
                                    }
                                    self.state = AppState::Recording;
                                    pending_audio_metadata = None;
                                    last_progress_stage = None;
                                    last_recording_tick = Instant::now();
                                    _tray.set_state(AppState::Recording);
                                    ui_status.state = "Recording".to_string();
                                    say(&mut ui_status, "Recording...".to_string());
                                    self.console.set_status(&ui_status);
                                }
                                AppState::Recording => {
                                    info!("Stopping recording...");
                                    match audio.stop() {
                                        Ok(recording) => {
                                            if recording.is_empty() {
                                                warn!("No audio recorded");
                                                let _ = session_store.mark_aborted(Some("no_audio".to_string()));
                                                live = None;
                                                dispatch_queue.clear();
                                                self.state = AppState::Idle;
                                                _tray.set_state(AppState::Idle);
                                                ui_status.state = "Idle".to_string();
                                                say(&mut ui_status, "No audio captured".to_string());
                                                self.console.set_status(&ui_status);
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
                                            say(&mut ui_status, "Transcribing audio...".to_string());
                                            self.console.set_status(&ui_status);

                                            // E-7: Track audio file path for cleanup
                                            pending_audio_path = Some(recording.path.clone());

                                            let streaming_ok = self.config.streaming
                                                && live.as_ref().map(|l| l.audio_path == recording.path).unwrap_or(false);
                                            if streaming_ok {
                                                let live_ref = live.as_mut().unwrap();
                                                // Register every segment the writer produced (the tail is new;
                                                // earlier ones may already be known from SegmentReady events).
                                                for (index, bounds) in recording.segments.iter().enumerate() {
                                                    if live_ref.add_segment(index, *bounds) {
                                                        let _ = session_store.set_segment_bounds(index, bounds.start, bounds.end);
                                                    }
                                                }
                                                live_ref.recording_done = true;
                                                dispatch_queue.clear();
                                                dispatch_queue.extend(live_ref.pending_indices());
                                                info!(
                                                    "Recording stopped: {} segments, {} already transcribed, {} to go",
                                                    live_ref.segments.len(),
                                                    live_ref.done_count(),
                                                    dispatch_queue.len()
                                                );
                                                say(&mut ui_status, format!(
                                                    "Finishing ({}/{} segments)...",
                                                    live_ref.done_count(),
                                                    live_ref.segments.len()
                                                ));
                                                self.console.set_status(&ui_status);
                                                if live_ref.all_done() {
                                                    let text = live_ref.joined_text();
                                                    let _ = self.event_tx.send(AppEvent::TranscriptionComplete(Ok(text)));
                                                }
                                            } else {
                                                live = None;
                                                dispatch_queue.clear();
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
                                                    say_error(&mut ui_status, "Transcription failed to start".to_string());
                                                    self.console.set_status(&ui_status);
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            error!("Failed to stop recording: {}", e);
                                            let _ = session_store
                                                .mark_failed("Failed to stop recording".to_string());
                                            live = None;
                                            dispatch_queue.clear();
                                            self.state = AppState::Idle;
                                            _tray.set_state(AppState::Idle);
                                            ui_status.state = "Idle".to_string();
                                            say_error(&mut ui_status, "Failed to stop recording".to_string());
                                            self.console.set_status(&ui_status);
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
                            // (or preloading during a recording).
                            if self.state == AppState::Transcribing
                                || (self.state == AppState::Recording && progress.stage == InferenceStage::LoadingModel)
                            {
                                _tray.set_progress(&progress.message);
                                say(&mut ui_status, progress.message.clone());
                                self.console.set_status(&ui_status);
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
                            ui_status.model_loaded = true;
                            ui_status.model = model;
                            debug!("Model size: {} MB", model_size_mb);
                            _tray.set_engine_status(&engine_status_line(&ui_status.model, using_gpu));
                            if self.config.use_gpu && !using_gpu {
                                say_error(&mut ui_status, "GPU init failed; using CPU".to_string());
                            }
                            self.console.set_status(&ui_status);
                        }
                        AppEvent::TranscriptionComplete(result) => {
                            // E-6: Guard against stale results from respawned worker
                            if self.state != AppState::Transcribing {
                                warn!("Ignoring stale TranscriptionComplete (state={:?})", self.state);
                                continue;
                            }
                            let stop_to_result_secs = transcribe_started_at
                                .map(|t| t.elapsed().as_secs_f64())
                                .unwrap_or(0.0);
                            last_activity = Instant::now();
                            idle_unload_requested = false;
                            // Reset respawn counter on successful transcription
                            if matches!(result, Ok(_)) {
                                watchdog_respawn_count = 0;
                            }
                            // E-7: Audio file retention — keep recordings on disk.
                            // On success: store path in history entry; old files are
                            // pruned by HistoryStore::apply_audio_retention.
                            // On failure: always keep the audio file for recovery.
                            let completed_audio_path = pending_audio_path.take();
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

                                    // last_transcript.txt always holds the newest text, so the
                                    // tray's "Open Last Transcript" works across restarts.
                                    if !history_text.is_empty() {
                                        if let Some(path) = last_transcript_output_path() {
                                            if let Err(err) = write_transcript_file(&path, &history_text) {
                                                warn!("Failed to save last transcript: {}", err);
                                            }
                                        }
                                    }
                                    let mut history_entry_id = None;
                                    let mut history_error = None;
                                    if !history_text.is_empty() {
                                        let metadata = pending_audio_metadata.take();
                                        let audio_file = completed_audio_path
                                            .as_ref()
                                            .map(|p| p.to_string_lossy().to_string());
                                        match history.append(history_text.clone(), metadata, audio_file) {
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
                                        say_error(&mut ui_status, "Clipboard copy failed".to_string());
                                        self.console.set_status(&ui_status);
                                    } else {
                                        info!("Copied to clipboard!");
                                        notifications.notify(notify_message);
                                        ui_status.state = "Idle".to_string();
                                        say(&mut ui_status, notify_message.to_string());
                                        self.console.set_status(&ui_status);
                                    }
                                }
                                Err(e) => {
                                    error!("Transcription failed: {}", e);
                                    let _ = session_store
                                        .mark_failed(format!("Transcription failed: {}", e));
                                    let short = shorten_error_message(&e, 140);
                                    notifications.notify(&format!("Transcription failed: {}", short));
                                    ui_status.state = "Idle".to_string();
                                    say_error(&mut ui_status, format!("Transcription failed: {}", short));
                                    self.console.set_status(&ui_status);
                                }
                            }

                            if let Some(finished) = live.take() {
                                if let Some(path) = partial_transcript_path(&finished.id) {
                                    let _ = fs::remove_file(path);
                                }
                                info!(
                                    "Stop-to-result latency: {:.2}s ({} segments)",
                                    stop_to_result_secs,
                                    finished.segments.len()
                                );
                            }
                            dispatch_queue.clear();
                            self.state = AppState::Idle;
                            pending_audio_metadata = None;
                            last_progress_stage = None;
                            _tray.set_state(AppState::Idle);
                            _tray.set_recoverable(recoverable_now(&session_store));
                            ui_status.recoverable = recoverable_now(&session_store);
                            self.console.set_status(&ui_status);

                            // Model stays resident for fast re-use.
                            // The idle_unload_seconds timer (default 30s) handles
                            // VRAM reclamation when the user stops using the app.
                        }
                        AppEvent::SegmentReady(seg) => {
                            let Some(live_ref) = live.as_mut() else { continue };
                            if live_ref.audio_path != seg.path || live_ref.recording_done {
                                debug!("Ignoring segment for another recording: {:?}", seg.path);
                                continue;
                            }
                            if live_ref.add_segment(seg.index, seg.bounds) {
                                let _ = session_store.set_segment_bounds(seg.index, seg.bounds.start, seg.bounds.end);
                                if self.config.streaming {
                                    dispatch_queue.push_back(seg.index);
                                }
                                debug!(
                                    "Segment {} closed: {:.1}s-{:.1}s",
                                    seg.index,
                                    seg.bounds.start as f64 / TARGET_SAMPLE_RATE as f64,
                                    seg.bounds.end as f64 / TARGET_SAMPLE_RATE as f64
                                );
                            }
                        }
                        AppEvent::SegmentTranscribed { session_id, index, result } => {
                            let Some(live_ref) = live.as_mut() else {
                                debug!("Ignoring segment result for finished session {}", session_id);
                                continue;
                            };
                            if live_ref.id != session_id || index >= live_ref.segments.len() {
                                debug!("Ignoring stale segment result {}#{}", session_id, index);
                                continue;
                            }
                            last_activity = Instant::now();
                            live_ref.segments[index].in_flight = false;
                            match result {
                                Ok(text) => {
                                    info!("Segment {} transcribed: {} chars", index, text.chars().count());
                                    live_ref.segments[index].text = Some(text.clone());
                                    let _ = session_store.set_segment_result(index, Ok(text));
                                    if let Some(path) = partial_transcript_path(&live_ref.id) {
                                        if let Err(err) = write_transcript_file(&path, &live_ref.joined_text()) {
                                            warn!("Failed to write partial transcript: {}", err);
                                        }
                                    }
                                    watchdog_respawn_count = 0;
                                }
                                Err(err) => {
                                    warn!("Segment {} failed (attempt {}): {}", index, live_ref.segments[index].attempts, err);
                                    if live_ref.segments[index].attempts < SEGMENT_MAX_ATTEMPTS {
                                        dispatch_queue.push_back(index);
                                    } else {
                                        live_ref.segments[index].failed = Some(err.clone());
                                        let _ = session_store.set_segment_result(index, Err(err));
                                    }
                                }
                            }

                            let done = live_ref.done_count();
                            let total = live_ref.segments.len();
                            if self.state == AppState::Transcribing {
                                say(&mut ui_status, format!("Finishing ({}/{} segments)...", done, total));
                                self.console.set_status(&ui_status);
                            }

                            if live_ref.recording_done && live_ref.all_settled() {
                                if live_ref.any_failed() {
                                    let partial = live_ref.joined_text();
                                    let failed: Vec<String> = live_ref
                                        .segments
                                        .iter()
                                        .enumerate()
                                        .filter_map(|(i, s)| s.failed.as_ref().map(|e| format!("#{}: {}", i + 1, e)))
                                        .collect();
                                    if !partial.is_empty() {
                                        if let Err(e) = clipboard.set_text(&partial) {
                                            error!("Failed to copy partial transcript: {}", e);
                                        }
                                    }
                                    let msg = format!(
                                        "{} of {} segments failed ({}). Partial text copied; use tray > Recover Last Recording",
                                        failed.len(),
                                        total,
                                        failed.join("; ")
                                    );
                                    let _ = self.event_tx.send(AppEvent::TranscriptionComplete(Err(msg)));
                                } else {
                                    let text = live_ref.joined_text();
                                    let _ = self.event_tx.send(AppEvent::TranscriptionComplete(Ok(text)));
                                }
                            }
                        }
                        AppEvent::RecoverLastRecording => {
                            last_activity = Instant::now();
                            idle_unload_requested = false;
                            if self.state != AppState::Idle {
                                notifications.notify("Busy — try again when idle");
                                continue;
                            }
                            let Some(session) = session_store.current().cloned() else {
                                notifications.notify("No previous recording to recover");
                                continue;
                            };
                            if !session.can_retry() {
                                notifications.notify("Last recording already completed");
                                say(&mut ui_status, "Last recording already completed".to_string());
                                self.console.set_status(&ui_status);
                                continue;
                            }
                            let Some(path) = session.audio_path().filter(|p| p.exists()) else {
                                notifications.notify("Audio for the last recording is gone");
                                let _ = session_store.mark_failed("Recovery audio file missing".to_string());
                                _tray.set_recoverable(false);
                                ui_status.recoverable = false;
                                self.console.set_status(&ui_status);
                                continue;
                            };
                            match recover_live_session(&session, &path, self.config.segmenter_config()) {
                                Ok((recovered, new_bounds, total_samples)) => {
                                    info!(
                                        "Recovering session {}: {} segments ({} done, {} new)",
                                        recovered.id,
                                        recovered.segments.len(),
                                        recovered.done_count(),
                                        new_bounds.len()
                                    );
                                    for (index, bounds) in new_bounds {
                                        let _ = session_store.set_segment_bounds(index, bounds.start, bounds.end);
                                    }
                                    let metadata = AudioMetadata::from_samples(total_samples, TARGET_SAMPLE_RATE);
                                    let _ = session_store.reopen_for_retry(metadata.clone());
                                    pending_audio_metadata = Some(metadata);
                                    pending_audio_path = Some(path);
                                    transcribe_started_at = Some(Instant::now());
                                    dispatch_queue.clear();
                                    dispatch_queue.extend(recovered.pending_indices());
                                    let all_done = recovered.all_done();
                                    let text = recovered.joined_text();
                                    live = Some(recovered);
                                    self.state = AppState::Transcribing;
                                    _tray.set_state(AppState::Transcribing);
                                    ui_status.state = "Recovering".to_string();
                                    say(&mut ui_status, "Recovering last recording...".to_string());
                                    self.console.set_status(&ui_status);
                                    if all_done {
                                        let _ = self.event_tx.send(AppEvent::TranscriptionComplete(Ok(text)));
                                    }
                                }
                                Err(err) => {
                                    warn!("Recovery failed: {}", err);
                                    notifications.notify("Recovery failed");
                                    say_error(&mut ui_status, format!("Recovery failed: {}", err));
                                    self.console.set_status(&ui_status);
                                }
                            }
                        }
                        AppEvent::HistoryUpdated(entry) => {
                            debug!("History updated: {}", entry.id);
                            self.console.history_changed();
                            ui_status.history_count = history.len();
                            self.console.set_status(&ui_status);
                        }
                        AppEvent::OpenConsole => {
                            self.console.open();
                        }
                        AppEvent::ConsoleCommand(cmd) => match cmd {
                            FromConsole::ToggleRecording => {
                                let _ = self.event_tx.send(AppEvent::HotkeyPressed);
                            }
                            FromConsole::RecoverLast => {
                                let _ = self.event_tx.send(AppEvent::RecoverLastRecording);
                            }
                            FromConsole::DeleteHistory { id } => match history.delete(&id) {
                                Ok(true) => {
                                    info!("Deleted history entry {}", id);
                                    self.console.history_changed();
                                    ui_status.history_count = history.len();
                                    self.console.set_status(&ui_status);
                                }
                                Ok(false) => debug!("History entry {} not found", id),
                                Err(e) => {
                                    warn!("Could not delete history entry: {}", e);
                                    say_error(&mut ui_status, format!("Could not delete the dictation: {}", e));
                                    self.console.set_status(&ui_status);
                                }
                            },
                        },
                        AppEvent::ConfigError(err) => {
                            _tray.set_tooltip_message(&shorten_error_message(&format!("config.toml not applied: {}", err), 120));
                            ui_status.config_error = Some(err);
                            self.console.set_status(&ui_status);
                        }
                        AppEvent::ApiTranscribeRange { path, start, end, reply } => {
                            last_activity = Instant::now();
                            idle_unload_requested = false;
                            let cmd = InferenceCommand::TranscribeRange { path, start, end, reply };
                            if let Err(err) = inference_tx.try_send(cmd) {
                                use crossbeam_channel::TrySendError;
                                let (msg, cmd) = match err {
                                    TrySendError::Full(c) => ("inference queue is full, retry shortly".to_string(), c),
                                    TrySendError::Disconnected(c) => ("inference worker is unavailable".to_string(), c),
                                };
                                if let InferenceCommand::TranscribeRange { reply, .. } = cmd {
                                    let _ = reply.send(Err(msg));
                                }
                            }
                        }
                        AppEvent::ApiTranscribed { source, text, sample_count } => {
                            info!("API transcribed {} ({} chars)", source, text.len());
                            if !text.trim().is_empty() {
                                let meta = AudioMetadata::from_samples(sample_count, TARGET_SAMPLE_RATE);
                                if let Err(e) = history.append(text, Some(meta), None) {
                                    warn!("Failed to record API transcription in history: {}", e);
                                }
                            }
                        }
                        AppEvent::Quit => {
                            last_activity = Instant::now();
                            idle_unload_requested = false;
                            info!("Quit requested");
                            inference_shutdown.store(true, Ordering::SeqCst);
                            // E-2: Use try_send to avoid blocking main thread
                            let _ = inference_tx.try_send(InferenceCommand::Shutdown);
                            // E-13: Cancel settings watcher on quit
                            settings_watcher_cancel.store(true, Ordering::Relaxed);
                            self.console.shutdown();
                            self.is_running = false;
                        }
                        // E-8/S-7: Apply runtime-updatable config fields
                        AppEvent::ConfigReloaded(new_config) => {
                            let old = self.config.clone();
                            self.config.hotkey = new_config.hotkey.clone();
                            self.config.history_hotkey = new_config.history_hotkey.clone();
                            self.config.api_enabled = new_config.api_enabled;
                            self.config.api_port = new_config.api_port;
                            self.config.api_token = new_config.api_token.clone();
                            let mut problems: Vec<String> = Vec::new();
                            // Model/GPU settings take effect when the inference worker
                            // next starts (it is restarted below if it is idle).
                            // Apply runtime-updatable fields
                            self.config.idle_unload_seconds = new_config.idle_unload_seconds;
                            self.config.show_notifications = new_config.show_notifications;
                            self.config.capitalize_first = new_config.capitalize_first;
                            self.config.add_trailing_space = new_config.add_trailing_space;
                            self.config.max_new_tokens = new_config.max_new_tokens;
                            self.config.language = new_config.language.clone();
                            self.config.history_max_entries = new_config.history_max_entries;
                            self.config.require_gpu = new_config.require_gpu;
                            self.config.audio_retention_count = new_config.audio_retention_count;
                            history.set_audio_retention_count(new_config.audio_retention_count);
                            history.set_max_entries(new_config.history_max_entries);
                            self.config.model = new_config.model.clone();
                            self.config.model_path = new_config.model_path.clone();
                            self.config.model_tier = new_config.model_tier.clone();
                            self.config.use_gpu = new_config.use_gpu;
                            self.config.engine = new_config.engine.clone();
                            self.config.parakeet_model = new_config.parakeet_model.clone();
                            self.config.cpu_threads = new_config.cpu_threads;
                            self.config.parakeet_phrases = new_config.parakeet_phrases.clone();
                            self.config.parakeet_unsure_log = new_config.parakeet_unsure_log;
                            self.config.quantization = new_config.quantization.clone();
                            self.config.adaptive_max_new_tokens = new_config.adaptive_max_new_tokens;
                            self.config.max_chunk_seconds = new_config.max_chunk_seconds;
                            self.config.legacy_prompt = new_config.legacy_prompt;
                            self.config.streaming = new_config.streaming;
                            self.config.segment_context = new_config.segment_context;
                            self.config.segment_min_seconds = new_config.segment_min_seconds;
                            self.config.segment_max_seconds = new_config.segment_max_seconds;
                            self.config.segment_pause_seconds = new_config.segment_pause_seconds;
                            // After the segment fields: Parakeet cuts whole files with them.
                            if let Err(err) = inference_tx.try_send(InferenceCommand::UpdateConfig(self.config.clone())) {
                                warn!("Could not forward settings to the inference worker: {}", err);
                            }
                            audio.set_segmenter_config(self.config.segmenter_config());
                            notifications.enabled = new_config.show_notifications;

                            // Hotkeys apply at once: release the old ones, take the new.
                            if old.hotkey != self.config.hotkey || old.history_hotkey != self.config.history_hotkey {
                                _hotkeys = None;
                                let bind = |c: &Config| {
                                    HotkeyManager::new_multi(
                                        &[(&c.hotkey, AppEvent::HotkeyPressed), (&c.history_hotkey, AppEvent::OpenConsole)],
                                        self.event_tx.clone(),
                                    )
                                };
                                match bind(&self.config) {
                                    Ok(h) => {
                                        info!("Hotkeys now {} (record), {} (window)", self.config.hotkey, self.config.history_hotkey);
                                        _hotkeys = Some(h);
                                    }
                                    Err(e) => {
                                        warn!("New _hotkeys not registered: {}", e);
                                        problems.push(format!("_hotkeys not changed ({})", e));
                                        self.config.hotkey = old.hotkey.clone();
                                        self.config.history_hotkey = old.history_hotkey.clone();
                                        _hotkeys = bind(&self.config).ok();
                                    }
                                }
                                _tray.set_hotkeys(&self.config.hotkey, &self.config.history_hotkey);
                            }

                            // The API serves the model it was started with: restart it
                            // when its settings or the model change.
                            let api_changed = old.api_enabled != self.config.api_enabled
                                || old.api_port != self.config.api_port
                                || old.api_token != self.config.api_token
                                || old.active_model() != self.config.active_model()
                                || old.use_gpu != self.config.use_gpu;
                            if api_changed {
                                if let Some(mut server) = api_server.take() {
                                    server.stop();
                                }
                                api_server = start_api(&self.config, &self.event_tx, &mut api_error);
                                if let Some(e) = &api_error {
                                    problems.push(format!("API: {}", e));
                                }
                            }

                            // A new model loads at the next recording.
                            if old.active_model() != self.config.active_model() || old.use_gpu != self.config.use_gpu || old.engine != self.config.engine {
                                ui_status.model = self.config.model_display_name();
                                ui_status.use_gpu = self.config.use_gpu;
                                ui_status.model_loaded = false;
                            }
                            ui_status.api_port = api_server.as_ref().map(|_| self.config.api_port);
                            ui_status.api_error = api_error.clone();
                            ui_status.record_hotkey = self.config.hotkey.clone();
                            ui_status.console_hotkey = self.config.history_hotkey.clone();
                            if ui_status.config_error.take().is_some() {
                                _tray.set_state(self.state); // clears the error tooltip
                            }
                            if problems.is_empty() {
                                info!("Settings applied");
                                say(&mut ui_status, "Settings applied");
                            } else {
                                let msg = format!("Settings applied, except: {}", problems.join("; "));
                                warn!("{}", msg);
                                say_error(&mut ui_status, msg);
                            }
                            self.console.set_status(&ui_status);
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

        if let Some(mut server) = api_server.take() {
            server.stop();
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
        shutdown: Arc<AtomicBool>,
    ) -> (Sender<InferenceCommand>, thread::JoinHandle<()>) {
        crate::worker::spawn_proxy(event_tx, config, shutdown)
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
                        trace!("Windows message: {} (hwnd: {:?}, wParam: {}, lParam: {})",
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



/// Whether the tray's "Recover Last Recording" has a recording to re-run.
fn recoverable_now(store: &SessionStore) -> bool {
    store
        .current()
        .map(|session| session.can_retry() && session.audio_path().is_some_and(|p| p.exists()))
        .unwrap_or(false)
}

/// The first line of the tray menu: the worker's model label without the
/// backend wrapper, plus the device class.
fn engine_status_line(label: &str, using_gpu: bool) -> String {
    let short = label
        .strip_prefix("transcribe.cpp (")
        .and_then(|s| s.strip_suffix(')'))
        .unwrap_or(label);
    format!("{} · {}", short, if using_gpu { "GPU" } else { "CPU" })
}





/// Reload config.toml whenever its modification time changes, for the whole
/// run: every 3 s, one `metadata` call. A file that fails to load is reported
/// once per change and the running settings stay as they are.
fn spawn_settings_watcher(path: PathBuf, event_tx: Sender<AppEvent>, cancel: Arc<AtomicBool>) {
    use std::sync::atomic::Ordering;
    let mtime = |p: &Path| fs::metadata(p).and_then(|m| m.modified()).ok();
    let mut seen = mtime(&path);
    let spawned = thread::Builder::new().name("settings-watcher".into()).spawn(move || loop {
        thread::sleep(Duration::from_secs(3));
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let now = mtime(&path);
        if now == seen {
            continue;
        }
        seen = now;
        match Config::load() {
            Ok(new_config) => {
                info!("Config reloaded after a change on disk");
                if event_tx.send(AppEvent::ConfigReloaded(new_config)).is_err() {
                    break;
                }
            }
            Err(err) => {
                warn!("Config change not applied (invalid settings): {}", err);
                if event_tx.send(AppEvent::ConfigError(err.to_string())).is_err() {
                    break;
                }
            }
        }
    });
    if let Err(err) = spawned {
        warn!("Could not start the settings watcher: {}", err);
    }
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

/// Show `text` in the window's status strip.
fn say(status: &mut UiStatus, text: impl Into<String>) {
    status.last_message = Some(text.into());
    status.last_message_is_error = false;
}

/// Show a failure in the window's status strip.
fn say_error(status: &mut UiStatus, text: impl Into<String>) {
    status.last_message = Some(text.into());
    status.last_message_is_error = true;
}

/// Start the local API if it is enabled. `error` says why it is not listening.
fn start_api(config: &Config, event_tx: &Sender<AppEvent>, error: &mut Option<String>) -> Option<crate::api::ApiServer> {
    *error = None;
    if !config.api_enabled {
        return None;
    }
    match crate::api::ApiServer::start(config, event_tx.clone()) {
        Ok(server) => Some(server),
        Err(e) => {
            warn!("API server not started: {}", e);
            *error = Some(e);
            None
        }
    }
}

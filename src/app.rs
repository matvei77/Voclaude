//! Main application orchestration.

use crate::audio::AudioCapture;
use crate::config::Config;
use crate::hotkey::HotkeyManager;
use crate::inference::WhisperEngine;
use crate::tray::TrayManager;

use crossbeam_channel::{bounded, Receiver, Sender};
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn, trace};

/// Application events
#[derive(Debug, Clone)]
pub enum AppEvent {
    /// Hotkey pressed - toggle recording
    HotkeyPressed,
    /// Quit requested
    Quit,
}

/// Application state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppState {
    Idle,
    Recording,
    Transcribing,
}

/// Main application
pub struct App {
    config: Config,
    state: AppState,
    event_tx: Sender<AppEvent>,
    event_rx: Receiver<AppEvent>,
    is_running: bool,
}

impl App {
    /// Run the application
    pub fn run(config: Config) -> Result<(), Box<dyn std::error::Error>> {
        let (event_tx, event_rx) = bounded::<AppEvent>(32);

        let mut app = App {
            config,
            state: AppState::Idle,
            event_tx,
            event_rx,
            is_running: true,
        };

        app.run_event_loop()
    }

    fn run_event_loop(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        info!("Initializing components...");

        // Initialize tray icon - MUST keep alive!
        let _tray = TrayManager::new(self.event_tx.clone())?;
        info!("Tray icon ready");

        // Initialize hotkey listener - MUST keep alive!
        let _hotkey = HotkeyManager::new(&self.config.hotkey, self.event_tx.clone())?;
        info!("Hotkey registered: {}", self.config.hotkey);

        // Initialize audio capture
        let audio = AudioCapture::new()?;
        info!("Audio capture ready");

        // Initialize Whisper inference engine (lazy - doesn't load model yet)
        let mut inference = WhisperEngine::new()?;
        info!("Whisper engine ready (model will load on first use)");

        // Clipboard
        let mut clipboard = arboard::Clipboard::new()?;

        // Track last activity for idle unload
        let mut last_activity = Instant::now();

        info!("=== VOCLAUDE READY ===");
        info!("Press {} to start recording", self.config.hotkey);
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
                if idle_duration > Duration::from_secs(self.config.idle_unload_seconds) {
                    if inference.is_loaded() {
                        info!("Unloading model after {} seconds idle", idle_duration.as_secs());
                        inference.unload();
                    }
                }
            }

            // Process events (non-blocking)
            match self.event_rx.try_recv() {
                Ok(event) => {
                    info!("=== APP EVENT RECEIVED ===");
                    info!("Event: {:?}", event);
                    info!("Time since last activity: {:?}", last_activity.elapsed());
                    last_activity = Instant::now();

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

                                            info!("Got {} samples, transcribing...", samples.len());
                                            self.state = AppState::Transcribing;
                                            _tray.set_state(AppState::Transcribing);

                                            // Transcribe
                                            match inference.transcribe(&samples) {
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

                                                    // Copy to clipboard
                                                    if let Err(e) = clipboard.set_text(&formatted) {
                                                        error!("Failed to copy to clipboard: {}", e);
                                                    } else {
                                                        info!("Copied to clipboard!");
                                                    }
                                                }
                                                Err(e) => {
                                                    error!("Transcription failed: {}", e);
                                                }
                                            }

                                            self.state = AppState::Idle;
                                            _tray.set_state(AppState::Idle);
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
                        AppEvent::Quit => {
                            info!("Quit requested");
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

        info!("Shutting down...");
        Ok(())
    }

    /// Pump Windows messages - MUST be called from main thread
    /// Returns the number of messages processed
    #[cfg(target_os = "windows")]
    fn pump_messages() -> u32 {
        use windows_sys::Win32::UI::WindowsAndMessaging::{
            DispatchMessageW, GetMessageW, PeekMessageW, TranslateMessage, MSG, PM_REMOVE, PM_NOREMOVE,
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

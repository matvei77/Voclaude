//! System tray icon with menu.

use crate::app::{AppEvent, AppState};

use crossbeam_channel::Sender;
use tray_icon::{
    menu::{Menu, MenuEvent, MenuItem, PredefinedMenuItem},
    TrayIcon, TrayIconBuilder,
    Icon,
};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tracing::{debug, error};

/// Embedded icons (will be replaced with actual assets)
const ICON_IDLE: &[u8] = include_bytes!("../assets/icon_idle.png");
const ICON_RECORDING: &[u8] = include_bytes!("../assets/icon_recording.png");
const ICON_PROCESSING: &[u8] = include_bytes!("../assets/icon_processing.png");

/// Menu item IDs
const MENU_TOGGLE: &str = "toggle";
const MENU_HISTORY: &str = "history";
const MENU_OPEN_TRANSCRIPTS: &str = "open_transcripts";
const MENU_OPEN_LAST_TRANSCRIPT: &str = "open_last_transcript";
const MENU_SETTINGS: &str = "settings";
const MENU_QUIT: &str = "quit";

/// Pre-decoded icon data cached as raw RGBA pixels to avoid re-decoding PNG on every state change.
struct CachedIcon {
    rgba: Vec<u8>,
    width: u32,
    height: u32,
}

impl CachedIcon {
    /// Decode a PNG byte slice into a `CachedIcon`.
    fn from_png(data: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let img = image::load_from_memory(data)?;
        let rgba = img.to_rgba8();
        let (width, height) = rgba.dimensions();
        Ok(Self {
            rgba: rgba.into_raw(),
            width,
            height,
        })
    }

    /// Reconstruct a `tray_icon::Icon` from the cached RGBA data.
    /// This performs only a memory copy — no PNG decoding.
    fn to_icon(&self) -> Result<Icon, Box<dyn std::error::Error>> {
        let icon = Icon::from_rgba(self.rgba.clone(), self.width, self.height)?;
        Ok(icon)
    }
}

pub struct TrayManager {
    tray: TrayIcon,
    toggle_item: MenuItem,
    state: Arc<Mutex<AppState>>,
    status_detail: Arc<Mutex<Option<String>>>,
    // Pre-decoded icon cache (BUG 30 fix)
    icon_idle: CachedIcon,
    icon_recording: CachedIcon,
    icon_processing: CachedIcon,
    // Shutdown flag for the menu event thread (BUG 31 fix)
    menu_shutdown: Arc<AtomicBool>,
}

impl TrayManager {
    pub fn new(event_tx: Sender<AppEvent>) -> Result<Self, Box<dyn std::error::Error>> {
        // Pre-decode all icon variants once at construction time (BUG 30 fix)
        let icon_idle = CachedIcon::from_png(ICON_IDLE)?;
        let icon_recording = CachedIcon::from_png(ICON_RECORDING)?;
        let icon_processing = CachedIcon::from_png(ICON_PROCESSING)?;

        // Build the initial tray icon from the pre-decoded idle icon
        let initial_icon = icon_idle.to_icon()?;

        // Create menu items
        let toggle_item = MenuItem::with_id(MENU_TOGGLE, "Start Recording", true, None);
        let history_item = MenuItem::with_id(MENU_HISTORY, "Show History", true, None);
        let open_transcripts_item =
            MenuItem::with_id(MENU_OPEN_TRANSCRIPTS, "Open Transcripts Folder", true, None);
        let open_last_transcript_item =
            MenuItem::with_id(MENU_OPEN_LAST_TRANSCRIPT, "Open Last Transcript", true, None);
        let settings_item = MenuItem::with_id(MENU_SETTINGS, "Settings...", true, None);
        let separator = PredefinedMenuItem::separator();
        let quit_item = MenuItem::with_id(MENU_QUIT, "Quit Voclaude", true, None);

        // Build menu
        let menu = Menu::new();
        menu.append(&toggle_item)?;
        menu.append(&history_item)?;
        menu.append(&open_last_transcript_item)?;
        menu.append(&open_transcripts_item)?;
        menu.append(&separator)?;
        menu.append(&settings_item)?;
        menu.append(&PredefinedMenuItem::separator())?;
        menu.append(&quit_item)?;

        // Build tray icon
        let tray = TrayIconBuilder::new()
            .with_tooltip("Voclaude - Ready")
            .with_icon(initial_icon)
            .with_menu(Box::new(menu))
            .build()?;

        let state = Arc::new(Mutex::new(AppState::Idle));
        let status_detail = Arc::new(Mutex::new(None));

        // Shutdown flag shared with the menu event thread (BUG 31 fix)
        let menu_shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = menu_shutdown.clone();

        // Spawn menu event handler
        let event_tx_clone = event_tx.clone();
        let _state_clone = state.clone();
        std::thread::spawn(move || {
            let receiver = MenuEvent::receiver();
            loop {
                // Check shutdown flag before blocking (BUG 31 fix)
                if shutdown_clone.load(Ordering::Relaxed) {
                    break;
                }
                match receiver.recv_timeout(Duration::from_millis(200)) {
                    Ok(event) => {
                        match event.id.0.as_str() {
                            MENU_TOGGLE => {
                                debug!("Toggle recording clicked");
                                let _ = event_tx_clone.send(AppEvent::HotkeyPressed);
                            }
                            MENU_SETTINGS => {
                                debug!("Settings clicked");
                                let _ = event_tx_clone.send(AppEvent::OpenSettings);
                            }
                            MENU_HISTORY => {
                                debug!("History clicked");
                                let _ = event_tx_clone.send(AppEvent::ShowHistoryWindow);
                            }
                            MENU_OPEN_TRANSCRIPTS => {
                                debug!("Open transcripts clicked");
                                let _ = event_tx_clone.send(AppEvent::OpenTranscriptsFolder);
                            }
                            MENU_OPEN_LAST_TRANSCRIPT => {
                                debug!("Open last transcript clicked");
                                let _ = event_tx_clone.send(AppEvent::OpenLastTranscript);
                            }
                            MENU_QUIT => {
                                debug!("Quit clicked");
                                let _ = event_tx_clone.send(AppEvent::Quit);
                            }
                            _ => {}
                        }
                    }
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                        // No event arrived within the timeout window; loop and re-check shutdown.
                        continue;
                    }
                    Err(_) => {
                        // Channel disconnected — exit the thread.
                        break;
                    }
                }
            }
        });

        Ok(Self {
            tray,
            toggle_item,
            state,
            status_detail,
            icon_idle,
            icon_recording,
            icon_processing,
            menu_shutdown,
        })
    }

    /// Update tray state (icon + menu text)
    pub fn set_state(&self, state: AppState) {
        // Update internal state
        if let Ok(mut s) = self.state.lock() {
            *s = state;
        }

        // O-3: Read and optionally clear status_detail in a single lock
        // acquisition to avoid TOCTOU between clear and re-read.
        let detail = if let Ok(mut detail) = self.status_detail.lock() {
            if state != AppState::Transcribing {
                *detail = None;
            }
            detail.clone()
        } else {
            None
        };
        let (cached_icon, tooltip, menu_text) = match state {
            AppState::Idle => (
                &self.icon_idle,
                "Voclaude - Ready".to_string(),
                "Start Recording",
            ),
            AppState::Recording => (
                &self.icon_recording,
                "Voclaude - Recording...".to_string(),
                "Stop Recording",
            ),
            AppState::Transcribing => (
                &self.icon_processing,
                detail
                    .map(|message| format!("Voclaude - {}", message))
                    .unwrap_or_else(|| "Voclaude - Processing...".to_string()),
                "Processing...",
            ),
        };

        // Reconstruct Icon from cached RGBA data — no PNG decode (BUG 30 fix)
        match cached_icon.to_icon() {
            Ok(icon) => {
                if let Err(e) = self.tray.set_icon(Some(icon)) {
                    error!("Failed to set tray icon: {}", e);
                }
            }
            Err(e) => {
                error!("Failed to build tray icon from cache: {}", e);
            }
        }

        // Update tooltip
        if let Err(e) = self.tray.set_tooltip(Some(&tooltip)) {
            error!("Failed to set tooltip: {}", e);
        }

        // Update menu item text
        self.toggle_item.set_text(menu_text);

        // Disable toggle during transcription
        self.toggle_item.set_enabled(state != AppState::Transcribing);
    }

    pub fn set_progress(&self, message: &str) {
        if let Ok(mut detail) = self.status_detail.lock() {
            *detail = Some(message.to_string());
        }

        let state = self.state.lock().ok().map(|state| *state).unwrap_or(AppState::Idle);
        if state == AppState::Transcribing {
            let tooltip = format!("Voclaude - {}", message);
            if let Err(e) = self.tray.set_tooltip(Some(&tooltip)) {
                error!("Failed to set tooltip: {}", e);
            }
        }
    }
}

impl Drop for TrayManager {
    fn drop(&mut self) {
        // Signal the menu event thread to exit cleanly (BUG 31 fix)
        self.menu_shutdown.store(true, Ordering::Relaxed);
    }
}

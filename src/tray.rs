//! System tray icon with menu.

use crate::app::{AppEvent, AppState};

use crossbeam_channel::Sender;
use tray_icon::{
    menu::{Menu, MenuEvent, MenuItem, PredefinedMenuItem},
    TrayIcon, TrayIconBuilder,
    Icon,
};
use std::sync::{Arc, Mutex};
use tracing::{debug, error};

/// Embedded icons (will be replaced with actual assets)
const ICON_IDLE: &[u8] = include_bytes!("../assets/icon_idle.png");
const ICON_RECORDING: &[u8] = include_bytes!("../assets/icon_recording.png");
const ICON_PROCESSING: &[u8] = include_bytes!("../assets/icon_processing.png");

/// Menu item IDs
const MENU_TOGGLE: &str = "toggle";
const MENU_HISTORY: &str = "history";
const MENU_SETTINGS: &str = "settings";
const MENU_QUIT: &str = "quit";

pub struct TrayManager {
    tray: TrayIcon,
    event_tx: Sender<AppEvent>,
    toggle_item: MenuItem,
    state: Arc<Mutex<AppState>>,
    status_detail: Arc<Mutex<Option<String>>>,
}

impl TrayManager {
    pub fn new(event_tx: Sender<AppEvent>) -> Result<Self, Box<dyn std::error::Error>> {
        // Load icons
        let icon_idle = Self::load_icon(ICON_IDLE)?;

        // Create menu items
        let toggle_item = MenuItem::with_id(MENU_TOGGLE, "Start Recording", true, None);
        let history_item = MenuItem::with_id(MENU_HISTORY, "Show History", true, None);
        let settings_item = MenuItem::with_id(MENU_SETTINGS, "Settings...", true, None);
        let separator = PredefinedMenuItem::separator();
        let quit_item = MenuItem::with_id(MENU_QUIT, "Quit Voclaude", true, None);

        // Build menu
        let menu = Menu::new();
        menu.append(&toggle_item)?;
        menu.append(&history_item)?;
        menu.append(&separator)?;
        menu.append(&settings_item)?;
        menu.append(&PredefinedMenuItem::separator())?;
        menu.append(&quit_item)?;

        // Build tray icon
        let tray = TrayIconBuilder::new()
            .with_tooltip("Voclaude - Ready")
            .with_icon(icon_idle)
            .with_menu(Box::new(menu))
            .build()?;

        let state = Arc::new(Mutex::new(AppState::Idle));
        let status_detail = Arc::new(Mutex::new(None));

        // Spawn menu event handler
        let event_tx_clone = event_tx.clone();
        let _state_clone = state.clone();
        std::thread::spawn(move || {
            let receiver = MenuEvent::receiver();
            loop {
                if let Ok(event) = receiver.recv() {
                    match event.id.0.as_str() {
                        MENU_TOGGLE => {
                            debug!("Toggle recording clicked");
                            let _ = event_tx_clone.send(AppEvent::HotkeyPressed);
                        }
                        MENU_SETTINGS => {
                            debug!("Settings clicked");
                            // TODO: Open settings window
                        }
                        MENU_HISTORY => {
                            debug!("History clicked");
                            let _ = event_tx_clone.send(AppEvent::ShowHistoryWindow);
                        }
                        MENU_QUIT => {
                            debug!("Quit clicked");
                            let _ = event_tx_clone.send(AppEvent::Quit);
                        }
                        _ => {}
                    }
                }
            }
        });

        Ok(Self {
            tray,
            event_tx,
            toggle_item,
            state,
            status_detail,
        })
    }

    fn load_icon(data: &[u8]) -> Result<Icon, Box<dyn std::error::Error>> {
        let img = image::load_from_memory(data)?;
        let rgba = img.to_rgba8();
        let (width, height) = rgba.dimensions();
        let icon = Icon::from_rgba(rgba.into_raw(), width, height)?;
        Ok(icon)
    }

    /// Update tray state (icon + menu text)
    pub fn set_state(&self, state: AppState) {
        // Update internal state
        if let Ok(mut s) = self.state.lock() {
            *s = state;
        }

        if state != AppState::Transcribing {
            if let Ok(mut detail) = self.status_detail.lock() {
                *detail = None;
            }
        }

        // Update icon and tooltip
        let detail = self
            .status_detail
            .lock()
            .ok()
            .and_then(|detail| detail.clone());
        let (icon_data, tooltip, menu_text) = match state {
            AppState::Idle => (
                ICON_IDLE,
                "Voclaude - Ready".to_string(),
                "Start Recording",
            ),
            AppState::Recording => (
                ICON_RECORDING,
                "Voclaude - Recording...".to_string(),
                "Stop Recording",
            ),
            AppState::Transcribing => (
                ICON_PROCESSING,
                detail
                    .map(|message| format!("Voclaude - {}", message))
                    .unwrap_or_else(|| "Voclaude - Processing...".to_string()),
                "Processing...",
            ),
        };

        // Update icon
        if let Ok(icon) = Self::load_icon(icon_data) {
            if let Err(e) = self.tray.set_icon(Some(icon)) {
                error!("Failed to set tray icon: {}", e);
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

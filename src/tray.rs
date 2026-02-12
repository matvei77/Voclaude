//! System tray icon with menu.

use crate::app::{AppEvent, AppState};

use crossbeam_channel::Sender;
use tray_icon::{
    menu::{Menu, MenuEvent, MenuItem, PredefinedMenuItem},
    TrayIcon, TrayIconBuilder,
    Icon,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tracing::{debug, error, info};

/// Embedded icons (will be replaced with actual assets)
const ICON_IDLE: &[u8] = include_bytes!("../assets/icon_idle.png");
const ICON_RECORDING: &[u8] = include_bytes!("../assets/icon_recording.png");
const ICON_PROCESSING: &[u8] = include_bytes!("../assets/icon_processing.png");

/// Menu item IDs
const MENU_TOGGLE: &str = "toggle";
const MENU_SETTINGS: &str = "settings";
const MENU_QUIT: &str = "quit";

pub struct TrayManager {
    tray: TrayIcon,
    #[allow(dead_code)]
    event_tx: Sender<AppEvent>,
    toggle_item: MenuItem,
    state: Arc<Mutex<AppState>>,
    shutdown: Arc<AtomicBool>,
}

impl TrayManager {
    pub fn new(event_tx: Sender<AppEvent>) -> Result<Self, Box<dyn std::error::Error>> {
        // Load icons
        let icon_idle = Self::load_icon(ICON_IDLE)?;

        // Create menu items
        let toggle_item = MenuItem::with_id(MENU_TOGGLE, "Start Recording", true, None);
        let settings_item = MenuItem::with_id(MENU_SETTINGS, "Settings...", true, None);
        let separator = PredefinedMenuItem::separator();
        let quit_item = MenuItem::with_id(MENU_QUIT, "Quit Voclaude", true, None);

        // Build menu
        let menu = Menu::new();
        menu.append(&toggle_item)?;
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
        let shutdown = Arc::new(AtomicBool::new(false));

        // Spawn menu event handler with shutdown signal
        let event_tx_clone = event_tx.clone();
        let shutdown_clone = shutdown.clone();
        std::thread::spawn(move || {
            info!("Tray menu event handler started");
            let receiver = MenuEvent::receiver();

            loop {
                // Check shutdown flag
                if shutdown_clone.load(Ordering::Relaxed) {
                    info!("Tray menu handler received shutdown signal");
                    break;
                }

                // Use recv_timeout to allow checking shutdown flag periodically
                match receiver.recv_timeout(Duration::from_millis(100)) {
                    Ok(event) => {
                        match event.id.0.as_str() {
                            MENU_TOGGLE => {
                                debug!("Toggle recording clicked");
                                if event_tx_clone.send(AppEvent::HotkeyPressed).is_err() {
                                    info!("Event channel closed, shutting down tray handler");
                                    break;
                                }
                            }
                            MENU_SETTINGS => {
                                debug!("Settings clicked");
                                // TODO: Open settings window
                            }
                            MENU_QUIT => {
                                debug!("Quit clicked");
                                if event_tx_clone.send(AppEvent::Quit).is_err() {
                                    break;
                                }
                            }
                            _ => {}
                        }
                    }
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                        // Normal timeout, check shutdown flag and continue
                    }
                    Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                        info!("Tray menu receiver disconnected, exiting thread");
                        break;
                    }
                }
            }
            info!("Tray menu event handler exited cleanly");
        });

        Ok(Self {
            tray,
            event_tx,
            toggle_item,
            state,
            shutdown,
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

        // Update icon and tooltip
        let (icon_data, tooltip, menu_text) = match state {
            AppState::Idle => (ICON_IDLE, "Voclaude - Ready", "Start Recording"),
            AppState::Recording => (ICON_RECORDING, "Voclaude - Recording...", "Stop Recording"),
            AppState::Transcribing => (ICON_PROCESSING, "Voclaude - Processing...", "Processing..."),
        };

        // Update icon
        if let Ok(icon) = Self::load_icon(icon_data) {
            if let Err(e) = self.tray.set_icon(Some(icon)) {
                error!("Failed to set tray icon: {}", e);
            }
        }

        // Update tooltip
        if let Err(e) = self.tray.set_tooltip(Some(tooltip)) {
            error!("Failed to set tooltip: {}", e);
        }

        // Update menu item text
        self.toggle_item.set_text(menu_text);

        // Disable toggle during transcription
        self.toggle_item.set_enabled(state != AppState::Transcribing);
    }
}

impl Drop for TrayManager {
    fn drop(&mut self) {
        // Signal shutdown to menu handler thread
        self.shutdown.store(true, Ordering::Relaxed);
        info!("TrayManager dropped, shutdown signaled");
    }
}

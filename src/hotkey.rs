//! Global hotkey registration and handling.

use crate::app::AppEvent;

use crossbeam_channel::Sender;
use global_hotkey::{
    hotkey::{Code, HotKey, Modifiers},
    GlobalHotKeyEvent, GlobalHotKeyManager,
};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, trace};

pub struct HotkeyManager {
    manager: GlobalHotKeyManager,
    hotkeys: Vec<HotKey>,
}

impl HotkeyManager {
    /// Register multiple hotkeys with a single listener thread.
    /// Takes pairs of (hotkey_string, AppEvent) and dispatches events
    /// based on the hotkey ID — avoiding the global receiver contention
    /// issue that occurs when multiple threads compete for events.
    pub fn new_multi(
        bindings: &[(&str, AppEvent)],
        event_tx: Sender<AppEvent>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let manager = GlobalHotKeyManager::new()?;
        let mut hotkeys = Vec::new();
        let mut id_to_event: HashMap<u32, AppEvent> = HashMap::new();

        for (hotkey_str, app_event) in bindings {
            let hotkey = Self::parse_hotkey(hotkey_str)?;
            info!("Registering hotkey '{}' (ID: {})", hotkey_str, hotkey.id());
            manager.register(hotkey)?;
            id_to_event.insert(hotkey.id(), app_event.clone());
            hotkeys.push(hotkey);
        }

        // Single unified listener thread for ALL hotkeys
        std::thread::spawn(move || {
            info!("Unified hotkey listener started ({} bindings)", id_to_event.len());
            let receiver = GlobalHotKeyEvent::receiver();
            let min_event_gap = Duration::from_millis(150);
            let mut last_trigger_per_id: HashMap<u32, Instant> = HashMap::new();

            loop {
                match receiver.recv_timeout(Duration::from_millis(100)) {
                    Ok(event) => {
                        if let Some(app_event) = id_to_event.get(&event.id) {
                            let now = Instant::now();
                            let last = last_trigger_per_id
                                .get(&event.id)
                                .copied()
                                .unwrap_or(Instant::now() - Duration::from_secs(1));

                            if now.duration_since(last) < min_event_gap {
                                debug!("Debounced hotkey {} ({:?})", event.id, event.state);
                                continue;
                            }

                            last_trigger_per_id.insert(event.id, now);
                            info!("Hotkey {} fired ({:?})", event.id, event.state);
                            if let Err(e) = event_tx.send(app_event.clone()) {
                                error!("Failed to send hotkey event: {}", e);
                            }
                        } else {
                            debug!("Unknown hotkey ID: {}", event.id);
                        }
                    }
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                        trace!("Hotkey receiver timeout (normal)");
                    }
                    Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                        error!("Hotkey receiver disconnected!");
                        break;
                    }
                }
            }
        });

        Ok(Self { manager, hotkeys })
    }

    fn parse_hotkey(s: &str) -> Result<HotKey, Box<dyn std::error::Error>> {
        let parts: Vec<&str> = s.split('+').map(|p| p.trim()).collect();

        let mut modifiers = Modifiers::empty();
        let mut key_code = None;

        for part in parts {
            match part.to_lowercase().as_str() {
                // Modifiers
                "ctrl" | "control" => modifiers |= Modifiers::CONTROL,
                "alt" => modifiers |= Modifiers::ALT,
                "shift" => modifiers |= Modifiers::SHIFT,
                "super" | "win" | "cmd" | "meta" => modifiers |= Modifiers::SUPER,

                // Letter keys
                "a" => key_code = Some(Code::KeyA),
                "b" => key_code = Some(Code::KeyB),
                "c" => key_code = Some(Code::KeyC),
                "d" => key_code = Some(Code::KeyD),
                "e" => key_code = Some(Code::KeyE),
                "f" => key_code = Some(Code::KeyF),
                "g" => key_code = Some(Code::KeyG),
                "h" => key_code = Some(Code::KeyH),
                "i" => key_code = Some(Code::KeyI),
                "j" => key_code = Some(Code::KeyJ),
                "k" => key_code = Some(Code::KeyK),
                "l" => key_code = Some(Code::KeyL),
                "m" => key_code = Some(Code::KeyM),
                "n" => key_code = Some(Code::KeyN),
                "o" => key_code = Some(Code::KeyO),
                "p" => key_code = Some(Code::KeyP),
                "q" => key_code = Some(Code::KeyQ),
                "r" => key_code = Some(Code::KeyR),
                "s" => key_code = Some(Code::KeyS),
                "t" => key_code = Some(Code::KeyT),
                "u" => key_code = Some(Code::KeyU),
                "v" => key_code = Some(Code::KeyV),
                "w" => key_code = Some(Code::KeyW),
                "x" => key_code = Some(Code::KeyX),
                "y" => key_code = Some(Code::KeyY),
                "z" => key_code = Some(Code::KeyZ),

                // Number keys
                "0" => key_code = Some(Code::Digit0),
                "1" => key_code = Some(Code::Digit1),
                "2" => key_code = Some(Code::Digit2),
                "3" => key_code = Some(Code::Digit3),
                "4" => key_code = Some(Code::Digit4),
                "5" => key_code = Some(Code::Digit5),
                "6" => key_code = Some(Code::Digit6),
                "7" => key_code = Some(Code::Digit7),
                "8" => key_code = Some(Code::Digit8),
                "9" => key_code = Some(Code::Digit9),

                // Function keys
                "f1" => key_code = Some(Code::F1),
                "f2" => key_code = Some(Code::F2),
                "f3" => key_code = Some(Code::F3),
                "f4" => key_code = Some(Code::F4),
                "f5" => key_code = Some(Code::F5),
                "f6" => key_code = Some(Code::F6),
                "f7" => key_code = Some(Code::F7),
                "f8" => key_code = Some(Code::F8),
                "f9" => key_code = Some(Code::F9),
                "f10" => key_code = Some(Code::F10),
                "f11" => key_code = Some(Code::F11),
                "f12" => key_code = Some(Code::F12),

                // Special keys
                "space" => key_code = Some(Code::Space),
                "enter" | "return" => key_code = Some(Code::Enter),
                "tab" => key_code = Some(Code::Tab),
                "backspace" => key_code = Some(Code::Backspace),
                "delete" => key_code = Some(Code::Delete),
                "escape" | "esc" => key_code = Some(Code::Escape),
                "home" => key_code = Some(Code::Home),
                "end" => key_code = Some(Code::End),
                "pageup" => key_code = Some(Code::PageUp),
                "pagedown" => key_code = Some(Code::PageDown),

                // Numpad keys
                "numpad0" | "num0" => key_code = Some(Code::Numpad0),
                "numpad1" | "num1" => key_code = Some(Code::Numpad1),
                "numpad2" | "num2" => key_code = Some(Code::Numpad2),
                "numpad3" | "num3" => key_code = Some(Code::Numpad3),
                "numpad4" | "num4" => key_code = Some(Code::Numpad4),
                "numpad5" | "num5" => key_code = Some(Code::Numpad5),
                "numpad6" | "num6" => key_code = Some(Code::Numpad6),
                "numpad7" | "num7" => key_code = Some(Code::Numpad7),
                "numpad8" | "num8" => key_code = Some(Code::Numpad8),
                "numpad9" | "num9" => key_code = Some(Code::Numpad9),
                "numpadadd" | "numadd" => key_code = Some(Code::NumpadAdd),
                "numpadsubtract" | "numsub" => key_code = Some(Code::NumpadSubtract),
                "numpadmultiply" | "nummul" => key_code = Some(Code::NumpadMultiply),
                "numpaddivide" | "numdiv" => key_code = Some(Code::NumpadDivide),
                "numpaddecimal" | "numdec" => key_code = Some(Code::NumpadDecimal),
                "numpadenter" | "numenter" => key_code = Some(Code::NumpadEnter),

                // Punctuation
                ";" | "semicolon" => key_code = Some(Code::Semicolon),
                "'" | "quote" => key_code = Some(Code::Quote),
                "," | "comma" => key_code = Some(Code::Comma),
                "." | "period" => key_code = Some(Code::Period),
                "/" | "slash" => key_code = Some(Code::Slash),
                "`" | "backquote" => key_code = Some(Code::Backquote),
                "[" | "bracketleft" => key_code = Some(Code::BracketLeft),
                "]" | "bracketright" => key_code = Some(Code::BracketRight),
                "\\" | "backslash" => key_code = Some(Code::Backslash),
                "-" | "minus" => key_code = Some(Code::Minus),
                "=" | "equal" => key_code = Some(Code::Equal),

                other => {
                    return Err(format!("Unknown key: {}", other).into());
                }
            }
        }

        let code = key_code.ok_or("No key specified in hotkey")?;
        Ok(HotKey::new(Some(modifiers), code))
    }
}

impl Drop for HotkeyManager {
    fn drop(&mut self) {
        for hotkey in &self.hotkeys {
            if let Err(e) = self.manager.unregister(*hotkey) {
                error!("Failed to unregister hotkey: {}", e);
            }
        }
    }
}

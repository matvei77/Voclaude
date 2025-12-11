//! Global hotkey registration and handling.

use crate::app::AppEvent;

use crossbeam_channel::Sender;
use global_hotkey::{
    hotkey::{Code, HotKey, Modifiers},
    GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState,
};
use std::time::Duration;
use tracing::{debug, error, info, warn, trace};

pub struct HotkeyManager {
    manager: GlobalHotKeyManager,
    hotkey: HotKey,
}

impl HotkeyManager {
    pub fn new(hotkey_str: &str, event_tx: Sender<AppEvent>) -> Result<Self, Box<dyn std::error::Error>> {
        info!("=== HOTKEY INITIALIZATION START ===");

        // Log Windows thread ID
        #[cfg(target_os = "windows")]
        {
            #[link(name = "kernel32")]
            extern "system" {
                fn GetCurrentThreadId() -> u32;
            }
            let win_thread_id = unsafe { GetCurrentThreadId() };
            info!("Hotkey init on Windows Thread ID: {}", win_thread_id);
        }

        info!("Creating GlobalHotKeyManager...");

        let manager = GlobalHotKeyManager::new()?;
        info!("GlobalHotKeyManager created successfully");

        // Parse hotkey string (e.g., "Super+C", "Ctrl+Shift+Space")
        info!("Parsing hotkey string: '{}'", hotkey_str);
        let hotkey = Self::parse_hotkey(hotkey_str)?;
        info!("Parsed hotkey - ID: {}", hotkey.id());

        // Register the hotkey
        info!("Registering hotkey with Windows...");
        match manager.register(hotkey) {
            Ok(_) => info!("Hotkey registered successfully with Windows!"),
            Err(e) => {
                error!("FAILED to register hotkey: {:?}", e);
                return Err(e.into());
            }
        }

        // Spawn event handler thread
        let hotkey_id = hotkey.id();
        info!("Hotkey ID: {}, spawning listener thread...", hotkey_id);

        std::thread::spawn(move || {
            info!("=== HOTKEY LISTENER THREAD STARTED ===");
            info!("Thread ID: {:?}", std::thread::current().id());
            info!("Getting GlobalHotKeyEvent receiver...");

            let receiver = GlobalHotKeyEvent::receiver();
            info!("Got receiver, entering event loop...");

            let mut loop_count: u64 = 0;
            let mut last_log = std::time::Instant::now();

            loop {
                loop_count += 1;

                // Log every 10 seconds to show the thread is alive
                if last_log.elapsed() > Duration::from_secs(10) {
                    info!("Hotkey listener alive - {} iterations, waiting for events...", loop_count);
                    last_log = std::time::Instant::now();
                }

                // Use recv_timeout instead of blocking recv for better diagnostics
                match receiver.recv_timeout(Duration::from_millis(100)) {
                    Ok(event) => {
                        info!("=== HOTKEY EVENT RECEIVED ===");
                        info!("Event ID: {}, Expected ID: {}", event.id, hotkey_id);
                        info!("Event state: {:?}", event.state);
                        info!("IDs match: {}", event.id == hotkey_id);

                        // Only trigger on key press, not release
                        if event.state == HotKeyState::Pressed {
                            info!("Event is PRESSED state");
                            if event.id == hotkey_id {
                                info!("IDs MATCH! Sending AppEvent::HotkeyPressed...");
                                match event_tx.send(AppEvent::HotkeyPressed) {
                                    Ok(_) => info!("AppEvent::HotkeyPressed sent successfully!"),
                                    Err(e) => error!("FAILED to send hotkey event: {}", e),
                                }
                            } else {
                                warn!("Event ID {} does not match expected {}", event.id, hotkey_id);
                            }
                        } else {
                            debug!("Ignoring non-pressed state: {:?}", event.state);
                        }
                    }
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                        // Normal timeout, continue loop
                        trace!("Receiver timeout (normal)");
                    }
                    Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                        error!("Hotkey receiver DISCONNECTED! Exiting thread.");
                        break;
                    }
                }
            }
            error!("Hotkey listener thread exiting!");
        });

        info!("=== HOTKEY INITIALIZATION COMPLETE ===");
        Ok(Self { manager, hotkey })
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
        if let Err(e) = self.manager.unregister(self.hotkey) {
            error!("Failed to unregister hotkey: {}", e);
        }
    }
}

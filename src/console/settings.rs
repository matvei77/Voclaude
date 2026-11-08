//! Settings that apply to every engine: hotkeys, text, live transcription,
//! storage. Engine settings are on the Model tab; the API on the API tab.
//! Everything else stays in config.toml ("Open config.toml").

use super::model::toggle;
use super::{widgets, Env};
use crate::config_edit::{self, Edit};
use eframe::egui::{self, RichText};

#[derive(Default)]
pub struct State {
    record_hotkey: Option<String>,
    window_hotkey: Option<String>,
    show_advanced: bool,
}

fn save(env: &mut Env, edits: &[Edit], what: &str) {
    match config_edit::save(edits) {
        Ok(()) => env.toast(format!("{} saved", what)),
        Err(e) => env.toast_error(format!("Not saved: {}", e)),
    }
}

/// A number field that saves when the drag or the typing ends.
fn number(ui: &mut egui::Ui, value: u64, range: std::ops::RangeInclusive<u64>, suffix: &str) -> Option<u64> {
    let mut v = value;
    let r = ui.add(egui::DragValue::new(&mut v).clamp_range(range).suffix(suffix).speed(1.0));
    ((r.drag_stopped() || r.lost_focus() || (r.changed() && !r.dragged() && !r.has_focus())) && v != value).then_some(v)
}

fn seconds(ui: &mut egui::Ui, value: f32, range: std::ops::RangeInclusive<f32>) -> Option<f32> {
    let mut v = value;
    let r = ui.add(egui::DragValue::new(&mut v).clamp_range(range).suffix(" s").speed(0.1).max_decimals(1));
    ((r.drag_stopped() || r.lost_focus() || (r.changed() && !r.dragged() && !r.has_focus())) && (v - value).abs() > 1e-3).then_some(v)
}

impl State {
    pub fn show(&mut self, ui: &mut egui::Ui, env: &mut Env) {
        widgets::page_header(ui, "Settings", "Saved to config.toml as you change them; Voclaude applies them within a few seconds.");
        if let Some(err) = env.config_error {
            widgets::banner(ui, &format!("config.toml has an error, so Voclaude uses its previous settings: {}", err), true);
        }
        let c = env.config.clone();

        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            if crate::autostart::supported() {
                widgets::section(ui, "Startup");
                widgets::card(ui, |ui| {
                    let current = crate::autostart::current();
                    let here = current.as_deref().is_some_and(crate::autostart::points_here);
                    let help = match &current {
                        None => "Off. Start Voclaude from the Start menu when you need it.".to_string(),
                        Some(_) if here => "On: this copy starts when you sign in to Windows.".to_string(),
                        Some(cmd) => format!("On, but it starts another copy: {}", cmd),
                    };
                    widgets::setting_row(ui, "Start with Windows", &help, |ui| {
                        let mut on = current.is_some();
                        if toggle(ui, &mut on).clicked() {
                            match crate::autostart::set(on) {
                                Ok(()) => env.toast(if on { "Voclaude starts when you sign in" } else { "Voclaude no longer starts with Windows" }),
                                Err(e) => env.toast_error(format!("Not changed: {}", e)),
                            }
                        }
                        if current.is_some() && !here && ui.button("Start this copy instead").clicked() {
                            match crate::autostart::set(true) {
                                Ok(()) => env.toast("This copy now starts when you sign in"),
                                Err(e) => env.toast_error(format!("Not changed: {}", e)),
                            }
                        }
                    });
                });
            }

            widgets::section(ui, "Hotkeys");
            widgets::card(ui, |ui| {
                for (label, help, key, slot, current) in [
                    ("Record", "Start and stop dictation, from any program.", "hotkey", &mut self.record_hotkey, c.hotkey.clone()),
                    ("Open this window", "Shows this window, or brings it to the front.", "history_hotkey", &mut self.window_hotkey, c.history_hotkey.clone()),
                ] {
                    let text = slot.get_or_insert_with(|| current.clone());
                    widgets::setting_row(ui, label, help, |ui| {
                        let r = ui.add(egui::TextEdit::singleline(text).desired_width(140.0).hint_text("e.g. Ctrl+Alt+V"));
                        if r.lost_focus() && text.trim() != current {
                            let v = text.trim().to_string();
                            match crate::hotkey::HotkeyManager::parse_hotkey(&v) {
                                Ok(_) => save(env, &[Edit::Set(key, v.into())], label),
                                Err(e) => {
                                    env.toast_error(format!("{} is not a hotkey: {}", v, e));
                                    *text = current.clone();
                                }
                            }
                        }
                    });
                }
                widgets::hint(ui, "Keys: F1–F12, A–Z, 0–9, Space, Enter, Tab, Escape, with Ctrl, Alt, Shift, Win. Example: Ctrl+Shift+Space. Changes apply at once.");
            });

            widgets::section(ui, "Text");
            widgets::card(ui, |ui| {
                widgets::setting_row(ui, "Capitalize the first letter", "", |ui| {
                    let mut on = c.capitalize_first;
                    if toggle(ui, &mut on).clicked() {
                        save(env, &[Edit::Set("capitalize_first", on.into())], "Capitalization");
                    }
                });
                widgets::setting_row(ui, "Add a space at the end", "So several dictations paste one after another.", |ui| {
                    let mut on = c.add_trailing_space;
                    if toggle(ui, &mut on).clicked() {
                        save(env, &[Edit::Set("add_trailing_space", on.into())], "Trailing space");
                    }
                });
            });

            widgets::section(ui, "Live transcription");
            widgets::card(ui, |ui| {
                widgets::setting_row(ui, "Transcribe while recording", "Pauses cut the recording into segments that are transcribed while you speak, so only the last seconds remain after you stop.", |ui| {
                    let mut on = c.streaming;
                    if toggle(ui, &mut on).clicked() {
                        save(env, &[Edit::Set("streaming", on.into())], "Live transcription");
                    }
                });
                if ui.link(if self.show_advanced { "Hide segment settings" } else { "Segment settings" }).clicked() {
                    self.show_advanced = !self.show_advanced;
                }
                if self.show_advanced {
                    ui.add_space(4.0);
                    let auto = if c.uses_parakeet() { 6.0 } else { 20.0 };
                    widgets::setting_row(
                        ui,
                        "Shortest segment",
                        &format!("A pause closes a segment only after this long. Automatic: {} s for this engine.", auto),
                        |ui| {
                            if c.segment_min_seconds.is_some() && ui.small_button("Automatic").clicked() {
                                save(env, &[Edit::Remove("segment_min_seconds")], "Shortest segment");
                            }
                            if let Some(v) = seconds(ui, c.segment_min_secs(), 1.0..=120.0) {
                                save(env, &[Edit::Set("segment_min_seconds", (v as f64).into())], "Shortest segment");
                            }
                        },
                    );
                    widgets::setting_row(ui, "Longest segment", "Cut at the quietest recent point after this long.", |ui| {
                        if let Some(v) = seconds(ui, c.segment_max_seconds, 5.0..=300.0) {
                            save(env, &[Edit::Set("segment_max_seconds", (v as f64).into())], "Longest segment");
                        }
                    });
                    widgets::setting_row(ui, "Pause length", "Silence that closes a segment once it is long enough.", |ui| {
                        if let Some(v) = seconds(ui, c.segment_pause_seconds, 0.1..=5.0) {
                            save(env, &[Edit::Set("segment_pause_seconds", (v as f64).into())], "Pause length");
                        }
                    });
                }
            });

            widgets::section(ui, "Storage and memory");
            widgets::card(ui, |ui| {
                widgets::setting_row(ui, "Dictations kept", "The newest dictations in history. Older ones are deleted with their recordings.", |ui| {
                    if let Some(v) = number(ui, c.history_max_entries as u64, 1..=100_000, "") {
                        save(env, &[Edit::Set("history_max_entries", (v as i64).into())], "Dictations kept");
                    }
                });
                widgets::setting_row(ui, "Recordings kept", "Audio of the newest dictations, for Names → Check. 0 deletes each recording after it is transcribed.", |ui| {
                    if let Some(v) = number(ui, c.audio_retention_count as u64, 0..=100_000, "") {
                        save(env, &[Edit::Set("audio_retention_count", (v as i64).into())], "Recordings kept");
                    }
                });
                widgets::setting_row(ui, "Unload the model after", "Idle time before the model leaves memory. The next dictation loads it again (a few seconds).", |ui| {
                    if let Some(v) = number(ui, c.idle_unload_seconds, 5..=86_400, " s") {
                        save(env, &[Edit::Set("idle_unload_seconds", (v as i64).into())], "Unload time");
                    }
                });
                if let Some(dirs) = crate::config::project_dirs() {
                    ui.horizontal(|ui| {
                        if ui.button("Open data folder").clicked() {
                            widgets::open_path(dirs.data_dir());
                        }
                        if ui.button("Open transcripts folder").on_hover_text("Dictations over 10 minutes or 20,000 characters are saved here as text files").clicked() {
                            let dir = dirs.data_dir().join("transcripts");
                            let _ = std::fs::create_dir_all(&dir);
                            widgets::open_path(&dir);
                        }
                    });
                }
            });

            widgets::section(ui, "config.toml");
            widgets::card(ui, |ui| {
                widgets::hint(ui, "Every setting, with comments. Saved changes apply within a few seconds. The hotkeys apply at once too.");
                ui.add_space(2.0);
                ui.horizontal(|ui| {
                    if let Some(path) = env.config_path.clone() {
                        if ui.button("Open config.toml").clicked() {
                            widgets::open_text(&path);
                        }
                        ui.label(RichText::new(path.display().to_string()).monospace().size(11.5).color(super::theme::FAINT));
                    }
                });
            });
            ui.add_space(8.0);
        });
    }
}

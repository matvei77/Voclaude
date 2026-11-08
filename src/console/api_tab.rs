//! API: the local OpenAI-compatible endpoint, its settings, and ready-to-copy
//! snippets (base URL, curl, command line, and a prompt for another agent).

use super::model::toggle;
use super::{theme, widgets, Env};
use crate::config_edit::{self, Edit};
use eframe::egui::{self, RichText};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

type Health = Arc<Mutex<Option<(Instant, Result<String, String>)>>>;

#[derive(Default)]
pub struct State {
    health: Health,
    checking: Arc<std::sync::atomic::AtomicBool>,
    show_token: bool,
}

fn save(env: &mut Env, edits: &[Edit], what: &str) {
    match config_edit::save(edits) {
        Ok(()) => env.toast(format!("{} saved. The API restarts in a few seconds.", what)),
        Err(e) => env.toast_error(format!("Not saved: {}", e)),
    }
}

/// 32 hex characters from the OS-seeded hasher keys.
fn random_token() -> String {
    use std::hash::{BuildHasher, Hasher};
    let mut out = String::new();
    for i in 0..2u64 {
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u64(i ^ std::process::id() as u64);
        h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0));
        out.push_str(&format!("{:016x}", h.finish()));
    }
    out
}

/// The prompt a user pastes into another agent or app.
pub fn agent_prompt(port: u16, token: Option<&str>) -> String {
    let base = format!("http://127.0.0.1:{}", port);
    let auth = match token {
        Some(t) => format!("\nAuth: send the header \"Authorization: Bearer {}\" with every request.", t),
        None => String::new(),
    };
    let curl_auth = token.map(|t| format!(" -H \"Authorization: Bearer {}\"", t)).unwrap_or_default();
    format!(
        "Local speech-to-text runs on this computer (Voclaude, OpenAI Whisper-compatible). Use it to transcribe audio or video files.\n\
Endpoint: POST {base}/v1/audio/transcriptions{auth}\n\
- Upload a file:  curl{curl_auth} -F file=@<audio file> {base}/v1/audio/transcriptions\n\
- Local file, no upload:  POST JSON {{\"path\": \"<absolute path>\", \"response_format\": \"text\"}}\n\
- OpenAI SDK:  base_url=\"{base}/v1\", api_key=\"unused\"; the \"model\" and \"language\" fields are ignored.\n\
response_format: json (default, {{\"text\", \"duration\"}}), text, or verbose_json (adds segments with start/end seconds).\n\
Formats: wav, mp3, flac, ogg, m4a/mp4/mov, mkv. Long files are fine; they are cut at pauses.\n\
Check that it runs: GET {base}/health"
    )
}

impl State {
    fn check_health(&self, port: u16, token: Option<String>) {
        use std::sync::atomic::Ordering;
        if self.checking.swap(true, Ordering::SeqCst) {
            return;
        }
        let (health, checking) = (self.health.clone(), self.checking.clone());
        std::thread::spawn(move || {
            let mut req = ureq::get(&format!("http://127.0.0.1:{}/health", port)).timeout(Duration::from_secs(2));
            if let Some(t) = token {
                req = req.set("Authorization", &format!("Bearer {}", t));
            }
            let result = match req.call() {
                Ok(resp) => match resp.into_json::<serde_json::Value>() {
                    Ok(v) => Ok(format!(
                        "ok · model {} · {} · v{}",
                        v.get("model").and_then(|m| m.as_str()).unwrap_or("?"),
                        if v.get("use_gpu").and_then(|g| g.as_bool()).unwrap_or(false) { "GPU" } else { "CPU" },
                        v.get("version").and_then(|m| m.as_str()).unwrap_or("?")
                    )),
                    Err(e) => Err(e.to_string()),
                },
                Err(e) => Err(e.to_string()),
            };
            *health.lock().unwrap_or_else(|e| e.into_inner()) = Some((Instant::now(), result));
            checking.store(false, Ordering::SeqCst);
        });
    }

    pub fn show(&mut self, ui: &mut egui::Ui, env: &mut Env) {
        widgets::page_header(
            ui,
            "API",
            "Other programs and agents on this computer can transcribe files with the loaded model. OpenAI Whisper-compatible; only this computer can connect.",
        );
        let c = env.config.clone();
        let port = c.api_port;
        let token = c.api_token.clone().filter(|t| !t.trim().is_empty());

        // Health check every 5 s while this tab is open.
        let last = self.health.lock().unwrap_or_else(|e| e.into_inner()).clone();
        if c.api_enabled && last.as_ref().map_or(true, |(t, _)| t.elapsed() > Duration::from_secs(5)) {
            self.check_health(port, token.clone());
        }
        ui.ctx().request_repaint_after(Duration::from_secs(1));

        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            widgets::card(ui, |ui| {
                ui.horizontal(|ui| {
                    let (text, color) = match (c.api_enabled, env.status.and_then(|s| s.api_port), env.status.and_then(|s| s.api_error.clone())) {
                        (false, _, _) => ("Off".to_string(), theme::FAINT),
                        (true, _, Some(err)) => (format!("Not listening: {}", err), theme::DANGER),
                        (true, Some(p), _) => (format!("Listening on http://127.0.0.1:{}", p), theme::SUCCESS),
                        (true, None, None) if env.connected() => ("Starting…".to_string(), theme::WARNING),
                        (true, None, None) => ("Voclaude is not running".to_string(), theme::FAINT),
                    };
                    widgets::state_pill(ui, &text, color);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let mut on = c.api_enabled;
                        if toggle(ui, &mut on).clicked() {
                            save(env, &[Edit::Set("api_enabled", on.into())], if on { "API on" } else { "API off" });
                        }
                        ui.label(RichText::new("Enabled").color(theme::MUTED));
                    });
                });
                if c.api_enabled {
                    let line = match &last {
                        Some((t, Ok(s))) => (format!("Health check: {}  ({} s ago)", s, t.elapsed().as_secs()), theme::MUTED),
                        Some((_, Err(e))) => (format!("Health check failed: {}", e), theme::DANGER),
                        None => ("Health check…".to_string(), theme::FAINT),
                    };
                    ui.label(RichText::new(line.0).size(12.0).color(line.1));
                }
                ui.add_space(4.0);
                widgets::setting_row(ui, "Port", "127.0.0.1 only. Default 7770.", |ui| {
                    let mut v = port;
                    let r = ui.add(egui::DragValue::new(&mut v).clamp_range(1024..=65535));
                    if (r.drag_stopped() || r.lost_focus()) && v != port {
                        save(env, &[Edit::Set("api_port", (v as i64).into())], "Port");
                    }
                });
                widgets::setting_row(ui, "Token", "Optional. When set, requests need \"Authorization: Bearer <token>\".", |ui| {
                    if token.is_some() {
                        if ui.small_button("Remove").clicked() {
                            save(env, &[Edit::Remove("api_token")], "Token removal");
                        }
                        if ui.small_button(if self.show_token { "Hide" } else { "Show" }).clicked() {
                            self.show_token = !self.show_token;
                        }
                        let shown = if self.show_token { token.clone().unwrap_or_default() } else { "••••••••••••".to_string() };
                        ui.label(RichText::new(shown).monospace());
                    } else if ui.button("Create a token").clicked() {
                        save(env, &[Edit::Set("api_token", random_token().into())], "Token");
                    }
                });
            });

            ui.add_space(6.0);
            widgets::section(ui, "Copy and paste");
            let base = format!("http://127.0.0.1:{}", port);
            let auth = if token.is_some() { " -H \"Authorization: Bearer <token>\"" } else { "" };
            let exe = std::env::current_exe().map(|p| p.display().to_string()).unwrap_or_else(|_| "voclaude.exe".into());
            let mut copied = false;
            copied |= widgets::copy_block(ui, "Prompt for an agent (Claude, Codex, …)", &agent_prompt(port, token.as_deref()));
            ui.add_space(4.0);
            copied |= widgets::copy_block(ui, "Base URL for OpenAI clients", &format!("{}/v1", base));
            ui.add_space(4.0);
            copied |= widgets::copy_block(ui, "Upload a file", &format!("curl{} -F file=@meeting.m4a {}/v1/audio/transcriptions", auth, base));
            ui.add_space(4.0);
            copied |= widgets::copy_block(
                ui,
                "A file on this computer, no upload",
                &format!("curl{} -H \"Content-Type: application/json\" -d \"{{\\\"path\\\": \\\"C:/Videos/talk.mp4\\\", \\\"response_format\\\": \\\"text\\\"}}\" {}/v1/audio/transcriptions", auth, base),
            );
            ui.add_space(4.0);
            copied |= widgets::copy_block(ui, "Command line (needs Voclaude running)", &format!("\"{}\" transcribe meeting.mp4", exe));
            if copied {
                env.toast("Copied to the clipboard");
            }
            ui.add_space(8.0);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_names_the_endpoint_and_token() {
        let p = agent_prompt(7771, None);
        assert!(p.contains("POST http://127.0.0.1:7771/v1/audio/transcriptions"));
        assert!(!p.contains("Bearer"));
        let p = agent_prompt(7770, Some("abc"));
        assert!(p.contains("Authorization: Bearer abc"));
        assert_eq!(random_token().len(), 32);
    }
}

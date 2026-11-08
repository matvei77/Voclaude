//! Model: pick the speech engine, then only that engine's settings appear.
//!
//! Each engine is described once (`ENGINES`): what it is, what it supports,
//! and which config.toml keys belong to it. The keys stay flat in config.toml
//! (`parakeet_*` and the Qwen keys), so switching engines keeps each one's
//! settings. Changes apply at the next recording (the worker restarts).

use super::{theme, widgets, Env};
use crate::config::{Config, DEFAULT_PARAKEET_MODEL, MODEL_QWEN3_ASR_0_6B, MODEL_QWEN3_ASR_1_7B};
use crate::config_edit::{self, Edit};
use eframe::egui::{self, RichText};

struct EngineInfo {
    id: &'static str,
    name: &'static str,
    summary: &'static str,
    facts: &'static [(&'static str, &'static str)],
}

const ENGINES: [EngineInfo; 2] = [
    EngineInfo {
        id: "parakeet",
        name: "Parakeet Ultra",
        summary: "Default. Fast on any GPU (Vulkan) and on the CPU.",
        facts: &[
            ("Size", "0.6B parameters, 0.6 GB file"),
            ("Languages", "25 European, incl. Russian; picked per segment"),
            ("Device", "GPU (Vulkan) or CPU"),
            ("Names", "Full: biased re-decode of unsure words"),
        ],
    },
    EngineInfo {
        id: "qwen",
        name: "Qwen3-ASR",
        summary: "Slower; strong English. 0.6B or 1.7B.",
        facts: &[
            ("Size", "0.6B or 1.7B parameters"),
            ("Languages", "Many; one language hint for all"),
            ("Device", if cfg!(feature = "cuda") { "NVIDIA GPU (CUDA) or CPU" } else { "CPU in this build (GPU needs the CUDA build)" }),
            ("Names", "Not supported (the list is ignored)"),
        ],
    },
];

/// Parakeet files we know exist in the default repository.
const PARAKEET_FILES: [(&str, &str, &str); 2] = [
    ("Q6_K", DEFAULT_PARAKEET_MODEL, "0.6 GB, default"),
    ("Q8_0", "handy-computer/parakeet-ultra-gguf/parakeet-ultra-Q8_0.gguf", "0.74 GB, slower on CPU"),
];

#[derive(Default)]
pub struct State {
    custom_model: Option<String>,
    language: Option<String>,
    show_advanced: bool,
}

/// (name, device) for the status strip from the worker's model label, e.g.
/// "transcribe.cpp (parakeet-ultra-Q6_K on vulkan (NVIDIA GeForce RTX 3060))".
pub fn split_label(label: &str, use_gpu: bool) -> (String, String) {
    let inner = label.strip_prefix("transcribe.cpp (").and_then(|s| s.strip_suffix(')')).unwrap_or(label);
    let (name, rest) = match inner.split_once(" on ") {
        Some((n, r)) => (n.to_string(), Some(r)),
        None => (inner.to_string(), None),
    };
    let detail = rest
        .and_then(|r| r.split_once('(').map(|(_, d)| d.trim_end_matches(')')))
        .map(|d| d.trim_start_matches("NVIDIA ").trim_start_matches("GeForce ").trim_start_matches("AMD ").to_string());
    let device = match (use_gpu, detail) {
        (true, Some(d)) => format!("GPU · {}", d),
        (true, None) => "GPU".to_string(),
        (false, _) => "CPU".to_string(),
    };
    (name, device)
}

/// Whether the model file(s) are on disk already.
fn downloaded(engine: &str, model: &str) -> bool {
    let cache = hf_hub::Cache::from_env();
    if engine == "parakeet" {
        if std::path::Path::new(model).is_file() {
            return true;
        }
        let parts: Vec<&str> = model.splitn(3, '/').collect();
        if parts.len() == 3 {
            return cache.model(format!("{}/{}", parts[0], parts[1])).get(parts[2]).is_some();
        }
        false
    } else {
        let repo = cache.model(model.to_string());
        repo.get("config.json").is_some() && (repo.get("model.safetensors").is_some() || repo.get("model.safetensors.index.json").is_some())
    }
}

fn save(env: &mut Env, edits: &[Edit], what: &str) {
    match config_edit::save(edits) {
        Ok(()) => env.toast(format!("{} saved. Applies at the next recording.", what)),
        Err(e) => env.toast_error(format!("Not saved: {}", e)),
    }
}

impl State {
    pub fn show(&mut self, ui: &mut egui::Ui, env: &mut Env) {
        widgets::page_header(ui, "Model", "The speech engine and its settings. Changes apply at the next recording; the model loads then (downloaded once).");
        let config = env.config.clone();
        let current = if config.uses_parakeet() { "parakeet" } else { "qwen" };

        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            widgets::section(ui, "Engine");
            ui.columns(ENGINES.len(), |cols| {
                for (i, e) in ENGINES.iter().enumerate() {
                    let selected = e.id == current;
                    let ui = &mut cols[i];
                    let model = if e.id == "parakeet" { config.parakeet_model.as_str() } else { config.model.as_str() };
                    let r = engine_card(ui, e, selected, downloaded(e.id, model));
                    if r.clicked() && !selected {
                        save(env, &[Edit::Set("engine", e.id.into())], &format!("Engine {}", e.name));
                    }
                }
            });
            ui.add_space(10.0);
            if current == "parakeet" {
                self.parakeet(ui, env, &config);
            } else {
                self.qwen(ui, env, &config);
            }
            ui.add_space(8.0);
            if let Some(s) = env.status {
                let (name, device) = split_label(&s.model, s.use_gpu);
                let state = if s.model_loaded { "loaded" } else { "not loaded (loads at the next recording)" };
                widgets::hint(ui, &format!("Running now: {} on {}, {}.", name, device, state));
            }
        });
    }

    fn parakeet(&mut self, ui: &mut egui::Ui, env: &mut Env, c: &Config) {
        widgets::section(ui, "Parakeet settings");
        widgets::card(ui, |ui| {
            widgets::setting_row(ui, "Model file", "Two quantizations of the same model, same accuracy. Q8_0 is ~40% slower on the CPU. Downloaded from Hugging Face once.", |ui| {
                let known = PARAKEET_FILES.iter().any(|(_, spec, _)| *spec == c.parakeet_model);
                ui.vertical(|ui| {
                    for (label, spec, note) in PARAKEET_FILES {
                        let have = downloaded("parakeet", spec);
                        let text = format!("{}  ·  {}{}", label, note, if have { "" } else { " · downloads" });
                        if ui.radio(c.parakeet_model == spec, text).clicked() && c.parakeet_model != spec {
                            save(env, &[Edit::Set("parakeet_model", spec.into())], "Model file");
                        }
                    }
                    if ui.radio(!known, "Other file…").clicked() && known {
                        self.custom_model = Some(String::new());
                    }
                });
            });
            let known = PARAKEET_FILES.iter().any(|(_, spec, _)| *spec == c.parakeet_model);
            if !known || self.custom_model.is_some() {
                let text = self.custom_model.get_or_insert_with(|| if known { String::new() } else { c.parakeet_model.clone() });
                widgets::setting_row(ui, "Custom model", "A local .gguf file, or <owner>/<repo>/<file> on Hugging Face.", |ui| {
                    let r = ui.add(egui::TextEdit::singleline(text).desired_width(250.0));
                    if r.lost_focus() && !text.trim().is_empty() && text.trim() != c.parakeet_model {
                        let v = text.trim().to_string();
                        save(env, &[Edit::Set("parakeet_model", v.into())], "Model file");
                    }
                });
            }
            ui.separator();
            widgets::setting_row(ui, "Device", "GPU through Vulkan (any vendor), or the CPU.", |ui| {
                if ui.selectable_label(!c.use_gpu, "CPU").clicked() && c.use_gpu {
                    save(env, &[Edit::Set("use_gpu", false.into())], "Device");
                }
                if ui.selectable_label(c.use_gpu, "GPU").clicked() && !c.use_gpu {
                    save(env, &[Edit::Set("use_gpu", true.into())], "Device");
                }
            });
            if !c.use_gpu {
                widgets::setting_row(ui, "CPU threads", "0 = automatic.", |ui| {
                    let max = std::thread::available_parallelism().map(|n| n.get() as u32).unwrap_or(16);
                    let mut v = c.cpu_threads.min(max);
                    let r = ui.add(egui::Slider::new(&mut v, 0..=max));
                    if r.drag_stopped() || (r.changed() && !r.dragged()) {
                        save(env, &[Edit::Set("cpu_threads", (v as i64).into())], "CPU threads");
                    }
                });
            }
            ui.separator();
            widgets::setting_row(
                ui,
                "Russian re-check",
                "Short Russian phrases inside English come out in Latin letters. With this on, unsure English words are decoded again and kept if they come back as confident Russian. ~5% extra GPU time.",
                |ui| {
                    let mut on = c.parakeet_recheck;
                    if toggle(ui, &mut on).clicked() {
                        save(env, &[Edit::Set("parakeet_recheck", on.into())], "Russian re-check");
                    }
                },
            );
            ui.add_space(2.0);
            if ui.link(if self.show_advanced { "Hide advanced" } else { "Advanced" }).clicked() {
                self.show_advanced = !self.show_advanced;
            }
            if self.show_advanced {
                ui.add_space(4.0);
                widgets::setting_row(ui, "Unsure-word log", "Keep the words Parakeet was unsure of (word, confidence; never sentences) for Names suggestions. At most ~1 MB.", |ui| {
                    let mut on = c.parakeet_unsure_log;
                    if toggle(ui, &mut on).clicked() {
                        save(env, &[Edit::Set("parakeet_unsure_log", on.into())], "Unsure-word log");
                    }
                });
            }
        });
    }

    fn qwen(&mut self, ui: &mut egui::Ui, env: &mut Env, c: &Config) {
        widgets::section(ui, "Qwen settings");
        widgets::card(ui, |ui| {
            widgets::setting_row(ui, "Size", "1.7B is more accurate, 0.6B is faster and smaller.", |ui| {
                for (label, id) in [("1.7B", MODEL_QWEN3_ASR_1_7B), ("0.6B", MODEL_QWEN3_ASR_0_6B)] {
                    let have = downloaded("qwen", id);
                    let text = if have { label.to_string() } else { format!("{} (downloads)", label) };
                    if ui.selectable_label(c.model == id, text).clicked() && c.model != id {
                        save(env, &[Edit::Set("model", id.into())], "Model size");
                    }
                }
            });
            if cfg!(feature = "cuda") {
                widgets::setting_row(ui, "Device", "NVIDIA GPU through CUDA, or the CPU.", |ui| {
                    if ui.selectable_label(!c.use_gpu, "CPU").clicked() && c.use_gpu {
                        save(env, &[Edit::Set("use_gpu", false.into())], "Device");
                    }
                    if ui.selectable_label(c.use_gpu, "GPU").clicked() && !c.use_gpu {
                        save(env, &[Edit::Set("use_gpu", true.into())], "Device");
                    }
                });
                widgets::setting_row(ui, "GPU weights", "q8_0: 8-bit, same quality as F16, faster, less video memory.", |ui| {
                    for q in ["none", "q8_0"] {
                        if ui.selectable_label(c.quantization.eq_ignore_ascii_case(q), if q == "none" { "F16" } else { q }).clicked() {
                            save(env, &[Edit::Set("quantization", q.into())], "GPU weights");
                        }
                    }
                });
            } else {
                widgets::setting_row(ui, "Device", "This build runs Qwen on the CPU only.", |ui| {
                    ui.label(RichText::new("CPU").color(theme::MUTED));
                });
            }
            let text = self.language.get_or_insert_with(|| c.language.clone().unwrap_or_default());
            widgets::setting_row(ui, "Language hint", "A language code such as en or ru. Empty: detect.", |ui| {
                let r = ui.add(egui::TextEdit::singleline(text).hint_text("auto").desired_width(80.0));
                if r.lost_focus() {
                    let v = text.trim().to_string();
                    if Some(v.clone()) != c.language.clone().filter(|l| !l.is_empty()) {
                        let edit = if v.is_empty() { Edit::Remove("language") } else { Edit::Set("language", v.into()) };
                        save(env, &[edit], "Language hint");
                    }
                }
            });
            if ui.link(if self.show_advanced { "Hide advanced" } else { "Advanced" }).clicked() {
                self.show_advanced = !self.show_advanced;
            }
            if self.show_advanced {
                widgets::setting_row(ui, "Token limit", "Most tokens one transcription may produce.", |ui| {
                    let mut v = c.max_new_tokens;
                    let r = ui.add(egui::DragValue::new(&mut v).clamp_range(64..=8192).speed(16));
                    if r.drag_stopped() || r.lost_focus() {
                        save(env, &[Edit::Set("max_new_tokens", (v as i64).into())], "Token limit");
                    }
                });
                widgets::setting_row(ui, "Adaptive token limit", "Fewer tokens for short recordings (less memory).", |ui| {
                    let mut on = c.adaptive_max_new_tokens;
                    if toggle(ui, &mut on).clicked() {
                        save(env, &[Edit::Set("adaptive_max_new_tokens", on.into())], "Adaptive token limit");
                    }
                });
                widgets::setting_row(ui, "Chunk length (s)", "Long audio is split at pauses into chunks of at most this length.", |ui| {
                    let mut v = c.max_chunk_seconds;
                    let r = ui.add(egui::DragValue::new(&mut v).clamp_range(10..=300));
                    if r.drag_stopped() || r.lost_focus() {
                        save(env, &[Edit::Set("max_chunk_seconds", (v as i64).into())], "Chunk length");
                    }
                });
            }
        });
    }
}

fn engine_card(ui: &mut egui::Ui, e: &EngineInfo, selected: bool, have: bool) -> egui::Response {
    let stroke = if selected { egui::Stroke::new(1.5, theme::ACCENT) } else { egui::Stroke::new(1.0, theme::BORDER) };
    let r = egui::Frame::none()
        .fill(if selected { theme::BG_CARD_HI } else { theme::BG_CARD })
        .stroke(stroke)
        .rounding(8.0)
        .inner_margin(egui::Margin::symmetric(14.0, 12.0))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.label(RichText::new(e.name).font(theme::heading(15.0)).color(theme::TEXT));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if selected {
                        ui.label(RichText::new("IN USE").font(theme::heading(10.5)).color(theme::ACCENT));
                    } else {
                        ui.label(RichText::new(if have { "downloaded" } else { "not downloaded" }).size(11.0).color(theme::FAINT));
                    }
                });
            });
            ui.label(RichText::new(e.summary).color(theme::MUTED));
            ui.add_space(4.0);
            egui::Grid::new(e.id).num_columns(2).spacing([10.0, 3.0]).show(ui, |ui| {
                for (k, v) in e.facts {
                    ui.label(RichText::new(*k).size(12.0).color(theme::FAINT));
                    ui.label(RichText::new(*v).size(12.0).color(theme::TEXT));
                    ui.end_row();
                }
            });
            if !selected {
                ui.add_space(4.0);
                ui.label(RichText::new("Click to use").size(11.5).color(theme::ACCENT));
            }
        })
        .response;
    let r = r.interact(egui::Sense::click());
    if r.hovered() && !selected {
        ui.output_mut(|o| o.cursor_icon = egui::CursorIcon::PointingHand);
    }
    r
}

/// An on/off switch.
pub fn toggle(ui: &mut egui::Ui, on: &mut bool) -> egui::Response {
    let size = egui::vec2(36.0, 20.0);
    let (rect, mut r) = ui.allocate_exact_size(size, egui::Sense::click());
    if r.clicked() {
        *on = !*on;
        r.mark_changed();
    }
    let t = ui.ctx().animate_bool(r.id, *on);
    let bg = if *on { egui::Color32::from_rgb(60, 102, 214) } else { theme::BG_CARD_HI };
    let p = ui.painter();
    p.rect(rect, 10.0, bg, egui::Stroke::new(1.0, if *on { theme::ACCENT } else { theme::BORDER_HI }));
    let x = egui::lerp((rect.left() + 10.0)..=(rect.right() - 10.0), t);
    p.circle_filled(egui::pos2(x, rect.center().y), 7.0, if *on { egui::Color32::WHITE } else { theme::MUTED });
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_labels_split_into_name_and_device() {
        let (n, d) = split_label("transcribe.cpp (parakeet-ultra-Q6_K on vulkan (NVIDIA GeForce RTX 3060))", true);
        assert_eq!((n.as_str(), d.as_str()), ("parakeet-ultra-Q6_K", "GPU · RTX 3060"));
        let (n, d) = split_label("parakeet-ultra-q6_k", false);
        assert_eq!((n.as_str(), d.as_str()), ("parakeet-ultra-q6_k", "CPU"));
    }
}

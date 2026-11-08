//! Names: the `parakeet_phrases` list, spellings worth adding, and a check of
//! what the list changes on recent recordings. The same operations as
//! `voclaude phrases`, on the same config.toml.

use super::{theme, widgets, Env};
use crate::inference::{parse_phrases, Phrase};
use crate::phrases::{self, Suggestion};
use eframe::egui::{self, RichText};
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Recordings `Check` transcribes.
const CHECK_RECORDINGS: usize = 20;

struct CheckRun {
    lines: Arc<Mutex<Vec<String>>>,
    running: Arc<AtomicBool>,
}

#[derive(Default)]
pub struct State {
    new_term: String,
    new_forms: String,
    form_inputs: HashMap<String, String>,
    confirm_remove: Option<String>,
    analysed_for: Option<Vec<String>>,
    analysed_at: Option<Instant>,
    suggestions: Vec<Suggestion>,
    recurring: Vec<(String, usize, f64)>,
    written: Vec<(String, usize)>,
    dismissed: Option<HashSet<String>>,
    check: Option<CheckRun>,
}

fn dismissed_path() -> Option<std::path::PathBuf> {
    crate::config::project_dirs().map(|d| d.data_dir().join("names-dismissed.json"))
}

fn key(term: &str, word: &str) -> String {
    format!("{}|{}", term.to_lowercase(), word.to_lowercase())
}

impl State {
    pub fn badge(&self, config: &crate::config::Config) -> Option<String> {
        let n = parse_phrases(&config.parakeet_phrases).len();
        Some(n.to_string())
    }

    fn dismissed(&mut self) -> &mut HashSet<String> {
        self.dismissed.get_or_insert_with(|| {
            dismissed_path()
                .and_then(|p| std::fs::read_to_string(p).ok())
                .and_then(|t| serde_json::from_str::<Vec<String>>(&t).ok())
                .map(|v| v.into_iter().collect())
                .unwrap_or_default()
        })
    }

    fn save_dismissed(&mut self) {
        let mut list: Vec<String> = self.dismissed().iter().cloned().collect();
        list.sort();
        if let (Some(path), Ok(text)) = (dismissed_path(), serde_json::to_string_pretty(&list)) {
            let _ = crate::config::Config::write_atomic(&path, &text);
        }
    }

    /// Recompute suggestions when the list changed, or every 30 s.
    fn analyse(&mut self, entries: &[String], phrases: &[Phrase]) {
        let fresh = self.analysed_for.as_deref() == Some(entries) && self.analysed_at.is_some_and(|t| t.elapsed() < Duration::from_secs(30));
        if fresh {
            return;
        }
        let history: Vec<String> = phrases::read_history().unwrap_or_default().into_iter().map(|(_, t, _)| t).collect();
        let unsure = phrases::read_unsure_log();
        self.suggestions = if phrases.is_empty() { Vec::new() } else { phrases::suggestions(phrases, &history, &unsure) };
        self.recurring = phrases::recurring_doubts(phrases, &unsure);
        self.written = phrases::written_counts(&unsure);
        self.analysed_for = Some(entries.to_vec());
        self.analysed_at = Some(Instant::now());
    }

    pub fn show(&mut self, ui: &mut egui::Ui, env: &mut Env) {
        let entries = env.config.parakeet_phrases.clone();
        let phrases_list = parse_phrases(&entries);
        self.analyse(&entries, &phrases_list);

        widgets::page_header(
            ui,
            "Names",
            "Names and terms Parakeet should spell exactly. A spoken form is how Parakeet writes the name when it mishears it. Changes apply within a few seconds.",
        );
        if !env.config.uses_parakeet() {
            widgets::banner(ui, "Names work with the Parakeet engine only. The current engine (Model tab) ignores this list.", false);
        }

        let mut changed: Option<Result<String, String>> = None;
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            ui.columns(2, |cols| {
                // ---- Left: the list -------------------------------------------------
                let ui = &mut cols[0];
                widgets::section(ui, &format!("Your names ({})", phrases_list.len()));
                widgets::card(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.add(egui::TextEdit::singleline(&mut self.new_term).hint_text("Name, e.g. Jonathan").desired_width(150.0));
                        let w = ui.available_width();
                        ui.add(egui::TextEdit::singleline(&mut self.new_forms).hint_text("Spoken forms, comma-separated").desired_width(w));
                    });
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        let can = !self.new_term.trim().is_empty();
                        if ui.add_enabled(can, widgets::primary("Add")).clicked() {
                            let forms: Vec<String> = self.new_forms.split(',').map(phrases::tidy).filter(|f| !f.is_empty()).collect();
                            changed = Some(edit_list(|list| {
                                let term = phrases::tidy(&self.new_term);
                                if !phrases::allowed(&term) || forms.iter().any(|f| !phrases::allowed(f)) {
                                    return Err("Names cannot contain '=' or ','.".into());
                                }
                                if !phrases::add_to(list, &term, &forms) {
                                    return Err(format!("{} is already in the list", term));
                                }
                                Ok(format!("Added {}", term))
                            }));
                            if matches!(changed, Some(Ok(_))) {
                                self.new_term.clear();
                                self.new_forms.clear();
                            }
                        }
                        widgets::hint(ui, "Add spoken forms to an existing name the same way.");
                    });
                });
                ui.add_space(6.0);
                if phrases_list.is_empty() {
                    widgets::hint(ui, "The list is empty. Add a name above, or click a misspelled word in Dictations.");
                }
                for p in &phrases_list {
                    let input = self.form_inputs.entry(p.term.clone()).or_default();
                    let mut remove_form: Option<String> = None;
                    let mut add_form = false;
                    let mut remove_term = false;
                    let confirming = self.confirm_remove.as_deref() == Some(&p.term);
                    widgets::card(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(&p.term).font(theme::heading(14.5)).color(theme::TEXT));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if confirming {
                                    if ui.small_button("Keep").clicked() {
                                        self.confirm_remove = None;
                                    }
                                    if ui.add(widgets::danger("Remove").small()).clicked() {
                                        remove_term = true;
                                    }
                                } else if ui.small_button("Remove").clicked() {
                                    self.confirm_remove = Some(p.term.clone());
                                }
                            });
                        });
                        ui.horizontal_wrapped(|ui| {
                            ui.spacing_mut().item_spacing = egui::vec2(4.0, 4.0);
                            for f in &p.forms {
                                if widgets::chip(ui, f, true) {
                                    remove_form = Some(f.clone());
                                }
                            }
                            let r = ui.add(egui::TextEdit::singleline(input).hint_text("+ spoken form").desired_width(110.0).font(egui::TextStyle::Small));
                            if r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) && !input.trim().is_empty() {
                                add_form = true;
                            }
                        });
                    });
                    ui.add_space(4.0);
                    if let Some(f) = remove_form {
                        changed = Some(edit_list(|list| {
                            phrases::remove_from(list, &p.term, &[f.clone()]);
                            Ok(format!("Removed {} from {}", f, p.term))
                        }));
                    }
                    if add_form {
                        let form = phrases::tidy(input);
                        input.clear();
                        changed = Some(edit_list(|list| {
                            if !phrases::allowed(&form) {
                                return Err("Spoken forms cannot contain '=' or ','.".into());
                            }
                            phrases::add_to(list, &p.term, &[form.clone()]);
                            Ok(format!("Added {} → {}", form, p.term))
                        }));
                    }
                    if remove_term {
                        self.confirm_remove = None;
                        changed = Some(edit_list(|list| {
                            phrases::remove_from(list, &p.term, &[]);
                            Ok(format!("Removed {}", p.term))
                        }));
                    }
                }

                // ---- Right: suggestions, feedback, check ----------------------------
                let ui = &mut cols[1];
                widgets::section(ui, "Spellings worth adding");
                let dismissed = self.dismissed().clone();
                let visible: Vec<Suggestion> = self.suggestions.iter().filter(|s| !dismissed.contains(&key(&s.term, &s.word))).take(12).cloned().collect();
                let mut dismiss: Option<String> = None;
                widgets::card(ui, |ui| {
                    if visible.is_empty() {
                        widgets::hint(ui, "No new spellings found in your history and the unsure-word log.");
                    }
                    for s in &visible {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(&s.word).color(theme::WARNING));
                            ui.label(RichText::new("→").color(theme::FAINT));
                            ui.label(RichText::new(&s.term).color(theme::TEXT));
                            let mut why = Vec::new();
                            if s.unsure > 0 {
                                why.push(format!("unsure {}×", s.unsure));
                            }
                            if s.history > 0 {
                                why.push(format!("history {}×", s.history));
                            }
                            ui.label(RichText::new(why.join(" · ")).size(11.5).color(theme::FAINT));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.small_button("Dismiss").clicked() {
                                    dismiss = Some(key(&s.term, &s.word));
                                }
                                if ui.small_button("Add").clicked() {
                                    let (t, w) = (s.term.clone(), s.word.clone());
                                    changed = Some(edit_list(|list| {
                                        phrases::add_to(list, &t, &[w.clone()]);
                                        Ok(format!("Added {} → {}", w, t))
                                    }));
                                }
                            });
                        });
                    }
                    if !visible.is_empty() {
                        ui.add_space(2.0);
                        widgets::hint(ui, "Check each one: add only spellings that really are this name.");
                    }
                });
                if let Some(k) = dismiss {
                    self.dismissed().insert(k);
                    self.save_dismissed();
                }

                // Words already proposed above are not repeated here.
                let recurring: Vec<&(String, usize, f64)> =
                    self.recurring.iter().filter(|(w, _, _)| !visible.iter().any(|s| s.word.eq_ignore_ascii_case(w))).take(8).collect();
                if !recurring.is_empty() {
                    ui.add_space(6.0);
                    widgets::section(ui, "Unsure more than once, no name yet");
                    widgets::card(ui, |ui| {
                        for (w, n, p) in recurring {
                            ui.horizontal(|ui| {
                                ui.label(RichText::new(w).color(theme::TEXT));
                                ui.label(RichText::new(format!("{}× · lowest confidence {:.2}", n, p)).size(11.5).color(theme::FAINT));
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    if ui.small_button("Use as spoken form…").on_hover_text("Fills the add form on the left").clicked() {
                                        self.new_forms = w.clone();
                                    }
                                });
                            });
                        }
                    });
                }

                if !self.written.is_empty() {
                    ui.add_space(6.0);
                    widgets::section(ui, "Written by the list");
                    ui.horizontal_wrapped(|ui| {
                        for (t, n) in &self.written {
                            widgets::chip(ui, &format!("{}  {}×", t, n), false);
                        }
                    });
                }

                ui.add_space(6.0);
                widgets::section(ui, "Check");
                widgets::card(ui, |ui| {
                    widgets::hint(
                        ui,
                        &format!(
                            "Transcribes your {} newest kept recordings with and without the list and shows every word the list changes. Each change should be a word you meant.",
                            CHECK_RECORDINGS
                        ),
                    );
                    ui.add_space(4.0);
                    let running = self.check.as_ref().is_some_and(|c| c.running.load(Ordering::Relaxed));
                    let blocked = if env.recording {
                        Some("Stop recording first.")
                    } else if !env.config.uses_parakeet() {
                        Some("Needs the Parakeet engine.")
                    } else if phrases_list.is_empty() {
                        Some("Add a name first.")
                    } else {
                        None
                    };
                    ui.horizontal(|ui| {
                        let label = if running { "Checking…" } else { "Run check" };
                        if ui.add_enabled(blocked.is_none() && !running, widgets::primary(label)).clicked() {
                            self.check = Some(start_check());
                        }
                        if let Some(why) = blocked {
                            ui.label(RichText::new(why).size(12.0).color(theme::MUTED));
                        } else if running {
                            ui.spinner();
                        }
                    });
                    if let Some(c) = &self.check {
                        let lines = c.lines.lock().unwrap_or_else(|e| e.into_inner()).clone();
                        ui.add_space(4.0);
                        egui::Frame::none().fill(theme::BG_CODE).rounding(6.0).inner_margin(8.0).show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            egui::ScrollArea::vertical().max_height(220.0).stick_to_bottom(true).show(ui, |ui| {
                                for l in &lines {
                                    ui.add(egui::Label::new(RichText::new(l).monospace().size(12.0).color(theme::TEXT)).wrap(true));
                                }
                            });
                        });
                        if running {
                            ui.ctx().request_repaint_after(Duration::from_millis(300));
                        }
                    }
                });
            });
        });

        match changed {
            Some(Ok(msg)) => {
                self.analysed_at = None;
                env.toast(msg);
            }
            Some(Err(e)) => env.toast_error(e),
            None => {}
        }
    }
}

/// Load the list from config.toml, change it, save it.
fn edit_list(change: impl FnOnce(&mut Vec<Phrase>) -> Result<String, String>) -> Result<String, String> {
    let (path, text) = crate::config_edit::read().map_err(|e| e.to_string())?;
    let mut list = parse_phrases(&phrases::entries_of(&text).map_err(|e| e.to_string())?);
    let msg = change(&mut list)?;
    phrases::save(&path, &text, &list).map_err(|e| e.to_string())?;
    Ok(msg)
}

/// Run `voclaude phrases check` in a child process and collect its output.
fn start_check() -> CheckRun {
    let lines = Arc::new(Mutex::new(vec![format!("$ voclaude phrases check {}", CHECK_RECORDINGS)]));
    let running = Arc::new(AtomicBool::new(true));
    let (l, r) = (lines.clone(), running.clone());
    std::thread::spawn(move || {
        let push = |s: String| l.lock().unwrap_or_else(|e| e.into_inner()).push(s);
        let exe = match std::env::current_exe() {
            Ok(e) => e,
            Err(e) => {
                push(format!("error: {}", e));
                r.store(false, Ordering::Relaxed);
                return;
            }
        };
        let mut cmd = Command::new(exe);
        cmd.args(["phrases", "check", &CHECK_RECORDINGS.to_string()]).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
        #[cfg(target_os = "windows")]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
        }
        match cmd.spawn() {
            Ok(mut child) => {
                let err = child.stderr.take();
                let l2 = l.clone();
                let err_thread = std::thread::spawn(move || {
                    if let Some(err) = err {
                        for line in BufReader::new(err).lines().map_while(Result::ok) {
                            l2.lock().unwrap_or_else(|e| e.into_inner()).push(line);
                        }
                    }
                });
                if let Some(out) = child.stdout.take() {
                    for line in BufReader::new(out).lines().map_while(Result::ok) {
                        push(line);
                    }
                }
                let _ = err_thread.join();
                match child.wait() {
                    Ok(s) if s.success() => {}
                    Ok(s) => push(format!("(exit code {})", s.code().unwrap_or(-1))),
                    Err(e) => push(format!("error: {}", e)),
                }
            }
            Err(e) => push(format!("could not start the check: {}", e)),
        }
        r.store(false, Ordering::Relaxed);
    });
    CheckRun { lines, running }
}

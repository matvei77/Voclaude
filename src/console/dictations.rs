//! Dictations: the history, newest first, searchable. Rows are laid out in
//! steps of `PAGE` as you scroll. Click a word to add it to Names.

use super::{localtime, theme, widgets, Env};
use crate::console::ipc::FromConsole;
use crate::inference::{parse_phrases, word_core};
use eframe::egui::{self, RichText};
use serde::Deserialize;
use std::collections::HashSet;

/// Rows added each time the list end scrolls into view.
const PAGE: usize = 30;
/// Longer texts are shortened until "more" is clicked.
const COLLAPSE_CHARS: usize = 360;

#[derive(Deserialize)]
struct RawAudio {
    duration_ms: u64,
}

#[derive(Deserialize)]
struct RawEntry {
    id: String,
    created_at_ms: u64,
    text: String,
    audio: Option<RawAudio>,
    audio_path: Option<String>,
}

struct Entry {
    id: String,
    created_at_ms: u64,
    text: String,
    lower: String,
    duration_ms: Option<u64>,
    words: usize,
    has_audio: bool,
}

/// The "add to Names" dialog.
struct Pick {
    heard: String,
    term: String,
}

#[derive(Default)]
pub struct State {
    loaded: bool,
    entries: Vec<Entry>,
    error: Option<String>,
    search: String,
    shown: usize,
    expanded: HashSet<String>,
    confirm_delete: Option<String>,
    pick: Option<Pick>,
}

impl State {
    /// history.json changed: read it again.
    pub fn mark_stale(&mut self) {
        self.reload();
    }

    pub fn count_label(&self) -> String {
        if self.loaded {
            self.entries.len().to_string()
        } else {
            String::new()
        }
    }

    fn reload(&mut self) {
        self.loaded = true;
        self.error = None;
        let Some(path) = crate::config::project_dirs().map(|d| d.data_dir().join("history.json")) else {
            self.error = Some("Could not find the data folder.".into());
            return;
        };
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                self.entries.clear();
                return;
            }
            Err(e) => {
                self.error = Some(format!("Could not read {}: {}", path.display(), e));
                return;
            }
        };
        match serde_json::from_str::<Vec<RawEntry>>(&text) {
            Ok(raw) => {
                let mut entries: Vec<Entry> = raw
                    .into_iter()
                    .map(|r| Entry {
                        lower: r.text.to_lowercase(),
                        words: r.text.split_whitespace().count(),
                        duration_ms: r.audio.map(|a| a.duration_ms),
                        has_audio: r.audio_path.as_deref().is_some_and(|p| std::path::Path::new(p).is_file()),
                        id: r.id,
                        created_at_ms: r.created_at_ms,
                        text: r.text,
                    })
                    .collect();
                entries.sort_by(|a, b| b.created_at_ms.cmp(&a.created_at_ms));
                self.entries = entries;
            }
            Err(e) => self.error = Some(format!("history.json could not be read: {}", e)),
        }
        if self.shown == 0 {
            self.shown = PAGE;
        }
    }

    pub fn show(&mut self, ui: &mut egui::Ui, env: &mut Env) {
        if !self.loaded {
            self.reload();
        }
        widgets::page_header(ui, "Dictations", "Newest first. Each dictation is also on the clipboard when it finishes. Click a word to add it to Names.");
        if let Some(e) = &self.error {
            widgets::banner(ui, e, true);
        }

        let query = self.search.trim().to_lowercase();
        let matching: Vec<usize> =
            (0..self.entries.len()).filter(|&i| query.is_empty() || self.entries[i].lower.contains(&query)).collect();

        ui.horizontal(|ui| {
            let search = egui::TextEdit::singleline(&mut self.search)
                .hint_text("Search all dictations   (Ctrl+F)")
                .desired_width((ui.available_width() - 180.0).max(200.0));
            let r = ui.add(search);
            if ui.input(|i| i.modifiers.command && i.key_pressed(egui::Key::F)) {
                r.request_focus();
            }
            if r.changed() {
                self.shown = PAGE;
            }
            if !self.search.is_empty() && ui.small_button("Clear").clicked() {
                self.search.clear();
                self.shown = PAGE;
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let label = if query.is_empty() {
                    format!("{} dictations", self.entries.len())
                } else {
                    format!("{} of {} match", matching.len(), self.entries.len())
                };
                ui.label(RichText::new(label).size(12.0).color(theme::MUTED));
            });
        });
        ui.add_space(8.0);

        if self.entries.is_empty() && self.error.is_none() {
            ui.add_space(40.0);
            ui.vertical_centered(|ui| {
                ui.label(RichText::new("No dictations yet").font(theme::heading(15.0)).color(theme::MUTED));
                let key = env.status.map(|s| s.record_hotkey.clone()).unwrap_or_else(|| env.config.hotkey.clone());
                ui.label(RichText::new(format!("Press {} anywhere, speak, press {} again.", key, key)).color(theme::FAINT));
            });
            return;
        }

        let now = localtime::now_ms();
        let mut action: Option<RowAction> = None;
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            ui.set_width(ui.available_width());
            for &i in matching.iter().take(self.shown) {
                let e = &self.entries[i];
                if let Some(a) = row(ui, e, now, &query, self.expanded.contains(&e.id), self.confirm_delete.as_deref() == Some(&e.id)) {
                    action = Some(a);
                }
                ui.add_space(6.0);
            }
            if matching.len() > self.shown {
                let r = ui.label(RichText::new("Loading more…").size(12.0).color(theme::FAINT));
                if ui.is_rect_visible(r.rect) {
                    self.shown += PAGE;
                    ui.ctx().request_repaint();
                }
            } else if !matching.is_empty() {
                ui.add_space(4.0);
                ui.vertical_centered(|ui| ui.label(RichText::new("End of history").size(11.5).color(theme::FAINT)));
                ui.add_space(8.0);
            }
        });

        match action {
            Some(RowAction::Copy(text)) => {
                ui.output_mut(|o| o.copied_text = text);
                env.toast("Copied to the clipboard");
            }
            Some(RowAction::Expand(id, on)) => {
                if on {
                    self.expanded.insert(id);
                } else {
                    self.expanded.remove(&id);
                }
            }
            Some(RowAction::AskDelete(id)) => self.confirm_delete = Some(id),
            Some(RowAction::CancelDelete) => self.confirm_delete = None,
            Some(RowAction::Delete(id)) => {
                self.confirm_delete = None;
                if !env.send(FromConsole::DeleteHistory { id }) {
                    env.toast_error("Voclaude is not running; start it to delete dictations.");
                }
            }
            Some(RowAction::Word(w)) => {
                let phrases = parse_phrases(&env.config.parakeet_phrases);
                // Preselect the name this spelling is already filed under, if any.
                let term = phrases
                    .iter()
                    .find(|p| p.term.eq_ignore_ascii_case(&w) || p.forms.iter().any(|f| f.eq_ignore_ascii_case(&w)))
                    .map(|p| p.term.clone())
                    .unwrap_or_default();
                self.pick = Some(Pick { heard: w, term });
            }
            None => {}
        }
        self.pick_dialog(ui.ctx(), env);
    }

    fn pick_dialog(&mut self, ctx: &egui::Context, env: &mut Env) {
        let Some(pick) = self.pick.as_mut() else { return };
        let phrases = parse_phrases(&env.config.parakeet_phrases);
        let mut close = false;
        let mut save = false;
        egui::Window::new("Add to Names")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .fixed_size(egui::vec2(420.0, 0.0))
            .show(ctx, |ui| {
                widgets::hint(ui, "The heard spelling becomes a spoken form of the name. Parakeet then writes the name when it hears this spelling and is unsure of it.");
                ui.add_space(6.0);
                egui::Grid::new("pick").num_columns(2).spacing([12.0, 8.0]).show(ui, |ui| {
                    ui.label(RichText::new("Heard as").color(theme::MUTED));
                    ui.add(egui::TextEdit::singleline(&mut pick.heard).desired_width(260.0));
                    ui.end_row();
                    ui.label(RichText::new("Correct spelling").color(theme::MUTED));
                    let r = ui.add(egui::TextEdit::singleline(&mut pick.term).hint_text("e.g. Jonathan").desired_width(260.0));
                    if r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        save = true;
                    }
                    ui.end_row();
                });
                let heard = pick.heard.trim().to_string();
                if let Some(p) = phrases.iter().find(|p| p.forms.iter().any(|f| f.eq_ignore_ascii_case(&heard))) {
                    ui.label(RichText::new(format!("{} is already a spoken form of {}.", heard, p.term)).size(12.0).color(theme::WARNING));
                } else if phrases.iter().any(|p| p.term.eq_ignore_ascii_case(&heard)) {
                    ui.label(RichText::new(format!("{} is already a name in the list.", heard)).size(12.0).color(theme::WARNING));
                }
                if !phrases.is_empty() {
                    ui.add_space(4.0);
                    ui.label(RichText::new("Existing names").size(12.0).color(theme::FAINT));
                    ui.horizontal_wrapped(|ui| {
                        for p in &phrases {
                            let selected = p.term == pick.term;
                            if ui.selectable_label(selected, &p.term).clicked() {
                                pick.term = p.term.clone();
                            }
                        }
                    });
                }
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.add_enabled(!pick.term.trim().is_empty(), widgets::primary("Add to Names")).clicked() {
                        save = true;
                    }
                    if ui.button("Cancel").clicked() {
                        close = true;
                    }
                });
            });
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            close = true;
        }
        if save {
            match add_name(&pick.term, &pick.heard) {
                Ok(msg) => {
                    env.toast(msg);
                    close = true;
                }
                Err(e) => env.toast_error(e),
            }
        }
        if close {
            self.pick = None;
        }
    }
}

/// Save `heard` as a spoken form of `term` (or `term` alone).
pub(crate) fn add_name(term: &str, heard: &str) -> Result<String, String> {
    let term = crate::phrases::tidy(term);
    let heard = crate::phrases::tidy(heard);
    if term.is_empty() {
        return Err("Type the correct spelling.".into());
    }
    if !crate::phrases::allowed(&term) || !crate::phrases::allowed(&heard) {
        return Err("Names and spellings cannot contain '=' or ','.".into());
    }
    let (path, text) = crate::config_edit::read().map_err(|e| e.to_string())?;
    let mut phrases = parse_phrases(&crate::phrases::entries_of(&text).map_err(|e| e.to_string())?);
    let forms: Vec<String> = if heard.is_empty() || heard.eq_ignore_ascii_case(&term) { Vec::new() } else { vec![heard.clone()] };
    if !crate::phrases::add_to(&mut phrases, &term, &forms) {
        return Ok(format!("{} is already in Names", if forms.is_empty() { term } else { format!("{} → {}", heard, term) }));
    }
    crate::phrases::save(&path, &text, &phrases).map_err(|e| e.to_string())?;
    Ok(if forms.is_empty() { format!("Added {} to Names", term) } else { format!("Added {} → {}. Applies in a few seconds.", heard, term) })
}

enum RowAction {
    Copy(String),
    Expand(String, bool),
    AskDelete(String),
    CancelDelete,
    Delete(String),
    Word(String),
}

fn row(ui: &mut egui::Ui, e: &Entry, now: u64, query: &str, expanded: bool, confirm: bool) -> Option<RowAction> {
    let mut action = None;
    widgets::card(ui, |ui| {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 12.0;
            ui.label(RichText::new(localtime::when(e.created_at_ms, now)).size(12.0).color(theme::MUTED));
            if let Some(ms) = e.duration_ms {
                ui.label(RichText::new(widgets::duration(ms)).monospace().size(11.5).color(theme::FAINT)).on_hover_text("Length of the recording");
            }
            ui.label(RichText::new(format!("{} words", e.words)).size(12.0).color(theme::FAINT));
            if e.has_audio {
                ui.label(RichText::new("recording kept").size(11.5).color(theme::FAINT))
                    .on_hover_text("The audio is still on disk (Settings → Recordings kept). Names → Check uses it.");
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                if confirm {
                    if ui.small_button("Keep").clicked() {
                        action = Some(RowAction::CancelDelete);
                    }
                    if ui.add(widgets::danger("Delete").small()).clicked() {
                        action = Some(RowAction::Delete(e.id.clone()));
                    }
                    ui.label(RichText::new("Delete this dictation and its recording?").size(12.0).color(theme::DANGER));
                } else {
                    if ui.small_button("Delete").on_hover_text("Delete this dictation and its recording").clicked() {
                        action = Some(RowAction::AskDelete(e.id.clone()));
                    }
                    if ui.small_button("Copy").clicked() {
                        action = Some(RowAction::Copy(e.text.clone()));
                    }
                }
            });
        });
        ui.add_space(2.0);

        let long = e.text.chars().count() > COLLAPSE_CHARS;
        let mut budget = if long && !expanded { COLLAPSE_CHARS } else { usize::MAX };
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = egui::vec2(4.0, 2.0);
            for w in e.text.split_whitespace() {
                if budget < w.len() {
                    break;
                }
                budget = budget.saturating_sub(w.len() + 1);
                let hit = !query.is_empty() && w.to_lowercase().contains(query);
                let text = RichText::new(w).size(14.5).color(if hit { theme::ACCENT } else { theme::TEXT });
                // A slot under the word for the hover highlight.
                let under = ui.painter().add(egui::Shape::Noop);
                let r = ui.add(egui::Label::new(text).sense(egui::Sense::click()));
                if r.hovered() {
                    ui.painter().set(under, egui::Shape::rect_filled(r.rect.expand(1.5), 3.0, theme::ACCENT_BG));
                    ui.output_mut(|o| o.cursor_icon = egui::CursorIcon::PointingHand);
                }
                if r.on_hover_text("Add to Names").clicked() {
                    let core = word_core(w);
                    if !core.is_empty() {
                        action = Some(RowAction::Word(core.to_string()));
                    }
                }
            }
            if long {
                let link = if expanded { "show less" } else { "… show all" };
                if ui.link(RichText::new(link).size(13.0)).clicked() {
                    action = Some(RowAction::Expand(e.id.clone(), !expanded));
                }
            }
        });
    });
    action
}

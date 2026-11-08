//! Log: the newest log file in data/logs, read as it grows. Local times,
//! a level filter, and routine lines (heartbeats, window messages) hidden.

use super::{localtime, theme, widgets, Env};
use eframe::egui::{self, RichText};
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Bytes read from the end of the file when the tab opens.
const TAIL_BYTES: u64 = 512 * 1024;
const MAX_LINES: usize = 5000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Level {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

struct Line {
    ms: Option<u64>,
    level: Level,
    text: String,
}

pub struct State {
    file: Option<PathBuf>,
    offset: u64,
    partial: String,
    lines: Vec<Line>,
    read_at: Option<Instant>,
    min_level: Level,
    hide_routine: bool,
    search: String,
    follow: bool,
}

impl Default for State {
    fn default() -> Self {
        Self {
            file: None,
            offset: 0,
            partial: String::new(),
            lines: Vec::new(),
            read_at: None,
            min_level: Level::Info,
            hide_routine: true,
            search: String::new(),
            follow: true,
        }
    }
}

fn logs_dir() -> Option<PathBuf> {
    crate::config::project_dirs().map(|d| d.data_dir().join("logs"))
}

fn newest_log() -> Option<PathBuf> {
    let dir = logs_dir()?;
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with("voclaude.log"))
        .max_by_key(|e| e.metadata().and_then(|m| m.modified()).ok())
        .map(|e| e.path())
}

/// Lines the app writes on a timer or for every window message.
fn routine(text: &str) -> bool {
    const ROUTINE: [&str; 10] = [
        "=== MAIN LOOP STATUS ===",
        "Loop iterations:",
        "Messages pumped total:",
        "Current state:",
        "Event channel len:",
        "Windows message:",
        "=== APP EVENT RECEIVED ===",
        "Time since last activity",
        "Main thread ID",
        "Windows Thread ID",
    ];
    ROUTINE.iter().any(|r| text.contains(r))
}

fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            // ESC [ ... letter
            if chars.peek() == Some(&'[') {
                chars.next();
                for d in chars.by_ref() {
                    if d.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
            continue;
        }
        out.push(c);
    }
    out
}

fn parse(line: &str) -> Option<Line> {
    let line = strip_ansi(line);
    let trimmed = line.trim_end();
    if trimmed.is_empty() {
        return None;
    }
    let mut parts = trimmed.splitn(2, ' ');
    let first = parts.next().unwrap_or("");
    let rest = parts.next().unwrap_or("").trim_start();
    let Some(ms) = localtime::parse_utc(first) else {
        return Some(Line { ms: None, level: Level::Info, text: trimmed.to_string() });
    };
    let (level_word, text) = rest.split_once(' ').unwrap_or((rest, ""));
    let level = match level_word {
        "ERROR" => Level::Error,
        "WARN" => Level::Warn,
        "INFO" => Level::Info,
        "DEBUG" => Level::Debug,
        "TRACE" => Level::Trace,
        _ => return Some(Line { ms: Some(ms), level: Level::Info, text: rest.to_string() }),
    };
    Some(Line { ms: Some(ms), level, text: text.trim_start().to_string() })
}

impl State {
    /// Read what was added to the log since the last call.
    fn refresh(&mut self) {
        if self.read_at.is_some_and(|t| t.elapsed() < Duration::from_millis(800)) {
            return;
        }
        self.read_at = Some(Instant::now());
        let Some(path) = newest_log() else { return };
        if self.file.as_ref() != Some(&path) {
            self.file = Some(path.clone());
            self.lines.clear();
            self.partial.clear();
            let len = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            self.offset = len.saturating_sub(TAIL_BYTES);
        }
        let Ok(mut f) = std::fs::File::open(&path) else { return };
        let len = f.metadata().map(|m| m.len()).unwrap_or(0);
        if len < self.offset {
            self.offset = 0; // truncated
        }
        if len == self.offset || f.seek(SeekFrom::Start(self.offset)).is_err() {
            return;
        }
        let mut buf = Vec::new();
        if f.take(len - self.offset).read_to_end(&mut buf).is_err() {
            return;
        }
        self.offset = len;
        let chunk = String::from_utf8_lossy(&buf);
        let text = std::mem::take(&mut self.partial) + &chunk;
        let mut pieces: Vec<&str> = text.split('\n').collect();
        let tail = pieces.pop().unwrap_or("");
        self.partial = tail.to_string();
        for p in pieces {
            if let Some(line) = parse(p) {
                if line.ms.is_none() {
                    if let Some(prev) = self.lines.last_mut() {
                        prev.text.push('\n');
                        prev.text.push_str(&line.text);
                        continue;
                    }
                }
                self.lines.push(line);
            }
        }
        if self.lines.len() > MAX_LINES {
            let excess = self.lines.len() - MAX_LINES;
            self.lines.drain(..excess);
        }
    }

    pub fn show(&mut self, ui: &mut egui::Ui, env: &mut Env) {
        self.refresh();
        ui.ctx().request_repaint_after(Duration::from_secs(1));
        widgets::page_header(ui, "Log", "What Voclaude did, newest at the bottom. Local times. Attach the log file when you report a problem.");

        ui.horizontal(|ui| {
            for (label, level) in [("Errors", Level::Warn), ("Info", Level::Info), ("Debug", Level::Debug)] {
                if ui.selectable_label(self.min_level == level, label).clicked() {
                    self.min_level = level;
                }
            }
            ui.separator();
            ui.checkbox(&mut self.hide_routine, "Hide routine lines");
            ui.checkbox(&mut self.follow, "Follow");
            ui.add(egui::TextEdit::singleline(&mut self.search).hint_text("Filter").desired_width(160.0));
        });
        ui.add_space(6.0);

        let q = self.search.to_lowercase();
        let shown: Vec<&Line> = self
            .lines
            .iter()
            .filter(|l| l.level >= self.min_level)
            .filter(|l| !self.hide_routine || !routine(&l.text))
            .filter(|l| q.is_empty() || l.text.to_lowercase().contains(&q))
            .collect();

        let footer_h = 34.0;
        let height = (ui.available_height() - footer_h).max(120.0);
        egui::Frame::none().fill(theme::BG_CODE).stroke(egui::Stroke::new(1.0, theme::BORDER)).rounding(6.0).inner_margin(8.0).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.set_height(height - 18.0);
            let row_h = ui.text_style_height(&egui::TextStyle::Monospace) + 4.0;
            ui.spacing_mut().item_spacing.y = 0.0;
            ui.spacing_mut().interact_size.y = row_h;
            egui::ScrollArea::both()
                .auto_shrink([false, false])
                .stick_to_bottom(self.follow)
                .show_rows(ui, row_h, shown.len(), |ui, range| {
                    for l in &shown[range] {
                        ui.allocate_ui(egui::vec2(ui.available_width(), row_h), |ui| ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 10.0;
                            let time = l.ms.map(localtime::clock).unwrap_or_default();
                            ui.label(RichText::new(time).monospace().color(theme::FAINT));
                            let (tag, color) = match l.level {
                                Level::Error => ("ERROR", theme::DANGER),
                                Level::Warn => ("WARN ", theme::WARNING),
                                Level::Info => ("INFO ", theme::MUTED),
                                Level::Debug => ("DEBUG", theme::FAINT),
                                Level::Trace => ("TRACE", theme::FAINT),
                            };
                            ui.label(RichText::new(tag).monospace().color(color));
                            let first = l.text.lines().next().unwrap_or("");
                            let text_color = match l.level {
                                Level::Error => theme::DANGER,
                                Level::Warn => theme::WARNING,
                                _ => theme::TEXT,
                            };
                            let r = ui.add(egui::Label::new(RichText::new(first).monospace().color(text_color)).wrap(false));
                            if l.text.contains('\n') {
                                r.on_hover_text(&l.text);
                            }
                        }));
                    }
                });
        });

        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label(RichText::new(format!("{} of {} lines", shown.len(), self.lines.len())).size(12.0).color(theme::MUTED));
            let file = self.file.as_ref().map(|f| f.display().to_string()).unwrap_or_default();
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("Open logs folder").clicked() {
                    if let Some(dir) = logs_dir() {
                        widgets::open_path(&dir);
                    }
                }
                if ui.button("Copy shown lines").clicked() {
                    let text: Vec<String> = shown
                        .iter()
                        .rev()
                        .take(500)
                        .rev()
                        .map(|l| format!("{} {:?} {}", l.ms.map(localtime::clock).unwrap_or_default(), l.level, l.text))
                        .collect();
                    ui.output_mut(|o| o.copied_text = text.join("\n"));
                    env.toast("Copied the last 500 shown lines");
                }
                ui.add(egui::Label::new(RichText::new(file).monospace().size(11.0).color(theme::FAINT)).truncate(true));
            });
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plain_and_coloured_lines() {
        let l = parse("2026-11-08T18:04:50.162151Z  INFO History toggle: false -> true").unwrap();
        assert_eq!(l.level, Level::Info);
        assert_eq!(l.text, "History toggle: false -> true");
        assert!(l.ms.is_some());
        let l = parse("\u{1b}[2m2026-11-08T18:04:50.162151Z\u{1b}[0m \u{1b}[33m WARN\u{1b}[0m disk full").unwrap();
        assert_eq!((l.level, l.text.as_str()), (Level::Warn, "disk full"));
        assert!(routine("Windows message: 96 (hwnd: 0x1)"));
        assert!(parse("   ").is_none());
    }
}

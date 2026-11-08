//! Small building blocks shared by the tabs.

use super::theme;
use eframe::egui::{self, Color32, RichText};

/// One sidebar entry. `badge`: a count or short value on the right.
pub fn nav_item(ui: &mut egui::Ui, label: &str, badge: Option<&str>, selected: bool, hint: &str) -> egui::Response {
    let width = ui.available_width();
    let (rect, response) = ui.allocate_exact_size(egui::vec2(width, 32.0), egui::Sense::click());
    let fill = if selected {
        theme::BG_CARD_HI
    } else if response.hovered() {
        theme::BG_CARD
    } else {
        Color32::TRANSPARENT
    };
    let painter = ui.painter();
    painter.rect_filled(rect, 6.0, fill);
    if selected {
        let bar = egui::Rect::from_min_size(rect.min + egui::vec2(0.0, 7.0), egui::vec2(3.0, rect.height() - 14.0));
        painter.rect_filled(bar, 2.0, theme::ACCENT);
    }
    let font = if selected { theme::heading(14.0) } else { egui::FontId::proportional(14.0) };
    painter.text(rect.left_center() + egui::vec2(14.0, 0.0), egui::Align2::LEFT_CENTER, label, font, if selected { theme::TEXT } else { theme::MUTED });
    if let Some(b) = badge {
        painter.text(rect.right_center() - egui::vec2(10.0, 0.0), egui::Align2::RIGHT_CENTER, b, egui::FontId::monospace(11.5), theme::FAINT);
    }
    response.on_hover_text(hint)
}

/// Coloured dot and state name.
pub fn state_pill(ui: &mut egui::Ui, text: &str, color: Color32) {
    egui::Frame::none()
        .fill(Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 28))
        .stroke(egui::Stroke::new(1.0, Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 90)))
        .rounding(12.0)
        .inner_margin(egui::Margin::symmetric(10.0, 3.0))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                let (r, _) = ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
                ui.painter().circle_filled(r.center(), 4.0, color);
                ui.label(RichText::new(text).font(theme::heading(13.0)).color(color));
            });
        });
}

/// Microphone level as a short bar.
pub fn level_meter(ui: &mut egui::Ui, level: f32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(90.0, 6.0), egui::Sense::hover());
    let p = ui.painter();
    p.rect_filled(rect, 3.0, theme::BG_CARD_HI);
    let l = level.clamp(0.0, 1.0);
    if l > 0.0 {
        let fill = egui::Rect::from_min_size(rect.min, egui::vec2(rect.width() * l, rect.height()));
        let c = if l < 0.6 { theme::SUCCESS } else if l < 0.85 { theme::WARNING } else { theme::DANGER };
        p.rect_filled(fill, 3.0, c);
    }
}

/// "key value" in the status strip.
pub fn kv(ui: &mut egui::Ui, key: &str, value: &str) -> egui::Response {
    kv_colored(ui, key, value, theme::TEXT)
}

pub fn kv_colored(ui: &mut egui::Ui, key: &str, value: &str, color: Color32) -> egui::Response {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 5.0;
        ui.label(RichText::new(key).size(12.0).color(theme::FAINT));
        ui.label(RichText::new(value).size(12.5).color(color));
    })
    .response
}

/// m:ss for a duration in milliseconds.
pub fn duration(ms: u64) -> String {
    let s = (ms + 500) / 1000;
    format!("{}:{:02}", s / 60, s % 60)
}

/// Title and one line of explanation at the top of a tab.
pub fn page_header(ui: &mut egui::Ui, title: &str, subtitle: &str) {
    ui.label(RichText::new(title).font(theme::heading(20.0)).color(theme::TEXT));
    if !subtitle.is_empty() {
        ui.label(RichText::new(subtitle).color(theme::MUTED));
    }
    ui.add_space(10.0);
}

/// A section title inside a tab.
pub fn section(ui: &mut egui::Ui, title: &str) {
    ui.add_space(6.0);
    ui.label(RichText::new(title.to_uppercase()).font(theme::heading(11.5)).color(theme::FAINT));
    ui.add_space(2.0);
}

/// A bordered card.
pub fn card<R>(ui: &mut egui::Ui, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    egui::Frame::none()
        .fill(theme::BG_CARD)
        .stroke(egui::Stroke::new(1.0, theme::BORDER))
        .rounding(8.0)
        .inner_margin(egui::Margin::symmetric(14.0, 12.0))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            add(ui)
        })
        .inner
}

/// One settings row: label and help on the left, the control on the right.
pub fn setting_row(ui: &mut egui::Ui, label: &str, help: &str, control: impl FnOnce(&mut egui::Ui)) {
    ui.horizontal(|ui| {
        let left = (ui.available_width() - 260.0).max(220.0);
        ui.allocate_ui_with_layout(egui::vec2(left, 0.0), egui::Layout::top_down(egui::Align::Min), |ui| {
            ui.label(RichText::new(label).color(theme::TEXT));
            if !help.is_empty() {
                ui.add(egui::Label::new(RichText::new(help).size(12.0).color(theme::MUTED)).wrap(true));
            }
        });
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), control);
    });
    ui.add_space(4.0);
}

/// Small grey text.
pub fn hint(ui: &mut egui::Ui, text: &str) {
    ui.add(egui::Label::new(RichText::new(text).size(12.0).color(theme::MUTED)).wrap(true));
}

/// A monospace block with a Copy button. Returns true when copied.
pub fn copy_block(ui: &mut egui::Ui, title: &str, text: &str) -> bool {
    let mut copied = false;
    egui::Frame::none()
        .fill(theme::BG_CODE)
        .stroke(egui::Stroke::new(1.0, theme::BORDER))
        .rounding(6.0)
        .inner_margin(egui::Margin::symmetric(12.0, 9.0))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.label(RichText::new(title).size(12.0).color(theme::FAINT));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("Copy").clicked() {
                        ui.output_mut(|o| o.copied_text = text.to_string());
                        copied = true;
                    }
                });
            });
            ui.add(egui::Label::new(RichText::new(text).monospace().color(theme::TEXT)).wrap(true));
        });
    copied
}

/// A pill-shaped tag with an optional remove "×". Returns true when removed.
pub fn chip(ui: &mut egui::Ui, text: &str, removable: bool) -> bool {
    let mut removed = false;
    egui::Frame::none()
        .fill(theme::BG_CARD_HI)
        .stroke(egui::Stroke::new(1.0, theme::BORDER_HI))
        .rounding(10.0)
        .inner_margin(egui::Margin { left: 8.0, right: if removable { 4.0 } else { 8.0 }, top: 1.0, bottom: 1.0 })
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 2.0;
                ui.label(RichText::new(text).size(12.5).color(theme::TEXT));
                if removable {
                    let x = ui.add(egui::Button::new(RichText::new("×").size(13.0).color(theme::MUTED)).frame(false).small());
                    if x.on_hover_text("Remove").clicked() {
                        removed = true;
                    }
                }
            });
        });
    removed
}

/// Primary (accent) button.
pub fn primary(text: &str) -> egui::Button<'static> {
    egui::Button::new(RichText::new(text.to_string()).color(Color32::WHITE)).fill(Color32::from_rgb(60, 102, 214)).stroke(egui::Stroke::new(1.0, theme::ACCENT))
}

/// A button for a destructive action.
pub fn danger(text: &str) -> egui::Button<'static> {
    egui::Button::new(RichText::new(text.to_string()).color(theme::DANGER)).fill(theme::BG_CARD_HI)
}

/// A one-line banner (warning or error) across the tab.
pub fn banner(ui: &mut egui::Ui, text: &str, error: bool) {
    let (fg, bg) = if error { (theme::DANGER, theme::DANGER_BG) } else { (theme::WARNING, theme::WARNING_BG) };
    egui::Frame::none()
        .fill(bg)
        .stroke(egui::Stroke::new(1.0, fg.linear_multiply(0.5)))
        .rounding(6.0)
        .inner_margin(egui::Margin::symmetric(12.0, 8.0))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.add(egui::Label::new(RichText::new(text).color(fg)).wrap(true));
        });
    ui.add_space(8.0);
}

/// Open a file or folder with the system handler.
pub fn open_path(path: &std::path::Path) {
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        let _ = std::process::Command::new("explorer.exe").arg(path).creation_flags(0x08000000).spawn();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open").arg(path).spawn();
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let _ = std::process::Command::new("xdg-open").arg(path).spawn();
    }
}

/// Open a text file in an editor (Notepad on Windows: .toml has no default app).
pub fn open_text(path: &std::path::Path) {
    #[cfg(target_os = "windows")]
    {
        let _ = std::process::Command::new("notepad.exe").arg(path).spawn();
    }
    #[cfg(not(target_os = "windows"))]
    open_path(path);
}

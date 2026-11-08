//! Colours, fonts and widget style of the console window. Dark, dense,
//! one colour per state. Fonts come from the system (Segoe UI, Cascadia
//! Mono) so the binary carries none; egui's own fonts stay as fallbacks.

use eframe::egui::{self, Color32, FontFamily, FontId, TextStyle};

pub const BG: Color32 = Color32::from_rgb(18, 19, 22);
pub const BG_SIDE: Color32 = Color32::from_rgb(22, 23, 27);
pub const BG_STRIP: Color32 = Color32::from_rgb(22, 23, 27);
pub const BG_CARD: Color32 = Color32::from_rgb(27, 28, 33);
pub const BG_CARD_HI: Color32 = Color32::from_rgb(35, 37, 43);
pub const BG_CODE: Color32 = Color32::from_rgb(13, 14, 16);
pub const BORDER: Color32 = Color32::from_rgb(40, 42, 49);
pub const BORDER_HI: Color32 = Color32::from_rgb(58, 61, 70);
pub const TEXT: Color32 = Color32::from_rgb(231, 232, 234);
pub const MUTED: Color32 = Color32::from_rgb(154, 157, 166);
pub const FAINT: Color32 = Color32::from_rgb(107, 110, 119);
pub const ACCENT: Color32 = Color32::from_rgb(109, 156, 255);
pub const ACCENT_BG: Color32 = Color32::from_rgb(30, 40, 64);
pub const SUCCESS: Color32 = Color32::from_rgb(79, 209, 139);
pub const WARNING: Color32 = Color32::from_rgb(232, 182, 76);
pub const WARNING_BG: Color32 = Color32::from_rgb(58, 46, 22);
pub const DANGER: Color32 = Color32::from_rgb(240, 100, 106);
pub const DANGER_BG: Color32 = Color32::from_rgb(58, 29, 32);

/// The family for headings and labels that need weight.
pub fn semibold() -> FontFamily {
    FontFamily::Name("semibold".into())
}

pub fn heading(size: f32) -> FontId {
    FontId::new(size, semibold())
}

fn system_font(names: &[&str]) -> Option<Vec<u8>> {
    #[cfg(target_os = "windows")]
    {
        let dir = std::env::var_os("WINDIR").map(std::path::PathBuf::from).unwrap_or_else(|| "C:\\Windows".into()).join("Fonts");
        names.iter().find_map(|n| std::fs::read(dir.join(n)).ok())
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = names;
        None
    }
}

/// Fonts, text sizes and visuals.
pub fn install(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    let proportional = fonts.families.get(&FontFamily::Proportional).cloned().unwrap_or_default();
    let mut semibold = proportional.clone();
    if let Some(bytes) = system_font(&["segoeui.ttf"]) {
        fonts.font_data.insert("ui".into(), egui::FontData::from_owned(bytes));
        fonts.families.entry(FontFamily::Proportional).or_default().insert(0, "ui".into());
        semibold.insert(0, "ui".into());
    }
    if let Some(bytes) = system_font(&["seguisb.ttf"]) {
        fonts.font_data.insert("ui-semibold".into(), egui::FontData::from_owned(bytes));
        semibold.insert(0, "ui-semibold".into());
    }
    fonts.families.insert(semibold_family(), semibold);
    if let Some(bytes) = system_font(&["CascadiaMono.ttf", "consola.ttf"]) {
        fonts.font_data.insert("mono".into(), egui::FontData::from_owned(bytes));
        fonts.families.entry(FontFamily::Monospace).or_default().insert(0, "mono".into());
    }
    ctx.set_fonts(fonts);

    let mut style = (*ctx.style()).clone();
    style.text_styles = [
        (TextStyle::Heading, FontId::new(20.0, semibold_family())),
        (TextStyle::Body, FontId::new(14.0, FontFamily::Proportional)),
        (TextStyle::Button, FontId::new(13.5, FontFamily::Proportional)),
        (TextStyle::Small, FontId::new(12.0, FontFamily::Proportional)),
        (TextStyle::Monospace, FontId::new(12.5, FontFamily::Monospace)),
    ]
    .into();
    style.spacing.item_spacing = egui::vec2(8.0, 6.0);
    style.spacing.button_padding = egui::vec2(10.0, 4.0);
    style.spacing.interact_size.y = 26.0;
    style.spacing.scroll.bar_width = 8.0;

    let v = &mut style.visuals;
    *v = egui::Visuals::dark();
    v.panel_fill = BG;
    v.window_fill = BG_CARD;
    v.window_stroke = egui::Stroke::new(1.0, BORDER_HI);
    v.window_rounding = 8.0.into();
    v.menu_rounding = 6.0.into();
    v.extreme_bg_color = BG_CODE;
    v.faint_bg_color = BG_CARD;
    v.code_bg_color = BG_CODE;
    v.hyperlink_color = ACCENT;
    v.selection.bg_fill = ACCENT_BG;
    v.selection.stroke = egui::Stroke::new(1.0, ACCENT);
    v.warn_fg_color = WARNING;
    v.error_fg_color = DANGER;
    v.override_text_color = None;
    let w = &mut v.widgets;
    w.noninteractive.fg_stroke = egui::Stroke::new(1.0, TEXT);
    w.noninteractive.bg_stroke = egui::Stroke::new(1.0, BORDER);
    w.noninteractive.bg_fill = BG_CARD;
    for (state, fill, stroke) in [
        (&mut w.inactive, BG_CARD_HI, BORDER_HI),
        (&mut w.hovered, Color32::from_rgb(44, 47, 55), Color32::from_rgb(80, 84, 96)),
        (&mut w.active, Color32::from_rgb(50, 54, 64), ACCENT),
        (&mut w.open, BG_CARD_HI, BORDER_HI),
    ] {
        state.bg_fill = fill;
        state.weak_bg_fill = fill;
        state.bg_stroke = egui::Stroke::new(1.0, stroke);
        state.rounding = 6.0.into();
        state.fg_stroke = egui::Stroke::new(1.0, TEXT);
    }
    w.inactive.fg_stroke = egui::Stroke::new(1.0, Color32::from_rgb(214, 216, 220));
    ctx.set_style(style);
}

fn semibold_family() -> FontFamily {
    semibold()
}

/// The app icon for the title bar and taskbar.
pub fn window_icon() -> egui::IconData {
    const PNG: &[u8] = include_bytes!("../../assets/icon_64.png");
    match image::load_from_memory(PNG) {
        Ok(img) => {
            let rgba = img.to_rgba8();
            let (width, height) = rgba.dimensions();
            egui::IconData { rgba: rgba.into_raw(), width, height }
        }
        Err(_) => egui::IconData::default(),
    }
}

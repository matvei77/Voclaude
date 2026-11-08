//! The Voclaude window: `voclaude console`.
//!
//! It runs as its own process, started by the tray app with `--ipc` (status
//! on stdin, commands on stdout; see `ipc`). Closing the window ends the
//! process, so the tray app stays small. Started by hand without `--ipc`, it
//! asks a running tray app to open its window, or runs on its own with the
//! controls that need the tray app disabled.
//!
//! Everything except live status comes from files: history.json, config.toml,
//! data/unsure-words.jsonl and data/logs. Settings are written to config.toml
//! (`config_edit`), and the tray app applies them within a few seconds.

pub mod ipc;

mod api_tab;
mod dictations;
mod localtime;
mod log_tab;
mod model;
mod names;
mod settings;
mod theme;
mod widgets;

use crate::config::Config;
use eframe::egui;
use ipc::{FromConsole, Status, ToConsole};
use std::io::{BufRead, Write};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
    Dictations,
    Names,
    Model,
    Settings,
    Api,
    Log,
}

const TABS: [(Tab, &str); 6] = [
    (Tab::Dictations, "Dictations"),
    (Tab::Names, "Names"),
    (Tab::Model, "Model"),
    (Tab::Settings, "Settings"),
    (Tab::Api, "API"),
    (Tab::Log, "Log"),
];

/// Live state from the tray app, filled by the stdin reader thread.
#[derive(Default)]
struct Live {
    status: Option<Status>,
    level: f32,
    level_at: Option<Instant>,
    history_changed: bool,
    focus: bool,
    /// The tray app closed the pipe (it quit).
    gone: bool,
}

/// What every tab may use.
pub(crate) struct Env<'a> {
    pub config: &'a Config,
    /// Path of config.toml, for "Open config.toml".
    pub config_path: Option<std::path::PathBuf>,
    /// None: the tray app is not connected.
    pub status: Option<&'a Status>,
    /// config.toml could not be loaded (the shown values are the last good ones).
    pub config_error: Option<&'a str>,
    pub recording: bool,
    commands: &'a dyn Fn(FromConsole) -> bool,
    toast: &'a mut Option<Toast>,
}

impl Env<'_> {
    /// Send a command to the tray app. False when it is not connected.
    pub fn send(&self, cmd: FromConsole) -> bool {
        (self.commands)(cmd)
    }
    pub fn connected(&self) -> bool {
        self.status.is_some()
    }
    pub fn toast(&mut self, text: impl Into<String>) {
        *self.toast = Some(Toast { text: text.into(), error: false, at: Instant::now() });
    }
    pub fn toast_error(&mut self, text: impl Into<String>) {
        *self.toast = Some(Toast { text: text.into(), error: true, at: Instant::now() });
    }
}

pub(crate) struct Toast {
    text: String,
    error: bool,
    at: Instant,
}

/// Entry point for `voclaude console [--ipc]`. Returns the exit code.
pub fn run(args: &[String]) -> i32 {
    let ipc = args.iter().any(|a| a == "--ipc");
    if !ipc && crate::instance::is_running() {
        // The tray app opens its own window (with live status) and keeps
        // one window per profile.
        return if crate::instance::signal_open() { 0 } else { 1 };
    }
    let live = Arc::new(Mutex::new(Live::default()));
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Voclaude")
            .with_app_id("voclaude")
            .with_inner_size([1000.0, 700.0])
            .with_min_inner_size([720.0, 480.0])
            .with_icon(theme::window_icon()),
        centered: true,
        ..Default::default()
    };
    let live_for_app = live.clone();
    let result = eframe::run_native(
        "Voclaude",
        options,
        Box::new(move |cc| {
            theme::install(&cc.egui_ctx);
            if ipc {
                spawn_reader(live_for_app.clone(), cc.egui_ctx.clone());
            }
            Box::new(ConsoleApp::new(live_for_app, ipc))
        }),
    );
    match result {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("voclaude console: {}", e);
            1
        }
    }
}

/// Read core -> console messages from stdin until the tray app goes away.
fn spawn_reader(live: Arc<Mutex<Live>>, ctx: egui::Context) {
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            let Ok(line) = line else { break };
            let Ok(msg) = serde_json::from_str::<ToConsole>(&line) else { continue };
            if matches!(msg, ToConsole::Focus) {
                // eframe drops repaints while the window is minimized, so
                // update() may not run: raise it from this thread. Not under
                // the lock: raising sends messages to the UI thread, which may
                // be waiting for the lock in update().
                raise_own_window();
            }
            let mut l = live.lock().unwrap_or_else(|e| e.into_inner());
            match msg {
                ToConsole::Status(s) => l.status = Some(s),
                ToConsole::Level { level } => {
                    l.level = level;
                    l.level_at = Some(Instant::now());
                }
                ToConsole::HistoryChanged => l.history_changed = true,
                ToConsole::Focus => l.focus = true,
            }
            drop(l);
            ctx.request_repaint();
        }
        live.lock().unwrap_or_else(|e| e.into_inner()).gone = true;
        ctx.request_repaint();
    });
}

/// This process's app window: visible, titled, not owned (winit also keeps an
/// untitled message window).
#[cfg(target_os = "windows")]
fn own_window() -> windows_sys::Win32::Foundation::HWND {
    use windows_sys::Win32::Foundation::{BOOL, HWND, LPARAM};
    use windows_sys::Win32::UI::WindowsAndMessaging::{EnumWindows, GetWindow, GetWindowTextLengthW, GetWindowThreadProcessId, IsWindowVisible, GW_OWNER};
    unsafe extern "system" fn each(hwnd: HWND, found: LPARAM) -> BOOL {
        let mut pid = 0u32;
        GetWindowThreadProcessId(hwnd, &mut pid);
        if pid == std::process::id() && IsWindowVisible(hwnd) != 0 && GetWindowTextLengthW(hwnd) > 0 && GetWindow(hwnd, GW_OWNER).is_null() {
            *(found as *mut HWND) = hwnd;
            return 0;
        }
        1
    }
    let mut found: HWND = std::ptr::null_mut();
    unsafe { EnumWindows(Some(each), &mut found as *mut HWND as LPARAM) };
    found
}

/// Restore the window if minimized and bring it to the front, from any
/// thread. The tray app has passed its foreground right to this process.
#[cfg(target_os = "windows")]
fn raise_own_window() {
    use windows_sys::Win32::System::Threading::{AttachThreadInput, GetCurrentThreadId};
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        BringWindowToTop, GetForegroundWindow, GetWindowThreadProcessId, IsIconic, SetForegroundWindow, ShowWindow, SW_RESTORE,
    };
    unsafe {
        // AttachThreadInput needs a message queue on this thread; any
        // PeekMessage call creates one.
        let mut msg: windows_sys::Win32::UI::WindowsAndMessaging::MSG = std::mem::zeroed();
        windows_sys::Win32::UI::WindowsAndMessaging::PeekMessageW(&mut msg, std::ptr::null_mut(), 0, 0, 0);
        let found = own_window();
        if !found.is_null() {
            if IsIconic(found) != 0 {
                ShowWindow(found, SW_RESTORE);
            }
            // Windows only lets the foreground thread hand over the focus:
            // share its input state for the call, as other launchers do.
            let fg = GetForegroundWindow();
            let fg_thread = if fg.is_null() { 0 } else { GetWindowThreadProcessId(fg, std::ptr::null_mut()) };
            let this_thread = GetCurrentThreadId();
            let attached = fg_thread != 0 && fg_thread != this_thread && AttachThreadInput(this_thread, fg_thread, 1) != 0;
            BringWindowToTop(found);
            SetForegroundWindow(found);
            if attached {
                AttachThreadInput(this_thread, fg_thread, 0);
            }
        }
    }
}

#[cfg(not(target_os = "windows"))]
fn raise_own_window() {}

/// Write one console -> core command on stdout.
fn send_command(cmd: &FromConsole) -> bool {
    let Ok(mut line) = serde_json::to_string(cmd) else { return false };
    line.push('\n');
    let mut out = std::io::stdout().lock();
    out.write_all(line.as_bytes()).and_then(|_| out.flush()).is_ok()
}

struct ConsoleApp {
    live: Arc<Mutex<Live>>,
    ipc: bool,
    tab: Tab,
    config: Config,
    config_error: Option<String>,
    config_mtime: Option<SystemTime>,
    config_checked: Instant,
    toast: Option<Toast>,
    frames: u64,
    dictations: dictations::State,
    names: names::State,
    model: model::State,
    settings: settings::State,
    api: api_tab::State,
    log: log_tab::State,
}

impl ConsoleApp {
    fn new(live: Arc<Mutex<Live>>, ipc: bool) -> Self {
        let mut app = Self {
            live,
            ipc,
            tab: Tab::Dictations,
            config: Config::default(),
            config_error: None,
            config_mtime: None,
            config_checked: Instant::now() - Duration::from_secs(10),
            toast: None,
            frames: 0,
            dictations: dictations::State::default(),
            names: names::State::default(),
            model: model::State::default(),
            settings: settings::State::default(),
            api: api_tab::State::default(),
            log: log_tab::State::default(),
        };
        app.reload_config(true);
        app
    }

    /// Re-read config.toml when it changed on disk (checked once a second).
    fn reload_config(&mut self, force: bool) {
        if !force && self.config_checked.elapsed() < Duration::from_secs(1) {
            return;
        }
        self.config_checked = Instant::now();
        let path = Config::config_path();
        let mtime = path.as_ref().and_then(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok());
        if !force && mtime == self.config_mtime {
            return;
        }
        self.config_mtime = mtime;
        match Config::load() {
            Ok(c) => {
                self.config = c;
                self.config_error = None;
            }
            Err(e) => self.config_error = Some(e.to_string()),
        }
    }
}

impl eframe::App for ConsoleApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.frames += 1;
        if self.frames == 3 && self.ipc {
            // Opened by a hotkey or tray click: come to the front.
            raise_own_window();
        }
        self.reload_config(false);

        // Live state from the tray app.
        let (status, level, history_changed, focus, gone) = {
            let mut l = self.live.lock().unwrap_or_else(|e| e.into_inner());
            let level = match l.level_at {
                Some(at) if at.elapsed() < Duration::from_millis(400) => l.level,
                _ => 0.0,
            };
            let out = (l.status.clone(), level, l.history_changed, l.focus, l.gone);
            l.history_changed = false;
            l.focus = false;
            out
        };
        if gone {
            // The tray app quit: close with it.
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        if focus && !cfg!(target_os = "windows") {
            ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(false));
            ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
        }
        if history_changed {
            self.dictations.mark_stale();
        }
        let status = if self.ipc { status } else { None };
        let recording = status.as_ref().is_some_and(|s| s.state == "Recording");

        // Keyboard: Ctrl+1..6 switch tabs.
        ctx.input(|i| {
            if i.modifiers.command {
                for (n, key) in [egui::Key::Num1, egui::Key::Num2, egui::Key::Num3, egui::Key::Num4, egui::Key::Num5, egui::Key::Num6]
                    .iter()
                    .enumerate()
                {
                    if i.key_pressed(*key) {
                        self.tab = TABS[n].0;
                    }
                }
            }
        });

        let ipc = self.ipc;
        let commands = move |cmd: FromConsole| ipc && send_command(&cmd);
        let mut env = Env {
            config: &self.config,
            config_path: Config::config_path(),
            status: status.as_ref(),
            config_error: self.config_error.as_deref(),
            recording,
            commands: &commands,
            toast: &mut self.toast,
        };

        egui::TopBottomPanel::top("strip")
            .exact_height(48.0)
            .frame(egui::Frame::none().fill(theme::BG_STRIP).inner_margin(egui::Margin::symmetric(16.0, 0.0)).stroke(egui::Stroke::new(1.0, theme::BORDER)))
            .show(ctx, |ui| status_strip(ui, &mut env, level));

        egui::SidePanel::left("tabs")
            .exact_width(176.0)
            .resizable(false)
            .frame(egui::Frame::none().fill(theme::BG_SIDE).inner_margin(egui::Margin { left: 10.0, right: 10.0, top: 14.0, bottom: 12.0 }).stroke(egui::Stroke::new(1.0, theme::BORDER)))
            .show(ctx, |ui| {
                let badges = [
                    Some(self.dictations.count_label()),
                    self.names.badge(env.config),
                    None,
                    None,
                    env.status.and_then(|s| s.api_port).map(|p| format!(":{}", p)),
                    None,
                ];
                for (n, (tab, label)) in TABS.iter().enumerate() {
                    if widgets::nav_item(ui, label, badges[n].as_deref(), self.tab == *tab, &format!("Ctrl+{}", n + 1)).clicked() {
                        self.tab = *tab;
                    }
                    ui.add_space(2.0);
                }
                ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
                    ui.label(egui::RichText::new(format!("v{}", env!("CARGO_PKG_VERSION"))).size(11.0).color(theme::FAINT));
                    ui.add_space(2.0);
                    let hotkeys = format!("{}  record\n{}  this window", env.config.hotkey, env.config.history_hotkey);
                    ui.label(egui::RichText::new(hotkeys).size(11.0).color(theme::MUTED));
                });
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(theme::BG).inner_margin(egui::Margin { left: 22.0, right: 22.0, top: 18.0, bottom: 12.0 }))
            .show(ctx, |ui| match self.tab {
                Tab::Dictations => self.dictations.show(ui, &mut env),
                Tab::Names => self.names.show(ui, &mut env),
                Tab::Model => self.model.show(ui, &mut env),
                Tab::Settings => self.settings.show(ui, &mut env),
                Tab::Api => self.api.show(ui, &mut env),
                Tab::Log => self.log.show(ui, &mut env),
            });

        // Toast, bottom right, for 3 s.
        if let Some(t) = &self.toast {
            if t.at.elapsed() > Duration::from_secs(3) {
                self.toast = None;
            } else {
                egui::Area::new(egui::Id::new("toast"))
                    .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-18.0, -16.0))
                    .order(egui::Order::Foreground)
                    .show(ctx, |ui| {
                        egui::Frame::none()
                            .fill(theme::BG_CARD_HI)
                            .stroke(egui::Stroke::new(1.0, if t.error { theme::DANGER } else { theme::BORDER_HI }))
                            .rounding(6.0)
                            .inner_margin(egui::Margin::symmetric(12.0, 8.0))
                            .show(ui, |ui| {
                                ui.label(egui::RichText::new(&t.text).color(if t.error { theme::DANGER } else { theme::TEXT }));
                            });
                    });
                ctx.request_repaint_after(Duration::from_millis(250));
            }
        }

        // Keep the clock and the mic level moving; idle otherwise.
        ctx.request_repaint_after(if recording { Duration::from_millis(50) } else { Duration::from_secs(1) });
    }
}

/// The strip across the top: what the app is doing right now.
fn status_strip(ui: &mut egui::Ui, env: &mut Env, level: f32) {
    let config_error = env.config_error;
    ui.horizontal_centered(|ui| {
        ui.spacing_mut().item_spacing.x = 14.0;
        let Some(s) = env.status else {
            widgets::state_pill(ui, "Not running", theme::FAINT);
            ui.label(egui::RichText::new("Start Voclaude to record. Settings and history still work here.").color(theme::MUTED));
            return;
        };
        let (label, color) = match s.state.as_str() {
            "Recording" => ("Recording", theme::DANGER),
            "Transcribing" => ("Transcribing", theme::WARNING),
            "Recovering" => ("Recovering", theme::WARNING),
            _ => ("Idle", theme::SUCCESS),
        };
        widgets::state_pill(ui, label, color);
        if s.state == "Recording" {
            widgets::level_meter(ui, level);
        }
        let (name, device) = model::split_label(&s.model, s.use_gpu);
        widgets::kv(ui, "model", &name).on_hover_text(if s.model_loaded { "Loaded" } else { "Loads at the next recording" });
        widgets::kv(ui, "device", &device);
        if let (Some(ms), Some(speed)) = (s.last_duration_ms, s.last_speed) {
            let proc = ms as f32 / 1000.0 / speed.max(0.01);
            widgets::kv(ui, "last", &format!("{} audio → {:.1} s", widgets::duration(ms), proc));
        }
        match (s.api_port, &s.api_error) {
            (Some(p), _) => {
                widgets::kv(ui, "API", &format!(":{}", p));
            }
            (None, Some(_)) => {
                widgets::kv_colored(ui, "API", "error", theme::DANGER);
            }
            _ => {}
        }

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let recording = s.state == "Recording";
            let busy = s.state == "Transcribing" || s.state == "Recovering";
            let text = if recording { format!("■  Stop  ({})", s.record_hotkey) } else { format!("●  Record  ({})", s.record_hotkey) };
            let button = egui::Button::new(egui::RichText::new(text).color(if recording { theme::TEXT } else { theme::DANGER }))
                .fill(if recording { theme::DANGER_BG } else { theme::BG_CARD })
                .stroke(egui::Stroke::new(1.0, if recording { theme::DANGER } else { theme::BORDER_HI }))
                .rounding(6.0)
                .min_size(egui::vec2(0.0, 30.0));
            if ui.add_enabled(!busy, button).clicked() {
                env.send(FromConsole::ToggleRecording);
            }
            if let Some(err) = config_error.or(s.config_error.as_deref()) {
                ui.label(egui::RichText::new("config.toml has an error (see Settings)").size(12.5).color(theme::DANGER)).on_hover_text(err);
            } else if let Some(msg) = &s.last_message {
                let color = if s.last_message_is_error { theme::DANGER } else { theme::MUTED };
                ui.add(egui::Label::new(egui::RichText::new(msg).color(color).size(12.5)).truncate(true));
            }
        });
    });
}

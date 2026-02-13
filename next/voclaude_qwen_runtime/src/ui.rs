//! History window, HUD overlay, and log buffer.
//!
//! The main window serves dual purpose: as a History window (normal size, decorated)
//! and as a HUD overlay (small, undecorated, positioned at top of screen). A keepalive
//! thread ensures update() runs regularly even when the window is off-screen.

use crossbeam_channel::{unbounded, Receiver, Sender};
use eframe::egui;
use std::collections::VecDeque;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{error, info};
use tracing_subscriber::fmt::MakeWriter;

#[cfg(target_os = "windows")]
use winit::platform::windows::EventLoopBuilderExtWindows;

// ---------------------------------------------------------------------------
// LogBuffer
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct LogBuffer {
    inner: Arc<Mutex<LogBufferInner>>,
}

struct LogBufferInner {
    lines: VecDeque<String>,
    limit: usize,
}

impl LogBuffer {
    pub fn new(limit: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(LogBufferInner {
                lines: VecDeque::new(),
                limit,
            })),
        }
    }

    pub fn make_writer(&self) -> LogBufferMakeWriter {
        LogBufferMakeWriter {
            buffer: self.clone(),
        }
    }

    pub fn snapshot(&self) -> Vec<String> {
        let inner = self.inner.lock().ok();
        inner
            .map(|buffer| buffer.lines.iter().cloned().collect())
            .unwrap_or_default()
    }

    fn push_chunk(&self, chunk: &str) {
        let mut inner = match self.inner.lock() {
            Ok(inner) => inner,
            Err(_) => return,
        };

        for line in chunk.lines() {
            let trimmed = line.trim_end();
            if trimmed.is_empty() {
                continue;
            }
            inner.lines.push_back(trimmed.to_string());
            if inner.lines.len() > inner.limit {
                inner.lines.pop_front();
            }
        }
    }
}

pub struct LogBufferMakeWriter {
    buffer: LogBuffer,
}

pub struct LogBufferWriter {
    buffer: LogBuffer,
    stdout: io::Stdout,
}

impl<'a> MakeWriter<'a> for LogBufferMakeWriter {
    type Writer = LogBufferWriter;

    fn make_writer(&'a self) -> Self::Writer {
        LogBufferWriter {
            buffer: self.buffer.clone(),
            stdout: io::stdout(),
        }
    }
}

impl Write for LogBufferWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let written = self.stdout.write(buf)?;
        let chunk = String::from_utf8_lossy(&buf[..written]);
        self.buffer.push_chunk(&chunk);
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.stdout.flush()
    }
}

// ---------------------------------------------------------------------------
// Unified command enum
// ---------------------------------------------------------------------------

pub(crate) enum UiCommand {
    Toggle,
    Show,
    Hide,
    AddHistory(String),
    SetStatus(UiStatus),
    HudSetState(HudState),
    HudSetAccel(bool),
    HudSetLevel(f32),
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct UiStatus {
    pub state: String,
    #[allow(dead_code)]
    pub hotkey: String,
    pub use_gpu: bool,
    pub model: String,
    pub model_size_mb: Option<u64>,
    pub history_count: usize,
    pub input_device: Option<String>,
    pub input_level: Option<f32>,
    pub last_duration_ms: Option<u64>,
    pub last_speed: Option<f32>,
    pub last_message: Option<String>,
}

impl UiStatus {
    pub fn new(hotkey: String, use_gpu: bool, model: String, model_size_mb: Option<u64>) -> Self {
        Self {
            state: "Idle".to_string(),
            hotkey,
            use_gpu,
            model,
            model_size_mb,
            history_count: 0,
            input_device: None,
            input_level: None,
            last_duration_ms: None,
            last_speed: None,
            last_message: None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum HudState {
    Idle,
    Recording,
    Transcribing { message: String, percent: Option<u8> },
    Ready { message: String },
}

// ---------------------------------------------------------------------------
// UiManager
// ---------------------------------------------------------------------------

pub struct UiManager {
    command_tx: Sender<UiCommand>,
    alive: Arc<AtomicBool>,
    repaint_ctx: Arc<Mutex<Option<egui::Context>>>,
}

impl UiManager {
    pub fn new(log_buffer: LogBuffer, gpu_enabled: bool) -> Result<Self, Box<dyn std::error::Error>> {
        let (command_tx, command_rx) = unbounded();
        let app = CombinedApp::new(command_rx, log_buffer, gpu_enabled);
        let alive = Arc::new(AtomicBool::new(true));
        let alive_thread = alive.clone();
        let repaint_ctx: Arc<Mutex<Option<egui::Context>>> = Arc::new(Mutex::new(None));
        let repaint_ctx_cc = repaint_ctx.clone();
        let alive_keepalive = alive.clone();

        std::thread::spawn(move || {
            let options = eframe::NativeOptions {
                viewport: egui::ViewportBuilder::default()
                    .with_title("Voclaude")
                    .with_inner_size([340.0, 72.0])
                    .with_visible(true)
                    .with_position(egui::pos2(-32000.0, -32000.0))
                    .with_decorations(false)
                    .with_always_on_top()
                    .with_taskbar(false),
                #[cfg(target_os = "windows")]
                event_loop_builder: Some(Box::new(|builder| {
                    builder.with_any_thread(true);
                })),
                ..Default::default()
            };

            if let Err(err) = eframe::run_native(
                "Voclaude",
                options,
                Box::new(move |cc| {
                    if let Ok(mut guard) = repaint_ctx_cc.lock() {
                        *guard = Some(cc.egui_ctx.clone());
                    }
                    let keepalive_ctx = cc.egui_ctx.clone();
                    std::thread::spawn(move || {
                        loop {
                            std::thread::sleep(Duration::from_millis(100));
                            if !alive_keepalive.load(Ordering::Relaxed) {
                                break;
                            }
                            keepalive_ctx.request_repaint();
                        }
                    });
                    Box::new(app)
                }),
            ) {
                error!("Failed to start UI: {}", err);
            }
            alive_thread.store(false, Ordering::Relaxed);
        });

        info!("UI thread started");
        let _ = command_tx.send(UiCommand::Hide);
        Ok(Self { command_tx, alive, repaint_ctx })
    }

    fn wake(&self) {
        if let Ok(guard) = self.repaint_ctx.lock() {
            if let Some(ctx) = guard.as_ref() {
                ctx.request_repaint();
            }
        }
    }

    pub fn toggle(&self) -> bool {
        if !self.alive.load(Ordering::Relaxed) {
            error!("UI is not running");
            return false;
        }
        if let Err(err) = self.command_tx.send(UiCommand::Toggle) {
            error!("Failed to send toggle command: {}", err);
            return false;
        }
        self.wake();
        true
    }

    pub fn show(&self) -> bool {
        if !self.alive.load(Ordering::Relaxed) {
            error!("UI is not running");
            return false;
        }
        if let Err(err) = self.command_tx.send(UiCommand::Show) {
            error!("Failed to send show command: {}", err);
            return false;
        }
        self.wake();
        true
    }

    pub fn push_history(&self, text: String) {
        if let Err(err) = self.command_tx.send(UiCommand::AddHistory(text)) {
            error!("Failed to send history entry: {}", err);
        }
        self.wake();
    }

    pub fn set_status(&self, status: UiStatus) {
        if let Err(err) = self.command_tx.send(UiCommand::SetStatus(status)) {
            error!("Failed to send status update: {}", err);
        }
        self.wake();
    }

    pub fn command_sender(&self) -> Sender<UiCommand> {
        self.command_tx.clone()
    }

    pub fn repaint_signal(&self) -> Arc<Mutex<Option<egui::Context>>> {
        self.repaint_ctx.clone()
    }
}

// ---------------------------------------------------------------------------
// HudManager
// ---------------------------------------------------------------------------

pub struct HudManager {
    command_tx: Sender<UiCommand>,
    repaint_ctx: Arc<Mutex<Option<egui::Context>>>,
}

impl HudManager {
    pub fn new(command_tx: Sender<UiCommand>, repaint_ctx: Arc<Mutex<Option<egui::Context>>>) -> Self {
        Self { command_tx, repaint_ctx }
    }

    fn wake(&self) {
        if let Ok(guard) = self.repaint_ctx.lock() {
            if let Some(ctx) = guard.as_ref() {
                ctx.request_repaint();
            }
        }
    }

    pub fn set_state(&self, state: HudState) -> bool {
        if let Err(err) = self.command_tx.send(UiCommand::HudSetState(state)) {
            error!("Failed to send HUD state: {}", err);
            return false;
        }
        self.wake();
        true
    }

    pub fn set_accel(&self, gpu_enabled: bool) -> bool {
        if let Err(err) = self.command_tx.send(UiCommand::HudSetAccel(gpu_enabled)) {
            error!("Failed to send HUD accel state: {}", err);
            return false;
        }
        self.wake();
        true
    }

    pub fn set_level(&self, level: f32) -> bool {
        if let Err(err) = self.command_tx.send(UiCommand::HudSetLevel(level)) {
            error!("Failed to send HUD level: {}", err);
            return false;
        }
        self.wake();
        true
    }
}

// ---------------------------------------------------------------------------
// Window modes
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WindowMode {
    Hidden,
    Hud,
    History,
}

// ---------------------------------------------------------------------------
// Combined app
// ---------------------------------------------------------------------------

struct CombinedApp {
    command_rx: Receiver<UiCommand>,

    // History state
    history: VecDeque<String>,
    log_buffer: LogBuffer,
    history_visible: bool,
    filter: String,
    show_logs: bool,
    status: UiStatus,

    // HUD state
    hud_state: HudState,
    hud_visible: bool,
    hud_recording_started_at: Option<Instant>,
    hud_ready_until: Option<Instant>,
    hud_gpu_enabled: bool,
    hud_input_level: f32,

    // Window management
    window_mode: WindowMode,
    theme_applied: bool,
}

impl CombinedApp {
    fn new(command_rx: Receiver<UiCommand>, log_buffer: LogBuffer, gpu_enabled: bool) -> Self {
        Self {
            command_rx,

            history: VecDeque::new(),
            log_buffer,
            history_visible: false,
            filter: String::new(),
            show_logs: false,
            status: UiStatus::new(String::new(), false, "qwen3-asr".to_string(), None),

            hud_state: HudState::Idle,
            hud_visible: false,
            hud_recording_started_at: None,
            hud_ready_until: None,
            hud_gpu_enabled: gpu_enabled,
            hud_input_level: 0.0,

            window_mode: WindowMode::Hidden,
            theme_applied: false,
        }
    }

    fn apply_commands(&mut self) {
        while let Ok(cmd) = self.command_rx.try_recv() {
            match cmd {
                UiCommand::Toggle => {
                    self.history_visible = !self.history_visible;
                }
                UiCommand::Show => {
                    self.history_visible = true;
                }
                UiCommand::Hide => {
                    self.history_visible = false;
                }
                UiCommand::AddHistory(entry) => {
                    if !entry.trim().is_empty() {
                        self.history.push_front(entry);
                        if self.history.len() > 500 {
                            self.history.pop_back();
                        }
                    }
                }
                UiCommand::SetStatus(status) => {
                    self.status = status;
                }
                UiCommand::HudSetState(state) => {
                    self.hud_state = state;
                    match &self.hud_state {
                        HudState::Idle => {
                            self.hud_visible = false;
                            self.hud_recording_started_at = None;
                            self.hud_ready_until = None;
                        }
                        HudState::Recording => {
                            if self.hud_recording_started_at.is_none() {
                                self.hud_recording_started_at = Some(Instant::now());
                            }
                            self.hud_visible = true;
                            self.hud_ready_until = None;
                        }
                        HudState::Transcribing { .. } => {
                            self.hud_visible = true;
                            self.hud_recording_started_at = None;
                            self.hud_ready_until = None;
                        }
                        HudState::Ready { .. } => {
                            self.hud_visible = true;
                            self.hud_recording_started_at = None;
                            self.hud_ready_until = Some(Instant::now() + Duration::from_secs(3));
                        }
                    }
                }
                UiCommand::HudSetAccel(gpu_enabled) => {
                    self.hud_gpu_enabled = gpu_enabled;
                }
                UiCommand::HudSetLevel(level) => {
                    self.hud_input_level = level.clamp(0.0, 1.0);
                }
            }
        }
    }

    fn tick_hud_timers(&mut self) {
        if let Some(ready_until) = self.hud_ready_until {
            if Instant::now() >= ready_until {
                self.hud_state = HudState::Idle;
                self.hud_visible = false;
                self.hud_ready_until = None;
            }
        }
    }

    fn transition_window(&self, ctx: &egui::Context, target: WindowMode) {
        info!("[UI] Window mode: {:?} -> {:?}", self.window_mode, target);
        match target {
            WindowMode::Hidden => {
                ctx.send_viewport_cmd(egui::ViewportCommand::Decorations(false));
                ctx.send_viewport_cmd(egui::ViewportCommand::OuterPosition(
                    egui::pos2(-32000.0, -32000.0),
                ));
            }
            WindowMode::Hud => {
                let monitor = get_primary_screen_size();
                let hud_w = 340.0_f32;
                let hud_h = 72.0_f32;
                let x = monitor.x - hud_w - 20.0;
                let y = 24.0;

                ctx.send_viewport_cmd(egui::ViewportCommand::Title("Voclaude".to_string()));
                ctx.send_viewport_cmd(egui::ViewportCommand::Decorations(false));
                ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(hud_w, hud_h)));
                ctx.send_viewport_cmd(egui::ViewportCommand::OuterPosition(egui::pos2(x, y)));
                ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
            }
            WindowMode::History => {
                ctx.send_viewport_cmd(egui::ViewportCommand::Title(
                    "Voclaude History".to_string(),
                ));
                ctx.send_viewport_cmd(egui::ViewportCommand::Decorations(true));
                ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(560.0, 520.0)));
                if let Some(cmd) = egui::ViewportCommand::center_on_screen(ctx) {
                    ctx.send_viewport_cmd(cmd);
                }
                ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
            }
        }
    }

    fn render_hud_panel(&self, ctx: &egui::Context) {
        let bg = egui::Color32::from_rgb(18, 18, 28);

        egui::CentralPanel::default()
            .frame(
                egui::Frame::none()
                    .fill(bg)
                    .inner_margin(egui::Margin::same(10.0)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    match &self.hud_state {
                        HudState::Recording => {
                            // Pulsing red dot
                            let (dot_rect, _) = ui.allocate_exact_size(
                                egui::vec2(14.0, 14.0),
                                egui::Sense::hover(),
                            );
                            let t = ui.input(|i| i.time);
                            let pulse = (t * 3.0).sin() as f32 * 0.3 + 0.7;
                            ui.painter().circle_filled(
                                dot_rect.center(),
                                5.0,
                                egui::Color32::from_rgb(
                                    (255.0 * pulse) as u8,
                                    50,
                                    50,
                                ),
                            );

                            let elapsed = self
                                .hud_recording_started_at
                                .map(|s| s.elapsed())
                                .unwrap_or_default();
                            ui.label(
                                egui::RichText::new(format!(
                                    "Recording  {}",
                                    format_duration(elapsed)
                                ))
                                .size(14.0)
                                .color(egui::Color32::from_rgb(230, 230, 230)),
                            );
                        }
                        HudState::Transcribing { message, percent } => {
                            ui.add(egui::Spinner::new().size(14.0));
                            let text = match percent {
                                Some(p) => format!("{} ({}%)", message, p),
                                None => message.clone(),
                            };
                            ui.label(
                                egui::RichText::new(text)
                                    .size(14.0)
                                    .color(egui::Color32::from_rgb(180, 180, 230)),
                            );
                        }
                        HudState::Ready { message } => {
                            let (dot_rect, _) = ui.allocate_exact_size(
                                egui::vec2(14.0, 14.0),
                                egui::Sense::hover(),
                            );
                            ui.painter().circle_filled(
                                dot_rect.center(),
                                5.0,
                                egui::Color32::from_rgb(78, 204, 163),
                            );
                            ui.label(
                                egui::RichText::new(message)
                                    .size(14.0)
                                    .color(egui::Color32::from_rgb(180, 230, 200)),
                            );
                        }
                        HudState::Idle => {}
                    }

                    // GPU/CPU badge on right
                    ui.with_layout(
                        egui::Layout::right_to_left(egui::Align::Center),
                        |ui| {
                            let (text, color) = if self.hud_gpu_enabled {
                                ("GPU", egui::Color32::from_rgb(78, 204, 163))
                            } else {
                                ("CPU", egui::Color32::from_rgb(130, 170, 230))
                            };
                            let badge_bg = egui::Color32::from_rgba_unmultiplied(
                                color.r() / 4,
                                color.g() / 4,
                                color.b() / 4,
                                180,
                            );
                            egui::Frame::none()
                                .fill(badge_bg)
                                .rounding(egui::Rounding::same(4.0))
                                .inner_margin(egui::Margin {
                                    left: 6.0,
                                    right: 6.0,
                                    top: 2.0,
                                    bottom: 2.0,
                                })
                                .show(ui, |ui| {
                                    ui.label(
                                        egui::RichText::new(text)
                                            .size(11.0)
                                            .color(color)
                                            .strong(),
                                    );
                                });
                        },
                    );
                });

                if matches!(self.hud_state, HudState::Recording) {
                    ui.add_space(6.0);
                    draw_level_bar(ui, self.hud_input_level, ui.available_width(), 6.0);
                }
            });

        ctx.request_repaint_after(Duration::from_millis(100));
    }

    fn render_history(&mut self, ctx: &egui::Context) {
        // Top panel: header
        egui::TopBottomPanel::top("header")
            .frame(
                egui::Frame::none()
                    .fill(egui::Color32::from_rgb(24, 24, 36))
                    .inner_margin(egui::Margin {
                        left: 12.0,
                        right: 12.0,
                        top: 8.0,
                        bottom: 8.0,
                    }),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Voclaude").size(18.0).strong());
                    ui.add_space(8.0);

                    // State indicator
                    let (state_text, state_color) = match self.status.state.as_str() {
                        "Recording" => ("Recording", egui::Color32::from_rgb(240, 70, 70)),
                        "Transcribing" => {
                            ("Transcribing", egui::Color32::from_rgb(220, 180, 60))
                        }
                        _ => ("Idle", egui::Color32::from_rgb(78, 204, 163)),
                    };
                    let (dot_rect, _) = ui.allocate_exact_size(
                        egui::vec2(10.0, 10.0),
                        egui::Sense::hover(),
                    );
                    ui.painter()
                        .circle_filled(dot_rect.center(), 4.0, state_color);
                    ui.label(
                        egui::RichText::new(state_text)
                            .size(12.0)
                            .color(state_color),
                    );

                    ui.with_layout(
                        egui::Layout::right_to_left(egui::Align::Center),
                        |ui| {
                            if ui.button("Hide").clicked() {
                                self.history_visible = false;
                            }
                            ui.checkbox(&mut self.show_logs, "Log");
                        },
                    );
                });
            });

        // Bottom panel: status bar
        egui::TopBottomPanel::bottom("status_bar")
            .frame(
                egui::Frame::none()
                    .fill(egui::Color32::from_rgb(24, 24, 36))
                    .inner_margin(egui::Margin {
                        left: 12.0,
                        right: 12.0,
                        top: 6.0,
                        bottom: 6.0,
                    }),
            )
            .show(ctx, |ui| {
                // Mic level bar
                if let Some(level) = self.status.input_level {
                    draw_level_bar(ui, level, ui.available_width(), 4.0);
                    ui.add_space(4.0);
                }

                ui.horizontal(|ui| {
                    // GPU/CPU badge
                    let (accel, color) = if self.status.use_gpu {
                        ("GPU", egui::Color32::from_rgb(78, 204, 163))
                    } else {
                        ("CPU", egui::Color32::from_rgb(130, 170, 230))
                    };
                    let badge_bg = egui::Color32::from_rgba_unmultiplied(
                        color.r() / 4,
                        color.g() / 4,
                        color.b() / 4,
                        180,
                    );
                    egui::Frame::none()
                        .fill(badge_bg)
                        .rounding(egui::Rounding::same(3.0))
                        .inner_margin(egui::Margin {
                            left: 5.0,
                            right: 5.0,
                            top: 1.0,
                            bottom: 1.0,
                        })
                        .show(ui, |ui| {
                            ui.label(
                                egui::RichText::new(accel)
                                    .size(10.0)
                                    .color(color)
                                    .strong(),
                            );
                        });

                    ui.separator();

                    if !self.status.model.is_empty() {
                        ui.label(
                            egui::RichText::new(&self.status.model)
                                .size(11.0)
                                .color(egui::Color32::GRAY),
                        );
                        ui.separator();
                    }

                    ui.label(
                        egui::RichText::new(format!("#{}", self.status.history_count))
                            .size(11.0)
                            .color(egui::Color32::GRAY),
                    );

                    if let Some(speed) = self.status.last_speed {
                        ui.separator();
                        ui.label(
                            egui::RichText::new(format!("{:.1}x", speed))
                                .size(11.0)
                                .color(egui::Color32::GRAY),
                        );
                    }
                    if let Some(ms) = self.status.last_duration_ms {
                        ui.separator();
                        ui.label(
                            egui::RichText::new(format_duration_ms(ms))
                                .size(11.0)
                                .color(egui::Color32::GRAY),
                        );
                    }

                    if let Some(device) = &self.status.input_device {
                        if !device.is_empty() {
                            ui.separator();
                            let short = if device.len() > 25 {
                                &device[..25]
                            } else {
                                device
                            };
                            ui.label(
                                egui::RichText::new(short)
                                    .size(10.0)
                                    .color(egui::Color32::from_gray(100)),
                            );
                        }
                    }
                });

                if let Some(msg) = &self.status.last_message {
                    if !msg.is_empty() {
                        ui.label(
                            egui::RichText::new(msg)
                                .size(11.0)
                                .color(egui::Color32::from_gray(160)),
                        );
                    }
                }
            });

        // Central panel: search + history entries + logs
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Search:");
                ui.add(
                    egui::TextEdit::singleline(&mut self.filter)
                        .desired_width(ui.available_width() - 60.0),
                );
                if ui.button("Clear").clicked() {
                    self.filter.clear();
                }
            });
            ui.add_space(4.0);

            let available = ui.available_height();
            let log_height = if self.show_logs { 150.0 } else { 0.0 };
            let history_height = (available - log_height - 8.0).max(100.0);

            egui::ScrollArea::vertical()
                .max_height(history_height)
                .show(ui, |ui| {
                    if self.history.is_empty() {
                        ui.add_space(40.0);
                        ui.vertical_centered(|ui| {
                            ui.label(
                                egui::RichText::new("No transcriptions yet")
                                    .size(14.0)
                                    .color(egui::Color32::from_gray(100)),
                            );
                        });
                    } else {
                        let filter = self.filter.trim().to_lowercase();
                        for entry in &self.history {
                            if !filter.is_empty()
                                && !entry.to_lowercase().contains(&filter)
                            {
                                continue;
                            }
                            egui::Frame::none()
                                .fill(egui::Color32::from_rgb(28, 28, 40))
                                .rounding(egui::Rounding::same(4.0))
                                .inner_margin(egui::Margin::same(8.0))
                                .stroke(egui::Stroke::new(
                                    1.0,
                                    egui::Color32::from_rgb(40, 40, 55),
                                ))
                                .show(ui, |ui| {
                                    ui.add(egui::Label::new(entry.as_str()).wrap(true));
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Min),
                                        |ui| {
                                            if ui.small_button("Copy").clicked() {
                                                if let Ok(mut clipboard) =
                                                    arboard::Clipboard::new()
                                                {
                                                    let _ = clipboard.set_text(entry);
                                                }
                                            }
                                        },
                                    );
                                });
                            ui.add_space(3.0);
                        }
                    }
                });

            if self.show_logs {
                ui.add_space(4.0);
                ui.separator();
                ui.label(egui::RichText::new("Log").size(12.0).strong());
                let log_lines = self.log_buffer.snapshot();
                egui::ScrollArea::vertical()
                    .max_height(log_height)
                    .stick_to_bottom(true)
                    .id_source("log_scroll")
                    .show(ui, |ui| {
                        for line in &log_lines {
                            ui.monospace(egui::RichText::new(line.as_str()).size(10.0));
                        }
                    });
            }
        });
    }
}

impl eframe::App for CombinedApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if !self.theme_applied {
            apply_theme(ctx);
            self.theme_applied = true;
        }

        // Intercept close: hide instead of quitting
        if ctx.input(|i| i.viewport().close_requested()) {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.history_visible = false;
        }

        self.apply_commands();
        self.tick_hud_timers();

        // Determine target window mode
        let target_mode = if self.history_visible {
            WindowMode::History
        } else if self.hud_visible {
            WindowMode::Hud
        } else {
            WindowMode::Hidden
        };

        if target_mode != self.window_mode {
            self.transition_window(ctx, target_mode);
            self.window_mode = target_mode;
        }

        match self.window_mode {
            WindowMode::Hidden => {}
            WindowMode::Hud => self.render_hud_panel(ctx),
            WindowMode::History => self.render_history(ctx),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[cfg(target_os = "windows")]
fn get_primary_screen_size() -> egui::Vec2 {
    use windows_sys::Win32::UI::WindowsAndMessaging::GetSystemMetrics;
    const SM_CXSCREEN: i32 = 0;
    const SM_CYSCREEN: i32 = 1;
    unsafe {
        let w = GetSystemMetrics(SM_CXSCREEN);
        let h = GetSystemMetrics(SM_CYSCREEN);
        if w > 0 && h > 0 {
            egui::vec2(w as f32, h as f32)
        } else {
            egui::vec2(1920.0, 1080.0)
        }
    }
}

#[cfg(not(target_os = "windows"))]
fn get_primary_screen_size() -> egui::Vec2 {
    egui::vec2(1920.0, 1080.0)
}

fn apply_theme(ctx: &egui::Context) {
    let mut v = egui::Visuals::dark();
    v.window_fill = egui::Color32::from_rgb(20, 20, 30);
    v.panel_fill = egui::Color32::from_rgb(20, 20, 30);
    v.extreme_bg_color = egui::Color32::from_rgb(14, 14, 20);
    v.faint_bg_color = egui::Color32::from_rgb(28, 28, 40);
    v.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(32, 32, 46);
    v.widgets.inactive.bg_fill = egui::Color32::from_rgb(40, 40, 58);
    v.widgets.inactive.weak_bg_fill = egui::Color32::from_rgb(36, 36, 52);
    v.widgets.hovered.bg_fill = egui::Color32::from_rgb(50, 50, 72);
    v.widgets.active.bg_fill = egui::Color32::from_rgb(60, 60, 86);
    v.selection.bg_fill = egui::Color32::from_rgb(50, 80, 140);
    ctx.set_visuals(v);
}

fn draw_level_bar(ui: &mut egui::Ui, level: f32, width: f32, height: f32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
    let painter = ui.painter();

    // Background track
    painter.rect_filled(rect, height / 2.0, egui::Color32::from_gray(40));

    // Filled portion with color based on level
    let clamped = level.clamp(0.0, 1.0);
    let filled_w = rect.width() * clamped;
    if filled_w > 0.5 {
        let filled = egui::Rect::from_min_size(rect.min, egui::vec2(filled_w, height));
        let color = if clamped < 0.4 {
            egui::Color32::from_rgb(78, 204, 163)
        } else if clamped < 0.7 {
            egui::Color32::from_rgb(230, 200, 60)
        } else {
            egui::Color32::from_rgb(230, 70, 60)
        };
        painter.rect_filled(filled, height / 2.0, color);
    }
}

fn format_duration(duration: Duration) -> String {
    let total_secs = duration.as_secs();
    let minutes = total_secs / 60;
    let seconds = total_secs % 60;
    format!("{:02}:{:02}", minutes, seconds)
}

fn format_duration_ms(duration_ms: u64) -> String {
    let total_secs = duration_ms / 1000;
    let minutes = total_secs / 60;
    let seconds = total_secs % 60;
    format!("{:02}:{:02}", minutes, seconds)
}

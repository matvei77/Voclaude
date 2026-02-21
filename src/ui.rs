//! History window, HUD overlay, and log buffer.
//!
//! Uses egui's multi-viewport support: the root viewport stays off-screen
//! while HUD and History are spawned as separate OS windows via
//! `show_viewport_deferred()`.

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
// Command enum (HUD-only; history/status use shared state directly)
// ---------------------------------------------------------------------------

pub(crate) enum UiCommand {
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
// Shared state for multi-viewport
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct SharedUiState {
    // HUD
    hud_visible: Arc<AtomicBool>,
    hud_state: Arc<Mutex<HudState>>,
    hud_gpu_enabled: Arc<AtomicBool>,
    hud_input_level: Arc<Mutex<f32>>,
    hud_recording_started_at: Arc<Mutex<Option<Instant>>>,
    hud_ready_until: Arc<Mutex<Option<Instant>>>,

    // History
    history_visible: Arc<AtomicBool>,
    history: Arc<Mutex<VecDeque<String>>>,
    filter: Arc<Mutex<String>>,
    show_logs: Arc<AtomicBool>,

    // Shared
    status: Arc<Mutex<UiStatus>>,
    log_buffer: LogBuffer,
}

impl SharedUiState {
    fn new(log_buffer: LogBuffer, gpu_enabled: bool) -> Self {
        Self {
            hud_visible: Arc::new(AtomicBool::new(false)),
            hud_state: Arc::new(Mutex::new(HudState::Idle)),
            hud_gpu_enabled: Arc::new(AtomicBool::new(gpu_enabled)),
            hud_input_level: Arc::new(Mutex::new(0.0)),
            hud_recording_started_at: Arc::new(Mutex::new(None)),
            hud_ready_until: Arc::new(Mutex::new(None)),

            history_visible: Arc::new(AtomicBool::new(false)),
            history: Arc::new(Mutex::new(VecDeque::new())),
            filter: Arc::new(Mutex::new(String::new())),
            show_logs: Arc::new(AtomicBool::new(false)),

            status: Arc::new(Mutex::new(UiStatus::new(
                String::new(),
                false,
                "model".to_string(),
                None,
            ))),
            log_buffer,
        }
    }
}

// ---------------------------------------------------------------------------
// UiManager
// ---------------------------------------------------------------------------

pub struct UiManager {
    shared: SharedUiState,
    command_tx: Sender<UiCommand>,
    alive: Arc<AtomicBool>,
    repaint_ctx: Arc<Mutex<Option<egui::Context>>>,
}

impl UiManager {
    pub fn new(log_buffer: LogBuffer, gpu_enabled: bool) -> Result<Self, Box<dyn std::error::Error>> {
        let (command_tx, command_rx) = unbounded();
        let shared = SharedUiState::new(log_buffer, gpu_enabled);
        let app = RootApp::new(command_rx, shared.clone());
        let alive = Arc::new(AtomicBool::new(true));
        let alive_thread = alive.clone();
        let alive_keepalive = alive.clone();
        let repaint_ctx: Arc<Mutex<Option<egui::Context>>> = Arc::new(Mutex::new(None));
        let repaint_ctx_cc = repaint_ctx.clone();

        std::thread::spawn(move || {
            let options = eframe::NativeOptions {
                viewport: egui::ViewportBuilder::default()
                    .with_title("Voclaude")
                    .with_inner_size([1.0, 1.0])
                    .with_visible(true)
                    .with_position(egui::pos2(-32000.0, -32000.0))
                    .with_decorations(false)
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

                    // Keepalive thread: pokes the root event loop every 100ms so it
                    // stays responsive to incoming commands even when no child
                    // viewports are visible.
                    let keepalive_ctx = cc.egui_ctx.clone();
                    let keepalive_alive = alive_keepalive.clone();
                    std::thread::spawn(move || {
                        loop {
                            std::thread::sleep(Duration::from_millis(100));
                            if !keepalive_alive.load(Ordering::Relaxed) {
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
        Ok(Self { shared, command_tx, alive, repaint_ctx })
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
        let prev = self.shared.history_visible.load(Ordering::Relaxed);
        self.shared.history_visible.store(!prev, Ordering::Relaxed);
        self.wake();
        true
    }

    pub fn show(&self) -> bool {
        if !self.alive.load(Ordering::Relaxed) {
            error!("UI is not running");
            return false;
        }
        self.shared.history_visible.store(true, Ordering::Relaxed);
        self.wake();
        true
    }

    pub fn push_history(&self, text: String) {
        if let Ok(mut history) = self.shared.history.lock() {
            if !text.trim().is_empty() {
                history.push_front(text);
                if history.len() > 500 {
                    history.pop_back();
                }
            }
        }
        self.wake();
    }

    pub fn reload_history(&self, entries: Vec<String>) {
        if let Ok(mut history) = self.shared.history.lock() {
            history.clear();
            for entry in entries {
                if !entry.trim().is_empty() {
                    history.push_back(entry);
                }
            }
        }
        self.wake();
    }

    pub fn set_status(&self, status: UiStatus) {
        if let Ok(mut s) = self.shared.status.lock() {
            *s = status;
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
// Root app (off-screen, spawns child viewports)
// ---------------------------------------------------------------------------

struct RootApp {
    shared: SharedUiState,
    command_rx: Receiver<UiCommand>,
    theme_applied: bool,
}

impl RootApp {
    fn new(command_rx: Receiver<UiCommand>, shared: SharedUiState) -> Self {
        Self {
            shared,
            command_rx,
            theme_applied: false,
        }
    }

    fn apply_commands(&self) {
        while let Ok(cmd) = self.command_rx.try_recv() {
            match cmd {
                UiCommand::HudSetState(state) => {
                    match &state {
                        HudState::Idle => {
                            self.shared.hud_visible.store(false, Ordering::Relaxed);
                            if let Ok(mut s) = self.shared.hud_recording_started_at.lock() {
                                *s = None;
                            }
                            if let Ok(mut s) = self.shared.hud_ready_until.lock() {
                                *s = None;
                            }
                        }
                        HudState::Recording => {
                            if let Ok(mut s) = self.shared.hud_recording_started_at.lock() {
                                if s.is_none() {
                                    *s = Some(Instant::now());
                                }
                            }
                            self.shared.hud_visible.store(true, Ordering::Relaxed);
                            if let Ok(mut s) = self.shared.hud_ready_until.lock() {
                                *s = None;
                            }
                        }
                        HudState::Transcribing { .. } => {
                            self.shared.hud_visible.store(true, Ordering::Relaxed);
                            if let Ok(mut s) = self.shared.hud_recording_started_at.lock() {
                                *s = None;
                            }
                            if let Ok(mut s) = self.shared.hud_ready_until.lock() {
                                *s = None;
                            }
                        }
                        HudState::Ready { .. } => {
                            self.shared.hud_visible.store(true, Ordering::Relaxed);
                            if let Ok(mut s) = self.shared.hud_recording_started_at.lock() {
                                *s = None;
                            }
                            if let Ok(mut s) = self.shared.hud_ready_until.lock() {
                                *s = Some(Instant::now() + Duration::from_secs(3));
                            }
                        }
                    }
                    if let Ok(mut s) = self.shared.hud_state.lock() {
                        *s = state;
                    }
                }
                UiCommand::HudSetAccel(gpu_enabled) => {
                    self.shared.hud_gpu_enabled.store(gpu_enabled, Ordering::Relaxed);
                }
                UiCommand::HudSetLevel(level) => {
                    if let Ok(mut s) = self.shared.hud_input_level.lock() {
                        *s = level.clamp(0.0, 1.0);
                    }
                }
            }
        }
    }

    fn tick_hud_timers(&self) {
        let ready_until = self.shared.hud_ready_until.lock().ok().and_then(|g| *g);
        if let Some(ready_until) = ready_until {
            if Instant::now() >= ready_until {
                if let Ok(mut s) = self.shared.hud_state.lock() {
                    *s = HudState::Idle;
                }
                self.shared.hud_visible.store(false, Ordering::Relaxed);
                if let Ok(mut s) = self.shared.hud_ready_until.lock() {
                    *s = None;
                }
            }
        }
    }
}

impl eframe::App for RootApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if !self.theme_applied {
            apply_theme(ctx);
            self.theme_applied = true;
        }

        // Root viewport must never die
        if ctx.input(|i| i.viewport().close_requested()) {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
        }

        self.apply_commands();
        self.tick_hud_timers();

        // Spawn HUD child viewport when visible
        if self.shared.hud_visible.load(Ordering::Relaxed) {
            let monitor = get_primary_screen_size();
            let hud_w = 340.0_f32;
            let hud_h = 72.0_f32;
            let x = monitor.x - hud_w - 20.0;
            let y = 24.0;

            let shared = self.shared.clone();
            ctx.show_viewport_deferred(
                egui::ViewportId::from_hash_of("voclaude_hud"),
                egui::ViewportBuilder::default()
                    .with_title("Voclaude HUD")
                    .with_inner_size([hud_w, hud_h])
                    .with_position(egui::pos2(x, y))
                    .with_decorations(false)
                    .with_always_on_top()
                    .with_taskbar(false),
                move |ctx, _class| {
                    render_hud(ctx, &shared);
                },
            );

            // Keep root repainting to tick HUD timers
            ctx.request_repaint_after(Duration::from_millis(100));
        }

        // Spawn History child viewport when visible
        if self.shared.history_visible.load(Ordering::Relaxed) {
            let shared = self.shared.clone();
            let monitor = get_primary_screen_size();
            let win_w = 560.0_f32;
            let win_h = 520.0_f32;
            let x = (monitor.x - win_w) / 2.0;
            let y = (monitor.y - win_h) / 2.0;

            ctx.show_viewport_deferred(
                egui::ViewportId::from_hash_of("voclaude_history"),
                egui::ViewportBuilder::default()
                    .with_title("Voclaude History")
                    .with_inner_size([win_w, win_h])
                    .with_position(egui::pos2(x, y))
                    .with_decorations(true)
                    .with_resizable(true),
                move |ctx, _class| {
                    if ctx.input(|i| i.viewport().close_requested()) {
                        shared.history_visible.store(false, Ordering::Relaxed);
                    }
                    render_history(ctx, &shared);
                },
            );
        }

        // Root must have a CentralPanel (egui requirement)
        egui::CentralPanel::default().show(ctx, |_ui| {});
    }
}

// ---------------------------------------------------------------------------
// Viewport renderers
// ---------------------------------------------------------------------------

fn render_hud(ctx: &egui::Context, shared: &SharedUiState) {
    let hud_state = shared
        .hud_state
        .lock()
        .ok()
        .map(|g| g.clone())
        .unwrap_or(HudState::Idle);
    let gpu_enabled = shared.hud_gpu_enabled.load(Ordering::Relaxed);
    let input_level = shared
        .hud_input_level
        .lock()
        .ok()
        .map(|g| *g)
        .unwrap_or(0.0);
    let recording_started_at = shared
        .hud_recording_started_at
        .lock()
        .ok()
        .and_then(|g| *g);

    let bg = egui::Color32::from_rgb(18, 18, 28);

    egui::CentralPanel::default()
        .frame(
            egui::Frame::none()
                .fill(bg)
                .inner_margin(egui::Margin::same(10.0)),
        )
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                match &hud_state {
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

                        let elapsed = recording_started_at
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
                        let (text, color) = if gpu_enabled {
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

            if matches!(hud_state, HudState::Recording) {
                ui.add_space(6.0);
                draw_level_bar(ui, input_level, ui.available_width(), 6.0);
            }
        });

    ctx.request_repaint_after(Duration::from_millis(100));
}

fn render_history(ctx: &egui::Context, shared: &SharedUiState) {
    let status = shared
        .status
        .lock()
        .ok()
        .map(|g| g.clone())
        .unwrap_or_else(|| UiStatus::new(String::new(), false, String::new(), None));

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
                let (state_text, state_color) = match status.state.as_str() {
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
                            shared.history_visible.store(false, Ordering::Relaxed);
                        }
                        let mut show_logs = shared.show_logs.load(Ordering::Relaxed);
                        ui.checkbox(&mut show_logs, "Log");
                        shared.show_logs.store(show_logs, Ordering::Relaxed);
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
            if let Some(level) = status.input_level {
                draw_level_bar(ui, level, ui.available_width(), 4.0);
                ui.add_space(4.0);
            }

            ui.horizontal(|ui| {
                // GPU/CPU badge
                let (accel, color) = if status.use_gpu {
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

                if !status.model.is_empty() {
                    ui.label(
                        egui::RichText::new(&status.model)
                            .size(11.0)
                            .color(egui::Color32::GRAY),
                    );
                    ui.separator();
                }

                ui.label(
                    egui::RichText::new(format!("#{}", status.history_count))
                        .size(11.0)
                        .color(egui::Color32::GRAY),
                );

                if let Some(speed) = status.last_speed {
                    ui.separator();
                    ui.label(
                        egui::RichText::new(format!("{:.1}x", speed))
                            .size(11.0)
                            .color(egui::Color32::GRAY),
                    );
                }
                if let Some(ms) = status.last_duration_ms {
                    ui.separator();
                    ui.label(
                        egui::RichText::new(format_duration_ms(ms))
                            .size(11.0)
                            .color(egui::Color32::GRAY),
                    );
                }

                if let Some(device) = &status.input_device {
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

            if let Some(msg) = &status.last_message {
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
    let show_logs = shared.show_logs.load(Ordering::Relaxed);

    egui::CentralPanel::default().show(ctx, |ui| {
        // Filter text input
        {
            let mut filter_guard = shared.filter.lock().unwrap_or_else(|e| e.into_inner());
            ui.horizontal(|ui| {
                ui.label("Search:");
                ui.add(
                    egui::TextEdit::singleline(&mut *filter_guard)
                        .desired_width(ui.available_width() - 60.0),
                );
                if ui.button("Clear").clicked() {
                    filter_guard.clear();
                }
            });
        }
        ui.add_space(4.0);

        // Snapshot data for rendering (avoids holding locks during layout)
        let filter_lower = shared
            .filter
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .trim()
            .to_lowercase();
        let history: Vec<String> = shared
            .history
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .cloned()
            .collect();

        let available = ui.available_height();
        let log_height = if show_logs { 150.0 } else { 0.0 };
        let history_height = (available - log_height - 8.0).max(100.0);

        egui::ScrollArea::vertical()
            .max_height(history_height)
            .show(ui, |ui| {
                if history.is_empty() {
                    ui.add_space(40.0);
                    ui.vertical_centered(|ui| {
                        ui.label(
                            egui::RichText::new("No transcriptions yet")
                                .size(14.0)
                                .color(egui::Color32::from_gray(100)),
                        );
                    });
                } else {
                    for entry in &history {
                        if !filter_lower.is_empty()
                            && !entry.to_lowercase().contains(&filter_lower)
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

        if show_logs {
            ui.add_space(4.0);
            ui.separator();
            ui.label(egui::RichText::new("Log").size(12.0).strong());
            let log_lines = shared.log_buffer.snapshot();
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

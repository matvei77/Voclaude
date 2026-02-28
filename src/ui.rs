//! History window and log buffer.
//!
//! Uses egui's multi-viewport support: the root viewport stays off-screen
//! while History is spawned as a separate OS window via
//! `show_viewport_deferred()`.

use eframe::egui;
use std::collections::VecDeque;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
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

    // U-12: Recover from poisoned lock instead of returning empty silently
    pub fn snapshot(&self) -> Vec<String> {
        match self.inner.lock() {
            Ok(buffer) => buffer.lines.iter().cloned().collect(),
            Err(poisoned) => {
                // Recover data from poisoned mutex
                poisoned.into_inner().lines.iter().cloned().collect()
            }
        }
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
        // U-9: Use the full buf slice (not partial write result) to avoid
        // mid-UTF-8 truncation in the ring buffer
        let chunk = String::from_utf8_lossy(buf);
        self.buffer.push_chunk(&chunk);
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.stdout.flush()
    }
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

// ---------------------------------------------------------------------------
// Shared state for multi-viewport
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct SharedUiState {
    // History
    history_visible: Arc<AtomicBool>,
    history_needs_focus: Arc<AtomicBool>,
    /// Whether the history viewport has been created and painted at least once.
    history_viewport_alive: Arc<AtomicBool>,
    history: Arc<Mutex<VecDeque<String>>>,
    filter: Arc<Mutex<String>>,
    show_logs: Arc<AtomicBool>,

    // Shared
    status: Arc<Mutex<UiStatus>>,
    log_buffer: LogBuffer,
}

impl SharedUiState {
    fn new(log_buffer: LogBuffer) -> Self {
        Self {
            history_visible: Arc::new(AtomicBool::new(false)),
            history_needs_focus: Arc::new(AtomicBool::new(false)),
            history_viewport_alive: Arc::new(AtomicBool::new(false)),
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
    alive: Arc<AtomicBool>,
    repaint_ctx: Arc<Mutex<Option<egui::Context>>>,
}

impl UiManager {
    pub fn new(log_buffer: LogBuffer, _gpu_enabled: bool) -> Result<Self, Box<dyn std::error::Error>> {
        let shared = SharedUiState::new(log_buffer);
        let app = RootApp::new(shared.clone());
        let alive = Arc::new(AtomicBool::new(true));
        let alive_thread = alive.clone();
        let alive_keepalive = alive.clone();
        let repaint_ctx: Arc<Mutex<Option<egui::Context>>> = Arc::new(Mutex::new(None));
        let repaint_ctx_cc = repaint_ctx.clone();

        std::thread::spawn(move || {
            let options = eframe::NativeOptions {
                // U-7: Place root viewport off-screen and use WS_EX_TOOLWINDOW
                // (via with_taskbar(false)) to prevent it from appearing in
                // Alt-Tab on Windows 11.
                viewport: egui::ViewportBuilder::default()
                    .with_title("Voclaude")
                    .with_inner_size([1.0, 1.0])
                    .with_position(egui::pos2(-100.0, -100.0))
                    .with_visible(false)
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
                    // stays responsive even when no child viewports are visible.
                    // U-10: Check alive flag BEFORE requesting repaint, and drop
                    // context clone promptly after loop exit (no strong ref past shutdown)
                    let keepalive_ctx = cc.egui_ctx.clone();
                    let keepalive_alive = alive_keepalive.clone();
                    std::thread::spawn(move || {
                        while keepalive_alive.load(Ordering::SeqCst) {
                            std::thread::sleep(Duration::from_millis(100));
                            if !keepalive_alive.load(Ordering::SeqCst) {
                                break;
                            }
                            keepalive_ctx.request_repaint();
                        }
                        drop(keepalive_ctx); // Release Context clone promptly
                    });

                    Box::new(app)
                }),
            ) {
                error!("Failed to start UI: {}", err);
            }
            alive_thread.store(false, Ordering::Relaxed);
        });

        info!("UI thread started");

        // U-2: Wait up to 500ms for repaint_ctx to be initialized by eframe
        // creation closure, so early wake() calls aren't silently dropped.
        let deadline = std::time::Instant::now() + Duration::from_millis(500);
        while std::time::Instant::now() < deadline {
            if let Ok(guard) = repaint_ctx.lock() {
                if guard.is_some() {
                    break;
                }
            }
            std::thread::sleep(Duration::from_millis(10));
        }

        Ok(Self { shared, alive, repaint_ctx })
    }

    fn wake(&self) {
        if let Ok(guard) = self.repaint_ctx.lock() {
            if let Some(ctx) = guard.as_ref() {
                ctx.request_repaint();
            }
        }
    }

    pub fn toggle(&self) -> bool {
        if !self.alive.load(Ordering::SeqCst) {
            error!("UI is not running");
            return false;
        }
        // U-8: Use fetch_xor for atomic toggle (avoids load→!→store race)
        let prev = self.shared.history_visible.fetch_xor(true, Ordering::SeqCst);
        let new_val = !prev;
        if new_val {
            self.shared.history_needs_focus.store(true, Ordering::SeqCst);
        }
        info!("History toggle: {} -> {}", prev, new_val);
        self.wake();
        true
    }

    pub fn show(&self) -> bool {
        if !self.alive.load(Ordering::SeqCst) {
            error!("UI is not running");
            return false;
        }
        // U-11: Use SeqCst for portable happens-before guarantee
        self.shared.history_visible.store(true, Ordering::SeqCst);
        self.shared.history_needs_focus.store(true, Ordering::SeqCst);
        info!("History show requested");
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
                    // U-5: Enforce the same 500-entry cap as push_history
                    if history.len() > 500 {
                        history.pop_front();
                    }
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
}

// ---------------------------------------------------------------------------
// Root app (off-screen, spawns child viewports)
// ---------------------------------------------------------------------------

struct RootApp {
    shared: SharedUiState,
    theme_applied: bool,
}

impl RootApp {
    fn new(shared: SharedUiState) -> Self {
        Self {
            shared,
            theme_applied: false,
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

        // Spawn History child viewport when visible
        let history_vp_id = egui::ViewportId::from_hash_of("voclaude_history");
        if self.shared.history_visible.load(Ordering::Relaxed) {
            let shared = self.shared.clone();
            let viewport_alive = self.shared.history_viewport_alive.load(Ordering::Relaxed);

            // Only send focus commands if the viewport has already been created
            // and painted at least once. On first open, the deferred viewport
            // callback will handle the initial focus.
            if viewport_alive && self.shared.history_needs_focus.swap(false, Ordering::Relaxed) {
                ctx.send_viewport_cmd_to(history_vp_id, egui::ViewportCommand::Minimized(false));
                ctx.send_viewport_cmd_to(history_vp_id, egui::ViewportCommand::Visible(true));
                ctx.send_viewport_cmd_to(history_vp_id, egui::ViewportCommand::Focus);
            }

            // Build viewport — only set position on first creation (when
            // viewport is not yet alive). After that, let the user's window
            // position persist by not overriding it.
            let mut builder = egui::ViewportBuilder::default()
                .with_title("Voclaude History")
                .with_inner_size([560.0, 520.0])
                .with_decorations(true)
                .with_resizable(true)
                .with_visible(true);

            if !viewport_alive {
                // U-6: GetSystemMetrics returns physical pixels for the primary
                // monitor. We divide by the root viewport's pixels_per_point(),
                // which is approximate on multi-monitor setups with different DPI.
                // Clamp to ensure the window doesn't open off-screen.
                let monitor_phys = get_primary_screen_size();
                let ppp = ctx.pixels_per_point();
                let monitor = egui::vec2(monitor_phys.x / ppp, monitor_phys.y / ppp);
                let x = ((monitor.x - 560.0) / 2.0).max(0.0);
                let y = ((monitor.y - 520.0) / 2.0).max(0.0);
                builder = builder.with_position(egui::pos2(x, y));
            }

            ctx.show_viewport_deferred(
                history_vp_id,
                builder,
                move |ctx, _class| {
                    // Mark viewport as alive on first paint
                    if !shared.history_viewport_alive.load(Ordering::Relaxed) {
                        shared.history_viewport_alive.store(true, Ordering::Relaxed);
                        // Request focus on first creation
                        ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
                    }

                    // Handle close
                    if ctx.input(|i| i.viewport().close_requested()) {
                        shared.history_visible.store(false, Ordering::Relaxed);
                        shared.history_viewport_alive.store(false, Ordering::Relaxed);
                    }
                    render_history(ctx, &shared);
                },
            );
            // U-1: Keep root repainting while history is visible so close events
            // from the deferred viewport are captured promptly. Use 50ms instead
            // of 200ms to avoid phantom window re-entry.
            ctx.request_repaint_after(Duration::from_millis(50));
        } else {
            // History not visible — reset viewport_alive so next open
            // gets fresh position and focus.
            self.shared.history_viewport_alive.store(false, Ordering::Relaxed);
        }

        // Root must have a CentralPanel (egui requirement)
        egui::CentralPanel::default().show(ctx, |_ui| {});
    }
}

// ---------------------------------------------------------------------------
// Viewport renderers
// ---------------------------------------------------------------------------

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
                        let short: String = device.chars().take(25).collect();
                        let short = short.as_str();
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
        // U-3: Use unwrap_or_else to recover from poisoned filter mutex
        let filter_lower;
        {
            let mut filter_guard = shared.filter.lock().unwrap_or_else(|e| e.into_inner());
            filter_lower = filter_guard.trim().to_lowercase();
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
                                            // U-4: arboard::Clipboard is created per-click.
                                            // Win32 OpenClipboard serializes access; if the
                                            // app thread holds the clipboard, this will fail
                                            // gracefully with the warning below.
                                            match arboard::Clipboard::new().and_then(|mut c| c.set_text(entry.to_string())) {
                                                Ok(()) => {}
                                                Err(e) => tracing::warn!("Clipboard copy failed (may be locked by another thread): {}", e),
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

fn format_duration_ms(duration_ms: u64) -> String {
    let total_secs = duration_ms / 1000;
    let minutes = total_secs / 60;
    let seconds = total_secs % 60;
    format!("{:02}:{:02}", minutes, seconds)
}

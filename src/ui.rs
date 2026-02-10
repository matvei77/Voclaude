//! History window and log buffer.

use crossbeam_channel::{unbounded, Receiver, Sender};
use eframe::egui;
use std::collections::VecDeque;
use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tracing::{error, info};
use tracing_subscriber::fmt::MakeWriter;

#[cfg(target_os = "windows")]
use winit::platform::windows::EventLoopBuilderExtWindows;

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

pub struct UiManager {
    command_tx: Sender<UiCommand>,
    alive: Arc<AtomicBool>,
}

impl UiManager {
    pub fn new(log_buffer: LogBuffer) -> Result<Self, Box<dyn std::error::Error>> {
        let (command_tx, command_rx) = unbounded();
        let app = HistoryApp::new(command_rx, log_buffer);
        let alive = Arc::new(AtomicBool::new(true));
        let alive_thread = alive.clone();

        std::thread::spawn(move || {
            let options = eframe::NativeOptions {
                viewport: egui::ViewportBuilder::default()
                    .with_title("Voclaude History")
                    .with_inner_size([520.0, 480.0])
                    .with_visible(true)
                    .with_position(egui::pos2(120.0, 120.0)),
                #[cfg(target_os = "windows")]
                event_loop_builder: Some(Box::new(|builder| {
                    builder.with_any_thread(true);
                })),
                ..Default::default()
            };

            if let Err(err) = eframe::run_native(
                "Voclaude History",
                options,
                Box::new(|_cc| Box::new(app)),
            ) {
                error!("Failed to start history window: {}", err);
            }
            alive_thread.store(false, Ordering::Relaxed);
        });

        info!("History window thread started");
        let _ = command_tx.send(UiCommand::Hide);
        Ok(Self { command_tx, alive })
    }

    pub fn toggle(&self) -> bool {
        if !self.alive.load(Ordering::Relaxed) {
            error!("History window is not running");
            return false;
        }
        if let Err(err) = self.command_tx.send(UiCommand::Toggle) {
            error!("Failed to send toggle command: {}", err);
            return false;
        }
        true
    }

    pub fn show(&self) -> bool {
        if !self.alive.load(Ordering::Relaxed) {
            error!("History window is not running");
            return false;
        }
        if let Err(err) = self.command_tx.send(UiCommand::Show) {
            error!("Failed to send show command: {}", err);
            return false;
        }
        true
    }

    pub fn push_history(&self, text: String) {
        if let Err(err) = self.command_tx.send(UiCommand::AddHistory(text)) {
            error!("Failed to send history entry: {}", err);
        }
    }

    pub fn set_status(&self, status: UiStatus) {
        if let Err(err) = self.command_tx.send(UiCommand::SetStatus(status)) {
            error!("Failed to send status update: {}", err);
        }
    }
}

enum UiCommand {
    Toggle,
    Show,
    Hide,
    AddHistory(String),
    SetStatus(UiStatus),
}

#[derive(Debug, Clone)]
pub struct UiStatus {
    pub state: String,
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

pub struct HudManager {
    command_tx: Sender<HudCommand>,
    alive: Arc<AtomicBool>,
}

impl HudManager {
    pub fn new(gpu_enabled: bool) -> Result<Self, Box<dyn std::error::Error>> {
        let (command_tx, command_rx) = unbounded();
        let alive = Arc::new(AtomicBool::new(true));
        let alive_thread = alive.clone();

        std::thread::spawn(move || {
            let app = HudApp::new(command_rx, gpu_enabled);
            let options = eframe::NativeOptions {
                viewport: egui::ViewportBuilder::default()
                    .with_title("Voclaude Status")
                    .with_inner_size([260.0, 72.0])
                    .with_visible(false)
                    .with_resizable(false)
                    .with_decorations(false)
                    .with_always_on_top(),
                #[cfg(target_os = "windows")]
                event_loop_builder: Some(Box::new(|builder| {
                    builder.with_any_thread(true);
                })),
                ..Default::default()
            };

            if let Err(err) = eframe::run_native(
                "Voclaude Status",
                options,
                Box::new(|_cc| Box::new(app)),
            ) {
                error!("Failed to start status HUD: {}", err);
            }
            alive_thread.store(false, Ordering::Relaxed);
        });

        Ok(Self { command_tx, alive })
    }

    pub fn set_state(&self, state: HudState) -> bool {
        if !self.alive.load(Ordering::Relaxed) {
            error!("Status HUD is not running");
            return false;
        }
        if let Err(err) = self.command_tx.send(HudCommand::SetState(state)) {
            error!("Failed to send HUD state: {}", err);
            return false;
        }
        true
    }

    pub fn set_accel(&self, gpu_enabled: bool) -> bool {
        if !self.alive.load(Ordering::Relaxed) {
            error!("Status HUD is not running");
            return false;
        }
        if let Err(err) = self.command_tx.send(HudCommand::SetAccel(gpu_enabled)) {
            error!("Failed to send HUD accel state: {}", err);
            return false;
        }
        true
    }

    pub fn set_level(&self, level: f32) -> bool {
        if !self.alive.load(Ordering::Relaxed) {
            error!("Status HUD is not running");
            return false;
        }
        if let Err(err) = self.command_tx.send(HudCommand::SetLevel(level)) {
            error!("Failed to send HUD level: {}", err);
            return false;
        }
        true
    }
}

enum HudCommand {
    SetState(HudState),
    SetAccel(bool),
    SetLevel(f32),
}

struct HudApp {
    command_rx: Receiver<HudCommand>,
    state: HudState,
    visible: bool,
    recording_started_at: Option<Instant>,
    ready_until: Option<Instant>,
    gpu_enabled: bool,
    input_level: f32,
}

impl HudApp {
    fn new(command_rx: Receiver<HudCommand>, gpu_enabled: bool) -> Self {
        Self {
            command_rx,
            state: HudState::Idle,
            visible: false,
            recording_started_at: None,
            ready_until: None,
            gpu_enabled,
            input_level: 0.0,
        }
    }

    fn apply_commands(&mut self, ctx: &egui::Context) {
        let mut visibility_changed = false;
        while let Ok(cmd) = self.command_rx.try_recv() {
            match cmd {
                HudCommand::SetState(state) => {
                    self.state = state;
                    match &self.state {
                        HudState::Idle => {
                            self.visible = false;
                            self.recording_started_at = None;
                            self.ready_until = None;
                        }
                        HudState::Recording => {
                            if self.recording_started_at.is_none() {
                                self.recording_started_at = Some(Instant::now());
                            }
                            self.visible = true;
                            self.ready_until = None;
                        }
                        HudState::Transcribing { .. } => {
                            self.visible = true;
                            self.recording_started_at = None;
                            self.ready_until = None;
                        }
                        HudState::Ready { .. } => {
                            self.visible = true;
                            self.recording_started_at = None;
                            self.ready_until = Some(Instant::now() + Duration::from_secs(2));
                        }
                    }
                    visibility_changed = true;
                }
                HudCommand::SetAccel(gpu_enabled) => {
                    self.gpu_enabled = gpu_enabled;
                }
                HudCommand::SetLevel(level) => {
                    self.input_level = level.clamp(0.0, 1.0);
                }
            }
        }

        if visibility_changed {
            ctx.send_viewport_cmd(egui::ViewportCommand::Visible(self.visible));
            if self.visible {
                ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
            }
        }
    }
}

impl eframe::App for HudApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.apply_commands(ctx);

        if let Some(ready_until) = self.ready_until {
            if Instant::now() >= ready_until {
                self.state = HudState::Idle;
                self.visible = false;
                self.ready_until = None;
                ctx.send_viewport_cmd(egui::ViewportCommand::Visible(false));
            }
        }

        if !self.visible {
            ctx.request_repaint_after(Duration::from_millis(200));
            return;
        }

        let indicator_color = if self.gpu_enabled {
            egui::Color32::from_rgb(64, 200, 120)
        } else {
            egui::Color32::from_rgb(80, 140, 220)
        };

        let status_text = match &self.state {
            HudState::Idle => "Idle".to_string(),
            HudState::Recording => {
                let elapsed = self
                    .recording_started_at
                    .map(|start| start.elapsed())
                    .unwrap_or_default();
                format!("Recording · {}", format_duration(elapsed))
            }
            HudState::Transcribing { message, percent } => {
                if let Some(percent) = percent {
                    format!("{} ({}%)", message, percent)
                } else {
                    message.clone()
                }
            }
            HudState::Ready { message } => message.clone(),
        };

        egui::CentralPanel::default().show(ctx, |ui| {
            let frame = egui::Frame::none()
                .fill(egui::Color32::from_rgba_unmultiplied(20, 20, 20, 220))
                .rounding(egui::Rounding::same(8.0))
                .inner_margin(egui::Margin::same(8.0));
            frame.show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.colored_label(indicator_color, "●");
                    ui.label(status_text);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let accel = if self.gpu_enabled { "GPU" } else { "CPU" };
                        ui.colored_label(indicator_color, accel);
                    });
                });
                if matches!(self.state, HudState::Recording) {
                    let bar = egui::ProgressBar::new(self.input_level)
                        .desired_width(220.0)
                        .text("Mic level");
                    ui.add(bar);
                }
            });
        });

        ctx.request_repaint_after(Duration::from_millis(100));
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

struct HistoryApp {
    command_rx: Receiver<UiCommand>,
    history: VecDeque<String>,
    log_buffer: LogBuffer,
    visible: bool,
    filter: String,
    show_logs: bool,
    status: UiStatus,
}

impl HistoryApp {
    fn new(command_rx: Receiver<UiCommand>, log_buffer: LogBuffer) -> Self {
        Self {
            command_rx,
            history: VecDeque::new(),
            log_buffer,
            visible: true,
            filter: String::new(),
            show_logs: false,
            status: UiStatus::new(String::new(), false, "whisper".to_string(), None),
        }
    }

    fn apply_commands(&mut self, ctx: &egui::Context) {
        let mut visibility_changed = false;
        while let Ok(cmd) = self.command_rx.try_recv() {
            match cmd {
                UiCommand::Toggle => {
                    self.visible = !self.visible;
                    visibility_changed = true;
                }
                UiCommand::Show => {
                    self.visible = true;
                    visibility_changed = true;
                }
                UiCommand::Hide => {
                    self.visible = false;
                    visibility_changed = true;
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
            }
        }

        if visibility_changed {
            ctx.send_viewport_cmd(egui::ViewportCommand::Visible(self.visible));
            if self.visible {
                ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
                if let Some(cmd) = egui::ViewportCommand::center_on_screen(ctx) {
                    ctx.send_viewport_cmd(cmd);
                }
            }
        }
    }
}

impl eframe::App for HistoryApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.apply_commands(ctx);
        ctx.request_repaint_after(Duration::from_millis(200));

        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Voclaude History");
                ui.checkbox(&mut self.show_logs, "Show log");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Hide").clicked() {
                        self.visible = false;
                        ctx.send_viewport_cmd(egui::ViewportCommand::Visible(false));
                    }
                });
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.group(|ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label(format!("Status: {}", self.status.state));
                    if !self.status.hotkey.is_empty() {
                        ui.label(format!("Hotkey: {}", self.status.hotkey));
                    }
                    let accel_color = if self.status.use_gpu {
                        egui::Color32::from_rgb(64, 200, 120)
                    } else {
                        egui::Color32::from_rgb(80, 140, 220)
                    };
                    let accel_label = if self.status.use_gpu { "GPU" } else { "CPU" };
                    ui.colored_label(accel_color, format!("Accel: {}", accel_label));
                    if !self.status.model.is_empty() {
                        let size_text = self
                            .status
                            .model_size_mb
                            .map(|size| format!(" (~{} MB)", size))
                            .unwrap_or_default();
                        ui.label(format!("Model: {}{}", self.status.model, size_text));
                    }
                    if let Some(device) = &self.status.input_device {
                        if !device.is_empty() {
                            ui.label(format!("Input: {}", device));
                        }
                    }
                    ui.label(format!("History: {}", self.status.history_count));
                    if let Some(duration_ms) = self.status.last_duration_ms {
                        ui.label(format!("Last: {}", format_duration_ms(duration_ms)));
                    }
                    if let Some(speed) = self.status.last_speed {
                        ui.label(format!("Speed: {:.2}x", speed));
                    }
                });
                if let Some(level) = self.status.input_level {
                    let clamped = level.clamp(0.0, 1.0);
                    let bar = egui::ProgressBar::new(clamped)
                        .desired_width(160.0)
                        .text("Mic level");
                    ui.add(bar);
                }
                if let Some(message) = &self.status.last_message {
                    if !message.is_empty() {
                        ui.label(message);
                    }
                }
            });

            ui.heading("History");
            ui.horizontal(|ui| {
                ui.label("Search:");
                ui.text_edit_singleline(&mut self.filter);
                if ui.button("Clear").clicked() {
                    self.filter.clear();
                }
            });
            let available = ui.available_height();
            let log_height = if self.show_logs { 180.0 } else { 0.0 };
            let history_height = (available - log_height).max(120.0);
            egui::ScrollArea::vertical()
                .max_height(history_height)
                .show(ui, |ui| {
                    if self.history.is_empty() {
                        ui.label("No transcriptions yet.");
                    } else {
                        let filter = self.filter.trim().to_lowercase();
                        for entry in &self.history {
                            if !filter.is_empty()
                                && !entry.to_lowercase().contains(&filter)
                            {
                                continue;
                            }
                            ui.horizontal(|ui| {
                                if ui.button("Copy").clicked() {
                                    if let Ok(mut clipboard) = arboard::Clipboard::new() {
                                        let _ = clipboard.set_text(entry);
                                    }
                                }
                                ui.label(entry);
                            });
                            ui.separator();
                        }
                    }
                });

            if self.show_logs {
                ui.add_space(8.0);
                ui.heading("Log");
                let log_lines = self.log_buffer.snapshot();
                egui::ScrollArea::vertical()
                    .max_height(log_height)
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        if log_lines.is_empty() {
                            ui.label("Log buffer is empty.");
                        } else {
                            for line in log_lines {
                                ui.monospace(line);
                            }
                        }
                    });
            }
        });
    }
}

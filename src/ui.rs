//! History window and log buffer.

use crossbeam_channel::{bounded, Receiver, Sender};
use eframe::egui;
use std::collections::VecDeque;
use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use std::time::Duration;
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
}

impl UiManager {
    pub fn new(log_buffer: LogBuffer) -> Result<Self, Box<dyn std::error::Error>> {
        let (command_tx, command_rx) = bounded(64);
        let app = HistoryApp::new(command_rx, log_buffer);

        std::thread::spawn(move || {
            let options = eframe::NativeOptions {
                viewport: egui::ViewportBuilder::default()
                    .with_title("Voclaude History")
                    .with_inner_size([520.0, 480.0])
                    .with_visible(false),
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
        });

        info!("History window thread started");
        Ok(Self { command_tx })
    }

    pub fn toggle(&self) {
        let _ = self.command_tx.send(UiCommand::Toggle);
    }

    pub fn show(&self) {
        let _ = self.command_tx.send(UiCommand::Show);
    }

    pub fn push_history(&self, text: String) {
        let _ = self.command_tx.send(UiCommand::AddHistory(text));
    }
}

enum UiCommand {
    Toggle,
    Show,
    AddHistory(String),
}

struct HistoryApp {
    command_rx: Receiver<UiCommand>,
    history: VecDeque<String>,
    log_buffer: LogBuffer,
    visible: bool,
}

impl HistoryApp {
    fn new(command_rx: Receiver<UiCommand>, log_buffer: LogBuffer) -> Self {
        Self {
            command_rx,
            history: VecDeque::new(),
            log_buffer,
            visible: false,
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
                UiCommand::AddHistory(entry) => {
                    if !entry.trim().is_empty() {
                        self.history.push_front(entry);
                        if self.history.len() > 50 {
                            self.history.pop_back();
                        }
                    }
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

impl eframe::App for HistoryApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.apply_commands(ctx);
        ctx.request_repaint_after(Duration::from_millis(200));

        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Voclaude History");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Hide").clicked() {
                        self.visible = false;
                        ctx.send_viewport_cmd(egui::ViewportCommand::Visible(false));
                    }
                });
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("History");
            egui::ScrollArea::vertical()
                .max_height(180.0)
                .show(ui, |ui| {
                    if self.history.is_empty() {
                        ui.label("No transcriptions yet.");
                    } else {
                        for entry in &self.history {
                            ui.label(entry);
                            ui.separator();
                        }
                    }
                });

            ui.add_space(8.0);
            ui.heading("Terminal");
            let log_lines = self.log_buffer.snapshot();
            egui::ScrollArea::vertical()
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
        });
    }
}

//! The tray app's side of the console window: start `voclaude console --ipc`
//! on demand, stream status to it, and turn its commands into app events.
//!
//! The window runs in its own process, so the tray app never loads the GUI
//! stack (OpenGL driver, fonts). Closing the window ends that process and
//! frees its memory.

use crate::app::AppEvent;
use crate::console::ipc::{FromConsole, Status, ToConsole};
use crossbeam_channel::{unbounded, Sender};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Mic level updates sent to the window at most this often.
const LEVEL_INTERVAL: Duration = Duration::from_millis(50);

struct Running {
    child: Child,
    lines: Sender<String>,
}

pub struct ConsoleHost {
    event_tx: Sender<AppEvent>,
    running: Option<Running>,
    status: Status,
    /// The last status line sent, to skip unchanged updates.
    sent_status: String,
    last_level: Instant,
}

impl ConsoleHost {
    pub fn new(event_tx: Sender<AppEvent>) -> Self {
        Self {
            event_tx,
            running: None,
            status: Status::default(),
            sent_status: String::new(),
            last_level: Instant::now(),
        }
    }

    /// Show the window: start it, or bring the running one forward.
    pub fn open(&mut self) {
        self.reap();
        if let Some(running) = &self.running {
            allow_foreground(running.child.id());
            let _ = running.lines.send(line(&ToConsole::Focus));
            return;
        }
        match self.spawn() {
            Ok(running) => {
                info!("Console started (pid {})", running.child.id());
                self.running = Some(running);
                self.sent_status.clear();
                let status = self.status.clone();
                self.set_status(&status);
            }
            Err(e) => warn!("Could not start the console window: {}", e),
        }
    }

    fn spawn(&self) -> std::io::Result<Running> {
        let exe = std::env::current_exe()?;
        let mut cmd = Command::new(exe);
        cmd.args(["console", "--ipc"]).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null());
        let mut child = cmd.spawn()?;
        allow_foreground(child.id());
        let mut stdin = child.stdin.take().ok_or_else(|| std::io::Error::other("no stdin"))?;
        let stdout = child.stdout.take().ok_or_else(|| std::io::Error::other("no stdout"))?;

        // Writer thread: a stuck window never blocks the event loop.
        let (lines, rx) = unbounded::<String>();
        std::thread::Builder::new().name("console-writer".into()).spawn(move || {
            for l in rx {
                if stdin.write_all(l.as_bytes()).and_then(|_| stdin.flush()).is_err() {
                    break;
                }
            }
        })?;

        let event_tx = self.event_tx.clone();
        std::thread::Builder::new().name("console-reader".into()).spawn(move || {
            for l in BufReader::new(stdout).lines() {
                let Ok(l) = l else { break };
                match serde_json::from_str::<FromConsole>(&l) {
                    Ok(cmd) => {
                        if event_tx.send(AppEvent::ConsoleCommand(cmd)).is_err() {
                            break;
                        }
                    }
                    Err(e) => debug!("Ignoring console line: {}", e),
                }
            }
        })?;
        Ok(Running { child, lines })
    }

    /// Forget a window that has exited.
    fn reap(&mut self) {
        if let Some(running) = self.running.as_mut() {
            if !matches!(running.child.try_wait(), Ok(None)) {
                debug!("Console window closed");
                self.running = None;
            }
        }
    }

    /// Called from the event loop: drop the handle once the window is closed.
    pub fn poll(&mut self) {
        self.reap();
    }

    /// Remember `status` and send it if a window is open and it changed.
    pub fn set_status(&mut self, status: &Status) {
        if *status != self.status {
            self.status = status.clone();
        }
        let Some(running) = &self.running else { return };
        let text = line(&ToConsole::Status(self.status.clone()));
        if text != self.sent_status {
            let _ = running.lines.send(text.clone());
            self.sent_status = text;
        }
    }

    /// Mic level while recording; rate-limited.
    pub fn set_level(&mut self, level: f32) {
        let Some(running) = &self.running else { return };
        if self.last_level.elapsed() >= LEVEL_INTERVAL {
            self.last_level = Instant::now();
            let _ = running.lines.send(line(&ToConsole::Level { level }));
        }
    }

    pub fn history_changed(&mut self) {
        if let Some(running) = &self.running {
            let _ = running.lines.send(line(&ToConsole::HistoryChanged));
        }
    }

    /// Close the window with the app (its stdin closes; it exits on EOF).
    pub fn shutdown(&mut self) {
        self.running = None;
    }
}

fn line(msg: &ToConsole) -> String {
    let mut s = serde_json::to_string(msg).unwrap_or_default();
    s.push('\n');
    s
}

/// Let the window process take the foreground: the tray app got the click or
/// hotkey, so Windows lets it pass that right on.
fn allow_foreground(pid: u32) {
    #[cfg(target_os = "windows")]
    unsafe {
        windows_sys::Win32::UI::WindowsAndMessaging::AllowSetForegroundWindow(pid);
    }
    #[cfg(not(target_os = "windows"))]
    let _ = pid;
}

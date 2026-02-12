# Voclaude - Context Dump

Generated: 2026-01-30

This file is a single-place “drop-in” project context snapshot for model-assisted review.

## Repo Snapshot

- Repo root: `C:\Users\matvei\Documents\code\Voclaude`
- Current branch: `feature/history-ui` (local `master` points to same commit)
- HEAD commit: `130cdc79105033388a9c6b1db5d77d66fcc5b682`
- Recent commits:
  - `130cdc7 (HEAD -> feature/history-ui, origin/feature/history-ui, master) Add history UI and improve capture`
  - `cabb87a (origin/master) Windows working version`
- Working tree status (porcelain):
  - `M .claude/settings.local.json`
- Toolchain:
  - `rustc 1.91.1 (ed61e7d7e 2025-11-07)`
  - `cargo 1.91.1 (ea2d97820 2025-10-10)`
- Platform: Windows (`win32`)

## What This Project Is

Voclaude is a Rust system-tray application that captures microphone audio, transcribes it locally with whisper.cpp (via `whisper-rs`), and copies the resulting text to the clipboard.

Core UX:
- Lives in the system tray.
- Global hotkey toggles recording (default `F4`).
- When you stop recording, it transcribes and copies text to clipboard.
- Optional hidden “History” window (egui) shows recent transcriptions + a live log buffer.

Local-first model behavior:
- Whisper medium model (`ggml-medium.bin`, ~1.5GB) is downloaded on first use.
- Model is lazily loaded and can be unloaded after idle to free RAM/VRAM.

## How To Build / Run

From `README.md`:

```bash
# CPU only
cargo build --release

# With CUDA (make sure CUDA_PATH/CUDA_HOME is set)
cargo build --release --features cuda

# Run
cargo run --release
```

WAV “test mode” (no tray/hotkeys; transcribes a WAV file):

```bash
cargo run --release -- --test "C:\\path\\to\\file.wav"
```

Notes:
- Repo includes `.cargo/config.toml` that pins `CUDA_PATH` to CUDA 12.8 and forces `CMAKE_GENERATOR=Ninja`.
- `build.rs` hardcodes a Windows CUDA library search path for CUDA 12.8.

## Runtime Files / Paths

Voclaude uses `directories::ProjectDirs::from("com", "voclaude", "Voclaude")`.

Computed locations in code:
- Config file: `ProjectDirs.config_dir()/config.toml` (`src/config.rs`)
- Models dir: `ProjectDirs.data_dir()/models/` (`src/config.rs`, used by Whisper download)
- History file: `ProjectDirs.data_dir()/history.json` (`src/history.rs`)

Important: On Windows, `directories` typically uses subfolders like `...\config\` and `...\data\`. The README’s Windows path example omits the `config/` segment; treat the code as authoritative.

## Architecture (High-Level)

From `README.md`:

```
┌─────────────────────────────────┐
│         Main Thread             │
│  - Tray icon                    │
│  - Hotkey listener              │
│  - State machine                │
└───────────────┬─────────────────┘
                │
┌───────────────▼─────────────────┐
│        Audio Thread             │
│  - cpal recording               │
│  - Lock-free ring buffer        │
│  - 16kHz mono resampling        │
└───────────────┬─────────────────┘
                │
┌───────────────▼─────────────────┐
│      Inference Thread           │
│  - whisper.cpp (via whisper-rs) │
│  - Lazy model loading           │
│  - GPU optional + CPU fallback  │
└─────────────────────────────────┘
```

More precise (from code):

- Main thread (`src/app.rs`):
  - Owns the `App` state machine (`Idle` / `Recording` / `Transcribing`).
  - Pumps Windows messages each tick (`pump_messages`) to keep hotkeys and tray responsive.
  - Receives `AppEvent` messages from other threads over a bounded crossbeam channel.
  - Owns the `TrayManager` (must stay alive) and two `HotkeyManager`s (record + history).
  - Starts/Stops `AudioCapture`, dispatches inference work, formats output, writes to clipboard, updates history, toggles UI.

- Audio capture (cpal callback; `src/audio/capture.rs`):
  - Builds an input stream based on best available device config.
  - Converts interleaved frames to mono, resamples to 16kHz if needed, and pushes into a shared buffer.
  - Hard cap: stops capture after 600s worth of 16kHz samples.

- Inference worker thread (`src/app.rs`):
  - Receives `InferenceCommand` (`Transcribe(Vec<f32>)`, `Unload`, `Shutdown`).
  - Owns `WhisperEngine`, which downloads and loads the model lazily.
  - Emits progress updates (`InferenceProgress`) and completion (`TranscriptionComplete`).

- Hotkey listener threads (`src/hotkey.rs`):
  - Registers a global hotkey via `global-hotkey`.
  - Spawns a thread that listens on `GlobalHotKeyEvent::receiver()` and forwards matching “Pressed” events as `AppEvent`.

- Tray menu listener thread (`src/tray.rs`):
  - Listens on `MenuEvent::receiver()` and forwards menu clicks as `AppEvent`.

- UI thread (`src/ui.rs`):
  - Runs an `eframe` window loop in a spawned thread.
  - Window starts invisible; main thread toggles visibility via a command channel.
  - UI shows two panels: recent history strings and an in-memory log buffer.

## State Machine / Event Flow (End-to-End)

Events (`AppEvent` in `src/app.rs`):
- `HotkeyPressed` (record toggle)
- `ToggleHistoryWindow` / `ShowHistoryWindow`
- `InferenceProgress`
- `TranscriptionComplete(Result<String,String>)`
- `HistoryUpdated(HistoryEntry)`
- `Quit`

Typical transcription flow:
1) Hotkey or tray menu triggers `HotkeyPressed`.
2) `Idle -> Recording`: start `AudioCapture`.
3) Hotkey again triggers `HotkeyPressed`.
4) `Recording -> Transcribing`: stop audio, get samples, send `InferenceCommand::Transcribe(samples)`.
5) Inference worker:
   - Ensures model file exists (downloads if missing).
   - Loads model context (GPU if requested + built with `--features cuda`; CPU otherwise; retries CPU if GPU init fails).
   - Runs whisper inference, concatenates segments, returns final text.
6) Main thread on completion:
   - Formats text (trim; optionally capitalize first; optionally add trailing space).
   - Copies formatted text to clipboard.
   - Appends trimmed text to persistent history JSON.
   - Updates tray back to idle.

## Key Risks / Issues (Ranked)

This is a non-exhaustive “what to look at first” list.

1) Audio callback is not real-time safe
   - Allocations + mutex lock + mono mixing + resampling happen inside the CPAL callback.
   - Likely to cause glitches/dropouts under load.
   - `src/audio/capture.rs`, `src/audio/ring_buffer.rs`.

2) “Lock-free ring buffer” mismatch
   - README / docs imply lock-free ring buffer; implementation is `Mutex<Vec<f32>>`.
   - `src/audio/ring_buffer.rs`, `README.md`.

3) Backpressure + blocking sends
   - App event channel is bounded(32); producers use blocking `send()`.
   - Inference progress callback uses `send()` too; could block inference/download.
   - `src/app.rs`, `src/hotkey.rs`, `src/tray.rs`.

4) UI channel can block main thread on startup
   - App seeds up to `history_max_entries` into UI via blocking `send()`.
   - UI command channel is bounded(64); if UI thread lags, main thread can stall.
   - `src/app.rs`, `src/ui.rs`.

5) Windows-centric plumbing + cross-platform gotchas
   - Main loop relies on explicit Windows message pumping.
   - UI is spawned on a background thread; macOS is commonly main-thread-only for UI.
   - `src/app.rs`, `src/ui.rs`.

6) CUDA build wiring is very brittle
   - `.cargo/config.toml` pins CUDA 12.8; `build.rs` hardcodes lib search path.
   - `build.rs`, `.cargo/config.toml`.

7) Model download robustness
   - No checksum/size validation for existing model file.
   - No retry/backoff.
   - `src/inference/whisper.rs`.

8) History persistence safety
   - Parse failure -> “empty history” -> immediately persisted.
   - Writes are non-atomic.
   - `src/history.rs`.

9) Config mismatch
   - Config has `language`, README says auto-detect, but Whisper hardcodes English.
   - `src/config.rs`, `src/inference/whisper.rs`.

## Notable Repository Discrepancies

- `src/inference/parakeet.rs`, `src/inference/model_manager.rs`, and `src/inference/mel.rs` are tracked but not compiled (not `mod`’d from `src/inference/mod.rs`). They also reference crates not present in `Cargo.toml` (would not compile if enabled).
- `rubato` is listed in `Cargo.toml` but is not referenced in the compiled code.
- `license = "MIT"` is declared in `Cargo.toml`, but no `LICENSE` file is tracked (based on `git ls-files`).

## File Index (Tracked)

Output of `git ls-files`:

```text
.cargo/config.toml
.claude/settings.local.json
.gitignore
Cargo.lock
Cargo.toml
README.md
assets/icon_idle.png
assets/icon_processing.png
assets/icon_recording.png
build.rs
cognitiveagents.md
src/app.rs
src/audio/capture.rs
src/audio/mod.rs
src/audio/processing.rs
src/audio/ring_buffer.rs
src/config.rs
src/history.rs
src/hotkey.rs
src/inference/mel.rs
src/inference/mod.rs
src/inference/model_manager.rs
src/inference/parakeet.rs
src/inference/whisper.rs
src/main.rs
src/tray.rs
src/ui.rs
```

## What’s Not Inlined Here

- `target/` is intentionally excluded (build artifacts, huge).
- Binary PNG contents are not inlined (they are referenced by path and embedded in the compiled binary via `include_bytes!`).
- `Cargo.lock` is not inlined (7013 lines; generated). If you want it embedded too, append it as an extra section at the end.

---

# Full File Contents

Below are the exact tracked text files (minus `Cargo.lock`) inlined for review.

## `.cargo/config.toml`

```toml
# Build configuration for Voclaude
# Sets CUDA environment for whisper-rs-sys cmake build

[env]
# Use CUDA 12.8 (not 13.1 which has linking issues)
CUDA_PATH = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"

# Use Ninja generator to avoid VS CUDA integration issues
# (Nsight VS integration not installed, causes CudaToolkitDir errors)
CMAKE_GENERATOR = "Ninja"
```

## `.claude/settings.local.json`

```json
{
  "permissions": {
    "allow": [
      "Bash(dir \"C:\\Users\\matvei\\Documents\\code\\voiceinput\")",
      "Bash(cargo build:*)",
      "Bash(taskkill:*)",
      "WebSearch",
      "WebFetch(domain:docs.rs)",
      "WebFetch(domain:api.github.com)",
      "WebFetch(domain:raw.githubusercontent.com)",
      "Bash(powershell -Command \"Stop-Process -Name voclaude -Force -ErrorAction SilentlyContinue; Start-Sleep -Seconds 1; Write-Host ''Done''\")",
      "Bash(timeout 120 ./target/release/voclaude.exe:*)",
      "Bash(./target/release/voclaude.exe:*)",
      "Bash(dir:*)",
      "Bash(findstr:*)",
      "Bash(dumpbin:*)",
      "Bash(wc:*)",
      "Bash(git add:*)",
      "Bash(git commit -m \"$(cat <<''EOF''\nWindows working version\n\nInitial commit with fully functional Windows speech-to-text application:\n- Whisper-based transcription with CUDA GPU acceleration\n- System tray integration with global hotkey (Win+C)\n- Auto-download of Whisper Medium model\n- Configurable idle unload to save VRAM\n- Audio capture via cpal with resampling to 16kHz\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\nEOF\n)\")",
      "Bash(git remote set-url:*)",
      "Bash(git push:*)"
    ],
    "deny": [],
    "ask": []
  }
}
```

## `.gitignore`

```gitignore
# Build artifacts
/target/
**/*.rs.bk

# IDE
.idea/
.vscode/
*.swp
*.swo

# Models (downloaded at runtime)
/models/

# OS files
.DS_Store
Thumbs.db

# Logs
*.log
```

## `Cargo.toml`

```toml
[package]
name = "voclaude"
version = "0.3.0"
edition = "2021"
description = "Voice input anywhere - local-first, GPU-accelerated speech-to-text with Whisper"
license = "MIT"
repository = "https://github.com/matvei77/Voclaude"

[dependencies]
# Audio capture
cpal = "0.15"
hound = "3.5"              # WAV encoding

# Whisper speech-to-text (whisper.cpp)
whisper-rs = { version = "0.14" }

# Audio processing
rubato = "0.15"            # High-quality resampling

# UI
tray-icon = "0.19"
global-hotkey = "0.6"
image = "0.25"
eframe = { version = "0.27", default-features = true }
winit = "0.29"

# System
arboard = "3.4"            # Clipboard
directories = "5.0"        # Config paths
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
toml = "0.8"
crossbeam-channel = "0.5"  # Fast MPSC channels

# Logging
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# HTTP for model downloads
reqwest = { version = "0.12", features = ["json", "blocking"] }

# Windows message loop
[target.'cfg(target_os = "windows")'.dependencies]
windows-sys = { version = "0.59", features = ["Win32_UI_WindowsAndMessaging"] }

[target.'cfg(not(target_os = "windows"))'.dependencies]
notify-rust = "4.11"

[features]
default = []
cuda = ["whisper-rs/cuda"]

[profile.release]
lto = true
codegen-units = 1
strip = true
opt-level = 3

[profile.dev]
opt-level = 1              # Faster dev builds but not painfully slow
```

## `README.md`

```markdown
# Voclaude

Voice input anywhere — local-first, GPU-accelerated speech-to-text.

## Features

- **Local-first**: Uses Whisper.cpp for on-device transcription
- **GPU optional**: CUDA acceleration with CPU fallback
- **Lazy loading**: Model loads on first use, unloads after idle
- **System tray**: Minimal UI, stays out of your way
- **Global hotkey**: F4 (configurable) to start/stop recording
- **History window**: Hidden UI with transcript + log history

## Installation

### Prerequisites

- Rust 1.70+ (`rustup` recommended)
- For GPU: CUDA toolkit (NVIDIA)

### Build

```bash
# CPU only
cargo build --release

# With CUDA (make sure CUDA_PATH/CUDA_HOME is set)
cargo build --release --features cuda
```

### Run

```bash
cargo run --release
```

Or copy `target/release/voclaude.exe` to your preferred location.

## Usage

1. Launch Voclaude — it appears in your system tray
2. Press **F4** to start recording (or click tray icon → "Start Recording")
3. Speak
4. Press **F4** again to stop and transcribe
5. Text is copied to clipboard — paste with Ctrl+V
6. Press **Ctrl+Shift+H** or use tray menu → "Show History" for history

## Configuration

Config file location:
- Windows: `%APPDATA%\voclaude\Voclaude\config.toml`
- macOS: `~/Library/Application Support/com.voclaude.Voclaude/config.toml`
- Linux: `~/.config/voclaude/config.toml`

```toml
# Hotkey to toggle recording
hotkey = "F4"

# Hotkey to toggle the history window
history_hotkey = "Ctrl+Shift+H"

# Language (null = auto-detect)
language = null

# Add trailing space after transcription
add_trailing_space = true

# Capitalize first letter
capitalize_first = true

# Unload model after N seconds idle (saves VRAM)
idle_unload_seconds = 300

# Show system notifications
show_notifications = true

# Maximum number of history entries
history_max_entries = 500

# Enable GPU acceleration when available
use_gpu = true
```

## Model

Voclaude currently downloads the Whisper medium model (~1.5GB) automatically on first use.

## Architecture

```
┌─────────────────────────────────┐
│         Main Thread             │
│  - Tray icon                    │
│  - Hotkey listener              │
│  - State machine                │
└───────────────┬─────────────────┘
                │
┌───────────────▼─────────────────┐
│        Audio Thread             │
│  - cpal recording               │
│  - Lock-free ring buffer        │
│  - 16kHz mono resampling        │
└───────────────┬─────────────────┘
                │
┌───────────────▼─────────────────┐
│      Inference Thread           │
│  - whisper.cpp (via whisper-rs) │
│  - Lazy model loading           │
│  - GPU optional + CPU fallback  │
└─────────────────────────────────┘
```

## License

MIT
```

## `build.rs`

```rust
use std::env;

fn main() {
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-search=native=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/lib/x64");
    } else {
        let cuda_root = env::var("CUDA_PATH").or_else(|_| env::var("CUDA_HOME"));
        if let Ok(cuda_root) = cuda_root {
            println!("cargo:rustc-link-search=native={}/lib64", cuda_root);
        } else {
            println!("cargo:warning=CUDA feature enabled but CUDA_PATH/CUDA_HOME not set; relying on system linker paths");
        }
    }

    // CUDA runtime linkage for whisper.cpp on Windows.
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cudart_static");
    println!("cargo:rustc-link-lib=cudadevrt");
    println!("cargo:rustc-link-lib=cuda");
}
```

## `cognitiveagents.md`

```markdown
# Voclaude Codebase Analysis

## What It Does
Voclaude is a Rust system-tray application that captures microphone audio, transcribes it locally with whisper.cpp, and copies the result to the clipboard. It runs a Windows message pump, listens for a global hotkey, and toggles recording/transcription states. Models are downloaded lazily and can be unloaded after idle.

## Current Features
- System tray icon with start/stop, history, and quit menu items.
- Global hotkey to toggle recording; separate hotkey to show history UI.
- Audio capture via cpal with resampling to 16 kHz mono.
- Local transcription via whisper-rs (whisper.cpp bindings).
- Clipboard output with optional capitalization and trailing space.
- Lazy model download and unload after idle.
- Hidden history window with transcript list and log buffer panel.
- JSON-backed history persistence with retention cap.

## Architecture Summary
- `src/main.rs`: entry point, config load, test mode.
- `src/app.rs`: event loop, state machine, worker threads, clipboard, history updates.
- `src/audio/*`: capture, resampling, mono conversion, ring buffer.
- `src/inference/whisper.rs`: whisper-rs engine and model download.
- `src/tray.rs`: tray icon and menu.
- `src/ui.rs`: hidden history/log window (egui).
- `src/history.rs`: persistent history store.

## Known Risks and Issues
- GPU acceleration depends on CUDA feature and installed toolkit; CPU fallback is provided.
- Recording buffer is bounded, but a hard cap stops capture if exceeded.
- The history UI is a separate window thread; it should be validated on Windows.
- The project is Windows-centric; non-Windows builds may need additional cleanup.

## Improvements (Next Candidates)
- Add a Windows-native notification backend for show_notifications.
- Implement settings UI and wire up the Settings menu item.
- Add optional audio capture persistence for playback in the history UI.
- Add streaming or chunked transcription for long recordings.
```

---

## `src/main.rs`

```rust
//! Voclaude - Voice input anywhere
//!
//! Local-first, GPU-accelerated speech-to-text that runs in your system tray.

mod app;
mod audio;
mod config;
mod history;
mod hotkey;
mod inference;
mod tray;
mod ui;

use app::App;
use audio::resample_linear;
use config::Config;
use inference::WhisperEngine;
use tracing::{info, error, Level};

fn main() {
    let log_buffer = ui::LogBuffer::new(400);
    let log_writer = log_buffer.make_writer();

    tracing_subscriber::fmt()
        .with_max_level(Level::DEBUG)
        .with_target(false)
        .compact()
        .with_writer(log_writer)
        .init();

    // Check for --test argument
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "--test" {
        let wav_path = if args.len() > 2 {
            &args[2]
        } else {
            r"C:\Users\matvei\Documents\Sound Recordings\pedro_email.wav"
        };

        if let Err(e) = run_test(wav_path) {
            error!("Test failed: {}", e);
            std::process::exit(1);
        }
        return;
    }

    info!("Voclaude starting...");

    // Load config
    let config = match Config::load() {
        Ok(c) => {
            info!("Config loaded successfully");
            info!("  hotkey: {}", c.hotkey);
            info!("  history_hotkey: {}", c.history_hotkey);
            info!("  idle_unload_seconds: {}", c.idle_unload_seconds);
            info!("  history_max_entries: {}", c.history_max_entries);
            info!("  use_gpu: {}", c.use_gpu);
            c
        }
        Err(e) => {
            error!("Failed to load config: {}", e);
            Config::default()
        }
    };

    // Run the app
    if let Err(e) = App::run(config, log_buffer) {
        error!("Application error: {}", e);
        std::process::exit(1);
    }
}

/// Test mode: load WAV file and transcribe it
fn run_test(wav_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    info!("=== TEST MODE ===");
    info!("Loading WAV file: {}", wav_path);

    // Read WAV file
    let mut reader = hound::WavReader::open(wav_path)?;
    let spec = reader.spec();
    info!("WAV format: {} Hz, {} channels, {} bits",
          spec.sample_rate, spec.channels, spec.bits_per_sample);

    // Convert samples to f32 mono at 16kHz
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => {
            reader.samples::<f32>()
                .filter_map(|s| s.ok())
                .collect()
        }
    };

    // Convert to mono if stereo
    let mono_samples: Vec<f32> = if spec.channels == 2 {
        samples.chunks(2)
            .map(|chunk| (chunk[0] + chunk.get(1).unwrap_or(&0.0)) / 2.0)
            .collect()
    } else {
        samples
    };

    // Resample to 16kHz if needed
    let samples_16k = if spec.sample_rate != 16000 {
        info!("Resampling from {} Hz to 16000 Hz...", spec.sample_rate);
        resample_linear(&mono_samples, spec.sample_rate, 16000)
    } else {
        mono_samples
    };

    // No limit - process full file

    info!("Audio: {:.2}s ({} samples at 16kHz)",
          samples_16k.len() as f32 / 16000.0, samples_16k.len());

    // Create engine and transcribe
    info!("Creating Whisper engine...");
    let test_config = Config::load().unwrap_or_default();
    let mut engine = WhisperEngine::new_with_config(&test_config)?;

    info!("Transcribing...");
    let start = std::time::Instant::now();
    let text = engine.transcribe(&samples_16k)?;
    let elapsed = start.elapsed();

    info!("=== RESULT ===");
    info!("Time: {:.2}s", elapsed.as_secs_f32());
    info!("Text: {}", text);
    println!("\n>>> TRANSCRIPTION:\n{}\n", text);

    Ok(())
}
```

## `src/app.rs`

```rust
//! Main application orchestration.

use crate::audio::AudioCapture;
use crate::audio::WHISPER_SAMPLE_RATE;
use crate::config::Config;
use crate::history::{AudioMetadata, HistoryEntry, HistoryStore};
use crate::hotkey::HotkeyManager;
use crate::inference::{InferenceProgress, InferenceStage, WhisperEngine};
use crate::tray::TrayManager;
use crate::ui::{LogBuffer, UiManager};

use crossbeam_channel::{bounded, Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn, trace};

#[cfg(not(target_os = "windows"))]
use notify_rust::Notification;

/// Application events
#[derive(Debug, Clone)]
pub enum AppEvent {
    /// Hotkey pressed - toggle recording
    HotkeyPressed,
    /// Toggle history window
    ToggleHistoryWindow,
    /// Show history window
    ShowHistoryWindow,
    /// Quit requested
    Quit,
    /// Inference progress update
    InferenceProgress(InferenceProgress),
    /// Transcription completed
    TranscriptionComplete(Result<String, String>),
    /// History updated (for UI listeners)
    HistoryUpdated(HistoryEntry),
}

/// Application state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppState {
    Idle,
    Recording,
    Transcribing,
}

#[derive(Debug)]
enum InferenceCommand {
    Transcribe(Vec<f32>),
    Unload,
    Shutdown,
}

struct NotificationManager {
    enabled: bool,
    last_message: Option<String>,
    last_sent: Instant,
}

impl NotificationManager {
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            last_message: None,
            last_sent: Instant::now() - Duration::from_secs(60),
        }
    }

    fn notify(&mut self, message: &str) {
        if !self.enabled {
            return;
        }

        let min_interval = Duration::from_secs(10);
        if self.last_message.as_deref() == Some(message) && self.last_sent.elapsed() < min_interval {
            return;
        }

        #[cfg(not(target_os = "windows"))]
        {
            if let Err(e) = Notification::new()
                .summary("Voclaude")
                .body(message)
                .show()
            {
                warn!("Failed to show notification: {}", e);
                return;
            }
        }

        #[cfg(target_os = "windows")]
        {
            debug!("Notification: {}", message);
        }

        self.last_message = Some(message.to_string());
        self.last_sent = Instant::now();
    }
}

/// Main application
pub struct App {
    config: Config,
    state: AppState,
    event_tx: Sender<AppEvent>,
    event_rx: Receiver<AppEvent>,
    is_running: bool,
    ui: UiManager,
}

impl App {
    /// Run the application
    pub fn run(config: Config, log_buffer: LogBuffer) -> Result<(), Box<dyn std::error::Error>> {
        let (event_tx, event_rx) = bounded::<AppEvent>(32);
        let ui = UiManager::new(log_buffer)?;

        let mut app = App {
            config,
            state: AppState::Idle,
            event_tx,
            event_rx,
            is_running: true,
            ui,
        };

        app.run_event_loop()
    }

    fn run_event_loop(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        info!("Initializing components...");

        // Initialize tray icon - MUST keep alive!
        let _tray = TrayManager::new(self.event_tx.clone())?;
        info!("Tray icon ready");

        // Initialize hotkey listener - MUST keep alive!
        let _hotkey = HotkeyManager::new(
            &self.config.hotkey,
            AppEvent::HotkeyPressed,
            self.event_tx.clone(),
        )?;
        let _history_hotkey = HotkeyManager::new(
            &self.config.history_hotkey,
            AppEvent::ToggleHistoryWindow,
            self.event_tx.clone(),
        )?;
        info!("Hotkey registered: {}", self.config.hotkey);
        info!("History hotkey registered: {}", self.config.history_hotkey);

        // Initialize audio capture
        let audio = AudioCapture::new()?;
        info!("Audio capture ready");

        // Initialize inference worker (lazy - loads model on demand)
        let inference_tx = Self::spawn_inference_worker(self.event_tx.clone(), self.config.clone());
        info!("Inference worker ready (model will load on first use)");

        // Clipboard
        let mut clipboard = arboard::Clipboard::new()?;

        let mut notifications = NotificationManager::new(self.config.show_notifications);

        // History storage
        let (history_update_tx, history_update_rx) = bounded::<HistoryEntry>(32);
        let mut history = HistoryStore::load(self.config.history_max_entries, history_update_tx)?;
        info!("History loaded: {} entries", history.len());
        for entry in history.entries() {
            self.ui.push_history(entry.text.clone());
        }

        // Track last activity for idle unload
        let mut last_activity = Instant::now();
        let mut idle_unload_requested = false;
        let mut pending_audio_metadata: Option<AudioMetadata> = None;
        let mut last_progress_stage: Option<InferenceStage> = None;

        info!("=== VOCLAUDE READY ===");
        info!("Press {} to start recording", self.config.hotkey);
        info!("Press {} to toggle history window", self.config.history_hotkey);
        info!("Main thread ID: {:?}", std::thread::current().id());

        // Log Windows thread ID for debugging
        #[cfg(target_os = "windows")]
        {
            #[link(name = "kernel32")]
            extern "system" {
                fn GetCurrentThreadId() -> u32;
            }
            let win_thread_id = unsafe { GetCurrentThreadId() };
            info!("Windows Thread ID: {}", win_thread_id);
        }

        info!("Entering main event loop...");

        let mut loop_count: u64 = 0;
        let mut last_status_log = Instant::now();
        let mut messages_pumped: u64 = 0;

        // Main event loop - runs on main thread with Windows message pump
        while self.is_running {
            loop_count += 1;

            // Pump Windows messages (required for hotkeys and tray to work)
            let pumped = Self::pump_messages();
            messages_pumped += pumped as u64;

            // Log status every 10 seconds
            if last_status_log.elapsed() > Duration::from_secs(10) {
                info!("=== MAIN LOOP STATUS ===");
                info!("Loop iterations: {}", loop_count);
                info!("Messages pumped total: {}", messages_pumped);
                info!("Current state: {:?}", self.state);
                info!("Event channel len: {}", self.event_rx.len());
                last_status_log = Instant::now();
            }

            // Check for idle unload
            if self.state == AppState::Idle {
                let idle_duration = last_activity.elapsed();
                if idle_duration > Duration::from_secs(self.config.idle_unload_seconds)
                    && !idle_unload_requested
                {
                    info!("Unloading model after {} seconds idle", idle_duration.as_secs());
                    if let Err(err) = inference_tx.send(InferenceCommand::Unload) {
                        warn!("Failed to request model unload: {}", err);
                    }
                    idle_unload_requested = true;
                }
            }

            while let Ok(entry) = history_update_rx.try_recv() {
                if let Err(err) = self.event_tx.try_send(AppEvent::HistoryUpdated(entry)) {
                    debug!("Dropping history update event: {}", err);
                }
            }

            // Process events (non-blocking)
            match self.event_rx.try_recv() {
                Ok(event) => {
                    info!("=== APP EVENT RECEIVED ===");
                    info!("Event: {:?}", event);
                    info!("Time since last activity: {:?}", last_activity.elapsed());
                    last_activity = Instant::now();
                    idle_unload_requested = false;

                    match event {
                        AppEvent::HotkeyPressed => {
                            info!("Processing HotkeyPressed, current state: {:?}", self.state);
                            match self.state {
                                AppState::Idle => {
                                    info!("Starting recording...");
                                    if let Err(e) = audio.start() {
                                        error!("Failed to start recording: {}", e);
                                        _tray.set_state(AppState::Idle);
                                        continue;
                                    }
                                    self.state = AppState::Recording;
                                    pending_audio_metadata = None;
                                    last_progress_stage = None;
                                    _tray.set_state(AppState::Recording);
                                }
                                AppState::Recording => {
                                    info!("Stopping recording...");
                                    match audio.stop() {
                                        Ok(samples) => {
                                            if samples.is_empty() {
                                                warn!("No audio recorded");
                                                self.state = AppState::Idle;
                                                _tray.set_state(AppState::Idle);
                                                continue;
                                            }

                                            let sample_count = samples.len();
                                            info!("Got {} samples, transcribing...", sample_count);
                                            self.state = AppState::Transcribing;
                                            pending_audio_metadata = Some(AudioMetadata::from_samples(
                                                sample_count,
                                                WHISPER_SAMPLE_RATE,
                                            ));
                                            _tray.set_state(AppState::Transcribing);

                                            if let Err(err) = inference_tx.send(InferenceCommand::Transcribe(samples)) {
                                                error!("Failed to start transcription: {}", err);
                                                self.state = AppState::Idle;
                                                pending_audio_metadata = None;
                                                _tray.set_state(AppState::Idle);
                                            }
                                        }
                                        Err(e) => {
                                            error!("Failed to stop recording: {}", e);
                                            self.state = AppState::Idle;
                                            _tray.set_state(AppState::Idle);
                                        }
                                    }
                                }
                                AppState::Transcribing => {
                                    // Ignore hotkey while transcribing
                                    debug!("Ignoring hotkey while transcribing");
                                }
                            }
                        }
                        AppEvent::InferenceProgress(progress) => {
                            if self.state == AppState::Transcribing {
                                _tray.set_progress(&progress.message);
                            }

                            if last_progress_stage != Some(progress.stage) {
                                notifications.notify(&progress.message);
                                last_progress_stage = Some(progress.stage);
                            }
                        }
                        AppEvent::TranscriptionComplete(result) => {
                            match result {
                                Ok(text) => {
                                    info!("Transcribed: {}", text);

                                    // Format text
                                    let mut formatted = text.trim().to_string();
                                    if self.config.capitalize_first && !formatted.is_empty() {
                                        let mut chars = formatted.chars();
                                        if let Some(first) = chars.next() {
                                            formatted = first.to_uppercase().collect::<String>() + chars.as_str();
                                        }
                                    }
                                    if self.config.add_trailing_space {
                                        formatted.push(' ');
                                    }

                                    let history_text = formatted.trim_end().to_string();
                                    if !history_text.is_empty() {
                                        let metadata = pending_audio_metadata.take();
                                        if let Err(e) = history.append(history_text, metadata) {
                                            error!("Failed to append history: {}", e);
                                        }
                                    }

                                    // Copy to clipboard
                                    if let Err(e) = clipboard.set_text(&formatted) {
                                        error!("Failed to copy to clipboard: {}", e);
                                    } else {
                                        info!("Copied to clipboard!");
                                        notifications.notify("Transcription copied to clipboard");
                                    }
                                }
                                Err(e) => {
                                    error!("Transcription failed: {}", e);
                                    notifications.notify("Transcription failed");
                                }
                            }

                            self.state = AppState::Idle;
                            pending_audio_metadata = None;
                            last_progress_stage = None;
                            _tray.set_state(AppState::Idle);
                        }
                        AppEvent::HistoryUpdated(entry) => {
                            debug!("History updated: {}", entry.id);
                            self.ui.push_history(entry.text);
                        }
                        AppEvent::ToggleHistoryWindow => {
                            self.ui.toggle();
                        }
                        AppEvent::ShowHistoryWindow => {
                            self.ui.show();
                        }
                        AppEvent::Quit => {
                            info!("Quit requested");
                            let _ = inference_tx.send(InferenceCommand::Shutdown);
                            self.is_running = false;
                        }
                    }
                }
                Err(crossbeam_channel::TryRecvError::Empty) => {
                    // No events, sleep briefly to avoid busy-spinning
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    error!("Event channel disconnected");
                    break;
                }
            }
        }

        let _ = inference_tx.send(InferenceCommand::Shutdown);
        info!("Shutting down...");
        Ok(())
    }

    fn spawn_inference_worker(
        event_tx: Sender<AppEvent>,
        config: Config,
    ) -> Sender<InferenceCommand> {
        let (inference_tx, inference_rx) = bounded::<InferenceCommand>(2);
        thread::spawn(move || Self::inference_worker(inference_rx, event_tx, config));
        inference_tx
    }

    fn inference_worker(
        inference_rx: Receiver<InferenceCommand>,
        event_tx: Sender<AppEvent>,
        config: Config,
    ) {
        let mut engine = match WhisperEngine::new_with_config(&config) {
            Ok(engine) => engine,
            Err(err) => {
                let _ = event_tx.send(AppEvent::TranscriptionComplete(Err(format!(
                    "Failed to initialize inference: {}",
                    err
                ))));
                return;
            }
        };

        for command in inference_rx.iter() {
            match command {
                InferenceCommand::Transcribe(samples) => {
                    let mut callback = |progress: InferenceProgress| {
                        let _ = event_tx.send(AppEvent::InferenceProgress(progress));
                    };
                    let result =
                        engine.transcribe_with_progress(&samples, Some(&mut callback));
                    let result = result.map_err(|err| err.to_string());
                    let _ = event_tx.send(AppEvent::TranscriptionComplete(result));
                }
                InferenceCommand::Unload => {
                    engine.unload();
                }
                InferenceCommand::Shutdown => {
                    break;
                }
            }
        }
    }

    /// Pump Windows messages - MUST be called from main thread
    /// Returns the number of messages processed
    #[cfg(target_os = "windows")]
    fn pump_messages() -> u32 {
        use windows_sys::Win32::UI::WindowsAndMessaging::{
            DispatchMessageW, PeekMessageW, TranslateMessage, MSG, PM_REMOVE, PM_NOREMOVE,
            WM_HOTKEY, WM_QUIT, WM_TIMER, WM_NULL,
        };

        let mut count = 0u32;
        unsafe {
            let mut msg: MSG = std::mem::zeroed();

            // First check if there are ANY messages (diagnostic)
            let has_messages = PeekMessageW(&mut msg, 0 as _, 0, 0, PM_NOREMOVE);
            if has_messages != 0 {
                trace!("PeekMessage found messages waiting");
            }

            // Process all pending messages (non-blocking)
            // Use -1 as hwnd to get thread messages (not bound to a window)
            // Actually, 0 should work for all thread messages including window messages
            while PeekMessageW(&mut msg, 0 as _, 0, 0, PM_REMOVE) != 0 {
                count += 1;

                // Log interesting messages (skip noisy ones)
                match msg.message {
                    WM_HOTKEY => {
                        info!("=== WM_HOTKEY MESSAGE RECEIVED ===");
                        info!("hwnd: {:?}", msg.hwnd);
                        info!("wParam (hotkey id): {}", msg.wParam);
                        info!("lParam: {:#x}", msg.lParam);
                    }
                    WM_QUIT => {
                        info!("WM_QUIT received");
                    }
                    WM_TIMER | WM_NULL => {
                        // Skip noisy messages
                    }
                    _ => {
                        // Log other messages at debug level to see what we're getting
                        debug!("Windows message: {} (hwnd: {:?}, wParam: {}, lParam: {})",
                               msg.message, msg.hwnd, msg.wParam, msg.lParam);
                    }
                }

                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }
        count
    }

    #[cfg(not(target_os = "windows"))]
    fn pump_messages() -> u32 {
        // No-op on non-Windows platforms
        0
    }
}
```

## `src/audio/mod.rs`

```rust
//! Audio capture with lock-free ring buffer.

mod capture;
mod processing;
mod ring_buffer;

pub use capture::{AudioCapture, WHISPER_SAMPLE_RATE};
pub use processing::{mono_from_interleaved, resample_linear};
pub use ring_buffer::RingBuffer;
```

## `src/audio/capture.rs`

```rust
//! Audio capture using cpal.

use super::{mono_from_interleaved, resample_linear, RingBuffer};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, SampleRate, Stream, StreamConfig, SupportedStreamConfig};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tracing::{debug, error, info, warn};

/// Target sample rate for Whisper (16kHz)
pub const WHISPER_SAMPLE_RATE: u32 = 16000;

/// Initial buffer capacity.
const INITIAL_BUFFER_SECONDS: usize = 60;
const INITIAL_BUFFER_SIZE: usize = WHISPER_SAMPLE_RATE as usize * INITIAL_BUFFER_SECONDS;

/// Guardrail to avoid runaway memory usage.
const MAX_BUFFER_SECONDS: usize = 600;

pub struct AudioCapture {
    device: Device,
    config: StreamConfig,
    sample_format: SampleFormat,
    buffer: Arc<RingBuffer>,
    stream: Arc<Mutex<Option<Stream>>>,
    is_recording: Arc<AtomicBool>,
}

impl AudioCapture {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let host = cpal::default_host();

        let device = host
            .default_input_device()
            .ok_or("No input device available")?;

        info!("Using audio device: {}", device.name()?);

        // Get supported config, prefer 16kHz mono
        let supported_configs = device.supported_input_configs()?;

        let config = Self::find_best_config(supported_configs)?;
        info!(
            "Audio config: {} Hz, {} channel(s), {:?}",
            config.sample_rate().0,
            config.channels(),
            config.sample_format()
        );

        let buffer = Arc::new(RingBuffer::new(INITIAL_BUFFER_SIZE));
        let is_recording = Arc::new(AtomicBool::new(false));
        let stream = Arc::new(Mutex::new(None));

        Ok(Self {
            device,
            config: config.clone().into(),
            sample_format: config.sample_format(),
            buffer,
            stream,
            is_recording,
        })
    }

    fn find_best_config(
        configs: cpal::SupportedInputConfigs,
    ) -> Result<SupportedStreamConfig, Box<dyn std::error::Error>> {
        // Try to find a config that supports our target sample rate
        let target_rate = SampleRate(WHISPER_SAMPLE_RATE);

        // Collect all configs
        let configs: Vec<_> = configs.collect();

        // First, try to find mono 16kHz
        for config in &configs {
            if config.channels() == 1
                && config.min_sample_rate() <= target_rate
                && config.max_sample_rate() >= target_rate
            {
                return Ok(config.with_sample_rate(target_rate));
            }
        }

        // Try stereo 16kHz (we'll convert to mono)
        for config in &configs {
            if config.min_sample_rate() <= target_rate
                && config.max_sample_rate() >= target_rate
            {
                return Ok(config.with_sample_rate(target_rate));
            }
        }

        // Fall back to any config with highest quality, we'll resample
        if let Some(config) = configs.into_iter().max_by_key(|c| c.max_sample_rate().0) {
            let rate = config.max_sample_rate();
            warn!(
                "Using non-ideal sample rate: {} Hz (will resample)",
                rate.0
            );
            return Ok(config.with_sample_rate(rate));
        }

        Err("No supported audio config found".into())
    }

    /// Start recording
    pub fn start(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.is_recording.load(Ordering::Relaxed) {
            return Ok(()); // Already recording
        }

        // Clear buffer
        self.buffer.clear();

        let buffer = self.buffer.clone();
        let is_recording = self.is_recording.clone();
        let channels = self.config.channels as usize;
        let source_rate = self.config.sample_rate.0;
        let target_rate = WHISPER_SAMPLE_RATE;
        let max_samples = WHISPER_SAMPLE_RATE as usize * MAX_BUFFER_SECONDS;
        let overflowed = Arc::new(AtomicBool::new(false));

        let err_fn = |err| {
            error!("Audio stream error: {}", err);
        };

        // Build the stream
        let stream = match self.sample_format {
            SampleFormat::F32 => {
                let buffer = buffer.clone();
                let is_recording = is_recording.clone();
                let overflowed = overflowed.clone();
                self.device.build_input_stream(
                    &self.config,
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        Self::handle_input(
                            data,
                            channels,
                            source_rate,
                            target_rate,
                            &buffer,
                            &is_recording,
                            max_samples,
                            &overflowed,
                            |sample| sample,
                        );
                    },
                    err_fn,
                    None,
                )?
            }
            SampleFormat::I16 => {
                let buffer = buffer.clone();
                let is_recording = is_recording.clone();
                let overflowed = overflowed.clone();
                self.device.build_input_stream(
                    &self.config,
                    move |data: &[i16], _: &cpal::InputCallbackInfo| {
                        Self::handle_input(
                            data,
                            channels,
                            source_rate,
                            target_rate,
                            &buffer,
                            &is_recording,
                            max_samples,
                            &overflowed,
                            Self::i16_to_f32,
                        );
                    },
                    err_fn,
                    None,
                )?
            }
            SampleFormat::U16 => {
                let buffer = buffer.clone();
                let is_recording = is_recording.clone();
                let overflowed = overflowed.clone();
                self.device.build_input_stream(
                    &self.config,
                    move |data: &[u16], _: &cpal::InputCallbackInfo| {
                        Self::handle_input(
                            data,
                            channels,
                            source_rate,
                            target_rate,
                            &buffer,
                            &is_recording,
                            max_samples,
                            &overflowed,
                            Self::u16_to_f32,
                        );
                    },
                    err_fn,
                    None,
                )?
            }
            other => {
                return Err(format!("Unsupported audio sample format: {:?}", other).into());
            }
        };

        // CRITICAL: Set is_recording BEFORE starting stream
        // Otherwise the callback will ignore samples until this flag is set
        self.is_recording.store(true, Ordering::SeqCst);

        stream.play()?;

        // Store stream so we can stop it later
        *self.stream.lock().unwrap() = Some(stream);

        debug!("Recording started");
        Ok(())
    }

    /// Stop recording and return captured samples
    pub fn stop(&self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        self.is_recording.store(false, Ordering::SeqCst);

        // Small delay to ensure last samples are captured
        std::thread::sleep(std::time::Duration::from_millis(50));

        // Stop and drop the stream
        if let Ok(mut stream_guard) = self.stream.lock() {
            if let Some(stream) = stream_guard.take() {
                // Pause the stream before dropping
                let _ = stream.pause();
                drop(stream);
                debug!("Stream stopped and dropped");
            }
        }

        let samples = self.buffer.pop_all();
        debug!("Recording stopped, got {} samples", samples.len());

        Ok(samples)
    }

    /// Check if currently recording
    pub fn is_recording(&self) -> bool {
        self.is_recording.load(Ordering::Relaxed)
    }

    fn handle_input<T, F>(
        data: &[T],
        channels: usize,
        source_rate: u32,
        target_rate: u32,
        buffer: &RingBuffer,
        is_recording: &AtomicBool,
        max_samples: usize,
        overflowed: &AtomicBool,
        to_f32: F,
    ) where
        T: Copy,
        F: Fn(T) -> f32 + Copy,
    {
        if !is_recording.load(Ordering::Relaxed) {
            return;
        }

        let mono = mono_from_interleaved(data, channels, to_f32);
        let samples = if source_rate != target_rate {
            resample_linear(&mono, source_rate, target_rate)
        } else {
            mono
        };

        let current_len = buffer.len();
        if current_len >= max_samples {
            Self::mark_overflow(is_recording, overflowed, max_samples);
            return;
        }

        let remaining = max_samples - current_len;
        if samples.len() > remaining {
            buffer.push(&samples[..remaining]);
            Self::mark_overflow(is_recording, overflowed, max_samples);
            return;
        }

        buffer.push(&samples);
    }

    fn mark_overflow(is_recording: &AtomicBool, overflowed: &AtomicBool, max_samples: usize) {
        if !overflowed.swap(true, Ordering::Relaxed) {
            warn!(
                "Reached max recording duration ({:.1}s); pausing capture",
                max_samples as f32 / WHISPER_SAMPLE_RATE as f32
            );
        }
        is_recording.store(false, Ordering::SeqCst);
    }

    fn i16_to_f32(sample: i16) -> f32 {
        sample as f32 / 32768.0
    }

    fn u16_to_f32(sample: u16) -> f32 {
        (sample as f32 - 32768.0) / 32768.0
    }
}
```

## `src/audio/processing.rs`

```rust
//! Shared audio processing helpers.

/// Convert interleaved samples into mono f32 samples.
pub fn mono_from_interleaved<T, F>(data: &[T], channels: usize, to_f32: F) -> Vec<f32>
where
    T: Copy,
    F: Fn(T) -> f32 + Copy,
{
    if data.is_empty() || channels == 0 {
        return Vec::new();
    }

    if channels == 1 {
        return data.iter().map(|&sample| to_f32(sample)).collect();
    }

    data.chunks(channels)
        .map(|frame| {
            let sum: f32 = frame.iter().map(|&sample| to_f32(sample)).sum();
            sum / channels as f32
        })
        .collect()
}

/// Simple linear resampling from source_rate to target_rate.
pub fn resample_linear(samples: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if source_rate == target_rate {
        return samples.to_vec();
    }

    if samples.is_empty() || source_rate == 0 || target_rate == 0 {
        return Vec::new();
    }

    let ratio = source_rate as f64 / target_rate as f64;
    let new_len = (samples.len() as f64 / ratio) as usize;
    let mut output = Vec::with_capacity(new_len);

    for i in 0..new_len {
        let src_idx = i as f64 * ratio;
        let idx = src_idx as usize;
        let frac = src_idx - idx as f64;

        let sample = if idx + 1 < samples.len() {
            samples[idx] * (1.0 - frac as f32) + samples[idx + 1] * frac as f32
        } else if idx < samples.len() {
            samples[idx]
        } else {
            0.0
        };

        output.push(sample);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::{mono_from_interleaved, resample_linear};

    #[test]
    fn mono_from_interleaved_stereo() {
        let input = [0.0_f32, 1.0_f32, 2.0_f32, 3.0_f32];
        let mono = mono_from_interleaved(&input, 2, |v| v);
        assert_eq!(mono, vec![0.5, 2.5]);
    }

    #[test]
    fn resample_linear_upsample() {
        let input = [0.0_f32, 1.0_f32];
        let output = resample_linear(&input, 2, 4);
        assert_eq!(output.len(), 4);
        assert!((output[1] - 0.5).abs() < 1e-6);
        assert!((output[3] - 1.0).abs() < 1e-6);
    }
}
```

## `src/audio/ring_buffer.rs`

```rust
//! Growable audio buffer for unlimited recording length.

use std::sync::Mutex;

/// Thread-safe growable buffer for audio samples (unlimited length)
pub struct RingBuffer {
    samples: Mutex<Vec<f32>>,
}

impl RingBuffer {
    /// Create a new growable buffer
    /// The capacity parameter is used as initial capacity hint
    pub fn new(initial_capacity: usize) -> Self {
        Self {
            samples: Mutex::new(Vec::with_capacity(initial_capacity)),
        }
    }

    /// Push samples into the buffer (called from audio callback)
    /// Returns the number of samples pushed (always all of them)
    pub fn push(&self, new_samples: &[f32]) -> usize {
        if let Ok(mut samples) = self.samples.lock() {
            samples.extend_from_slice(new_samples);
            new_samples.len()
        } else {
            0
        }
    }

    /// Pop all available samples from the buffer
    pub fn pop_all(&self) -> Vec<f32> {
        if let Ok(mut samples) = self.samples.lock() {
            std::mem::take(&mut *samples)
        } else {
            Vec::new()
        }
    }

    /// Clear the buffer
    pub fn clear(&self) {
        if let Ok(mut samples) = self.samples.lock() {
            samples.clear();
        }
    }

    /// Get the number of samples available
    pub fn len(&self) -> usize {
        if let Ok(samples) = self.samples.lock() {
            samples.len()
        } else {
            0
        }
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get recording duration in seconds
    pub fn duration_secs(&self, sample_rate: u32) -> f32 {
        self.len() as f32 / sample_rate as f32
    }
}
```

## `src/config.rs`

```rust
//! Configuration management with sensible defaults.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use directories::ProjectDirs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Hotkey to toggle recording (e.g., "F4", "Ctrl+Alt+V")
    pub hotkey: String,

    /// Hotkey to toggle the history window
    #[serde(default = "default_history_hotkey")]
    pub history_hotkey: String,

    /// Language for transcription (None = auto-detect)
    /// Note: Parakeet is English-only, this is for future multi-language support
    pub language: Option<String>,

    /// Add trailing space after pasted text
    pub add_trailing_space: bool,

    /// Capitalize first letter
    pub capitalize_first: bool,

    /// Unload model after N seconds idle (saves VRAM)
    pub idle_unload_seconds: u64,

    /// Show notifications
    pub show_notifications: bool,

    /// Maximum number of history entries to retain
    #[serde(default = "default_history_max_entries")]
    pub history_max_entries: usize,

    /// Enable GPU acceleration when available
    #[serde(default = "default_use_gpu")]
    pub use_gpu: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            hotkey: "F4".to_string(),
            history_hotkey: default_history_hotkey(),
            language: None, // Parakeet is English-only for now
            add_trailing_space: true,
            capitalize_first: true,
            idle_unload_seconds: 300, // 5 minutes
            show_notifications: true,
            history_max_entries: default_history_max_entries(),
            use_gpu: true,
        }
    }
}

impl Config {
    /// Get the config directory
    pub fn config_dir() -> Option<PathBuf> {
        ProjectDirs::from("com", "voclaude", "Voclaude")
            .map(|dirs| dirs.config_dir().to_path_buf())
    }

    /// Get the models directory
    pub fn models_dir() -> Option<PathBuf> {
        ProjectDirs::from("com", "voclaude", "Voclaude")
            .map(|dirs| dirs.data_dir().join("models"))
    }

    /// Get the config file path
    pub fn config_path() -> Option<PathBuf> {
        Self::config_dir().map(|dir| dir.join("config.toml"))
    }

    /// Load config from disk, or create default
    pub fn load() -> Result<Self, Box<dyn std::error::Error>> {
        let path = Self::config_path().ok_or("Could not determine config path")?;

        if path.exists() {
            let contents = std::fs::read_to_string(&path)?;
            let config: Config = toml::from_str(&contents)?;
            Ok(config)
        } else {
            let config = Config::default();
            config.save()?;
            Ok(config)
        }
    }

    /// Save config to disk
    pub fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        let path = Self::config_path().ok_or("Could not determine config path")?;

        // Ensure directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let contents = toml::to_string_pretty(self)?;
        std::fs::write(&path, contents)?;
        Ok(())
    }
}

fn default_use_gpu() -> bool {
    true
}

fn default_history_max_entries() -> usize {
    500
}

fn default_history_hotkey() -> String {
    "Ctrl+Shift+H".to_string()
}
```

## `src/history.rs`

```rust
//! Transcription history storage and retention.

use crossbeam_channel::Sender;
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, warn};

static ENTRY_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioMetadata {
    pub sample_rate: u32,
    pub sample_count: usize,
    pub duration_ms: u64,
}

impl AudioMetadata {
    pub fn from_samples(sample_count: usize, sample_rate: u32) -> Self {
        let duration_ms = (sample_count as u64 * 1000) / sample_rate as u64;
        Self {
            sample_rate,
            sample_count,
            duration_ms,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub id: String,
    pub created_at_ms: u64,
    pub text: String,
    pub audio: Option<AudioMetadata>,
}

impl HistoryEntry {
    fn new(text: String, audio: Option<AudioMetadata>) -> Self {
        let created_at_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_millis() as u64)
            .unwrap_or_default();
        let counter = ENTRY_COUNTER.fetch_add(1, Ordering::Relaxed);
        let id = format!("{}-{}", created_at_ms, counter);
        Self {
            id,
            created_at_ms,
            text,
            audio,
        }
    }
}

#[derive(Debug)]
pub struct HistoryStore {
    entries: Vec<HistoryEntry>,
    path: PathBuf,
    max_entries: usize,
    update_tx: Sender<HistoryEntry>,
}

impl HistoryStore {
    pub fn load(
        max_entries: usize,
        update_tx: Sender<HistoryEntry>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let path = Self::history_path()?;
        let entries = if path.exists() {
            match std::fs::read_to_string(&path) {
                Ok(contents) => match serde_json::from_str::<Vec<HistoryEntry>>(&contents) {
                    Ok(entries) => entries,
                    Err(err) => {
                        warn!("Failed to parse history file: {}", err);
                        Vec::new()
                    }
                },
                Err(err) => {
                    warn!("Failed to read history file: {}", err);
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        };

        let mut store = Self {
            entries,
            path,
            max_entries: max_entries.max(1),
            update_tx,
        };
        store.apply_retention();
        store.persist()?;
        Ok(store)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn entries(&self) -> &[HistoryEntry] {
        &self.entries
    }

    pub fn append(
        &mut self,
        text: String,
        audio: Option<AudioMetadata>,
    ) -> Result<HistoryEntry, Box<dyn std::error::Error>> {
        let entry = HistoryEntry::new(text, audio);
        self.entries.push(entry.clone());
        self.apply_retention();
        self.persist()?;
        if let Err(err) = self.update_tx.try_send(entry.clone()) {
            debug!("Dropping history update: {}", err);
        }
        Ok(entry)
    }

    fn apply_retention(&mut self) {
        if self.entries.len() > self.max_entries {
            let excess = self.entries.len() - self.max_entries;
            self.entries.drain(0..excess);
        }
    }

    fn persist(&self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let contents = serde_json::to_string_pretty(&self.entries)?;
        std::fs::write(&self.path, contents)?;
        Ok(())
    }

    fn history_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
        ProjectDirs::from("com", "voclaude", "Voclaude")
            .map(|dirs| dirs.data_dir().join("history.json"))
            .ok_or_else(|| "Could not determine history path".into())
    }
}
```

## `src/hotkey.rs`

```rust
//! Global hotkey registration and handling.

use crate::app::AppEvent;

use crossbeam_channel::Sender;
use global_hotkey::{
    hotkey::{Code, HotKey, Modifiers},
    GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState,
};
use std::time::Duration;
use tracing::{debug, error, info, warn, trace};

pub struct HotkeyManager {
    manager: GlobalHotKeyManager,
    hotkey: HotKey,
}

impl HotkeyManager {
    pub fn new(
        hotkey_str: &str,
        app_event: AppEvent,
        event_tx: Sender<AppEvent>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        info!("=== HOTKEY INITIALIZATION START ===");

        // Log Windows thread ID
        #[cfg(target_os = "windows")]
        {
            #[link(name = "kernel32")]
            extern "system" {
                fn GetCurrentThreadId() -> u32;
            }
            let win_thread_id = unsafe { GetCurrentThreadId() };
            info!("Hotkey init on Windows Thread ID: {}", win_thread_id);
        }

        info!("Creating GlobalHotKeyManager...");

        let manager = GlobalHotKeyManager::new()?;
        info!("GlobalHotKeyManager created successfully");

        // Parse hotkey string (e.g., "Super+C", "Ctrl+Shift+Space")
        info!("Parsing hotkey string: '{}'", hotkey_str);
        let hotkey = Self::parse_hotkey(hotkey_str)?;
        info!("Parsed hotkey - ID: {}", hotkey.id());

        // Register the hotkey
        info!("Registering hotkey with Windows...");
        match manager.register(hotkey) {
            Ok(_) => info!("Hotkey registered successfully with Windows!"),
            Err(e) => {
                error!("FAILED to register hotkey: {:?}", e);
                return Err(e.into());
            }
        }

        // Spawn event handler thread
        let hotkey_id = hotkey.id();
        let event_to_send = app_event.clone();
        info!("Hotkey ID: {}, spawning listener thread...", hotkey_id);

        std::thread::spawn(move || {
            info!("=== HOTKEY LISTENER THREAD STARTED ===");
            info!("Thread ID: {:?}", std::thread::current().id());
            info!("Getting GlobalHotKeyEvent receiver...");

            let receiver = GlobalHotKeyEvent::receiver();
            info!("Got receiver, entering event loop...");

            let mut loop_count: u64 = 0;
            let mut last_log = std::time::Instant::now();

            loop {
                loop_count += 1;

                // Log every 10 seconds to show the thread is alive
                if last_log.elapsed() > Duration::from_secs(10) {
                    info!("Hotkey listener alive - {} iterations, waiting for events...", loop_count);
                    last_log = std::time::Instant::now();
                }

                // Use recv_timeout instead of blocking recv for better diagnostics
                match receiver.recv_timeout(Duration::from_millis(100)) {
                    Ok(event) => {
                        info!("=== HOTKEY EVENT RECEIVED ===");
                        info!("Event ID: {}, Expected ID: {}", event.id, hotkey_id);
                        info!("Event state: {:?}", event.state);
                        info!("IDs match: {}", event.id == hotkey_id);

                        // Only trigger on key press, not release
                        if event.state == HotKeyState::Pressed {
                            info!("Event is PRESSED state");
                            if event.id == hotkey_id {
                                info!("IDs MATCH! Sending hotkey event...");
                                match event_tx.send(event_to_send.clone()) {
                                    Ok(_) => info!("Hotkey event sent successfully!"),
                                    Err(e) => error!("FAILED to send hotkey event: {}", e),
                                }
                            } else {
                                warn!("Event ID {} does not match expected {}", event.id, hotkey_id);
                            }
                        } else {
                            debug!("Ignoring non-pressed state: {:?}", event.state);
                        }
                    }
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                        // Normal timeout, continue loop
                        trace!("Receiver timeout (normal)");
                    }
                    Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                        error!("Hotkey receiver DISCONNECTED! Exiting thread.");
                        break;
                    }
                }
            }
            error!("Hotkey listener thread exiting!");
        });

        info!("=== HOTKEY INITIALIZATION COMPLETE ===");
        Ok(Self { manager, hotkey })
    }

    fn parse_hotkey(s: &str) -> Result<HotKey, Box<dyn std::error::Error>> {
        let parts: Vec<&str> = s.split('+').map(|p| p.trim()).collect();

        let mut modifiers = Modifiers::empty();
        let mut key_code = None;

        for part in parts {
            match part.to_lowercase().as_str() {
                // Modifiers
                "ctrl" | "control" => modifiers |= Modifiers::CONTROL,
                "alt" => modifiers |= Modifiers::ALT,
                "shift" => modifiers |= Modifiers::SHIFT,
                "super" | "win" | "cmd" | "meta" => modifiers |= Modifiers::SUPER,

                // Letter keys
                "a" => key_code = Some(Code::KeyA),
                "b" => key_code = Some(Code::KeyB),
                "c" => key_code = Some(Code::KeyC),
                "d" => key_code = Some(Code::KeyD),
                "e" => key_code = Some(Code::KeyE),
                "f" => key_code = Some(Code::KeyF),
                "g" => key_code = Some(Code::KeyG),
                "h" => key_code = Some(Code::KeyH),
                "i" => key_code = Some(Code::KeyI),
                "j" => key_code = Some(Code::KeyJ),
                "k" => key_code = Some(Code::KeyK),
                "l" => key_code = Some(Code::KeyL),
                "m" => key_code = Some(Code::KeyM),
                "n" => key_code = Some(Code::KeyN),
                "o" => key_code = Some(Code::KeyO),
                "p" => key_code = Some(Code::KeyP),
                "q" => key_code = Some(Code::KeyQ),
                "r" => key_code = Some(Code::KeyR),
                "s" => key_code = Some(Code::KeyS),
                "t" => key_code = Some(Code::KeyT),
                "u" => key_code = Some(Code::KeyU),
                "v" => key_code = Some(Code::KeyV),
                "w" => key_code = Some(Code::KeyW),
                "x" => key_code = Some(Code::KeyX),
                "y" => key_code = Some(Code::KeyY),
                "z" => key_code = Some(Code::KeyZ),

                // Number keys
                "0" => key_code = Some(Code::Digit0),
                "1" => key_code = Some(Code::Digit1),
                "2" => key_code = Some(Code::Digit2),
                "3" => key_code = Some(Code::Digit3),
                "4" => key_code = Some(Code::Digit4),
                "5" => key_code = Some(Code::Digit5),
                "6" => key_code = Some(Code::Digit6),
                "7" => key_code = Some(Code::Digit7),
                "8" => key_code = Some(Code::Digit8),
                "9" => key_code = Some(Code::Digit9),

                // Function keys
                "f1" => key_code = Some(Code::F1),
                "f2" => key_code = Some(Code::F2),
                "f3" => key_code = Some(Code::F3),
                "f4" => key_code = Some(Code::F4),
                "f5" => key_code = Some(Code::F5),
                "f6" => key_code = Some(Code::F6),
                "f7" => key_code = Some(Code::F7),
                "f8" => key_code = Some(Code::F8),
                "f9" => key_code = Some(Code::F9),
                "f10" => key_code = Some(Code::F10),
                "f11" => key_code = Some(Code::F11),
                "f12" => key_code = Some(Code::F12),

                // Special keys
                "space" => key_code = Some(Code::Space),
                "enter" | "return" => key_code = Some(Code::Enter),
                "tab" => key_code = Some(Code::Tab),
                "backspace" => key_code = Some(Code::Backspace),
                "delete" => key_code = Some(Code::Delete),
                "escape" | "esc" => key_code = Some(Code::Escape),
                "home" => key_code = Some(Code::Home),
                "end" => key_code = Some(Code::End),
                "pageup" => key_code = Some(Code::PageUp),
                "pagedown" => key_code = Some(Code::PageDown),

                // Numpad keys
                "numpad0" | "num0" => key_code = Some(Code::Numpad0),
                "numpad1" | "num1" => key_code = Some(Code::Numpad1),
                "numpad2" | "num2" => key_code = Some(Code::Numpad2),
                "numpad3" | "num3" => key_code = Some(Code::Numpad3),
                "numpad4" | "num4" => key_code = Some(Code::Numpad4),
                "numpad5" | "num5" => key_code = Some(Code::Numpad5),
                "numpad6" | "num6" => key_code = Some(Code::Numpad6),
                "numpad7" | "num7" => key_code = Some(Code::Numpad7),
                "numpad8" | "num8" => key_code = Some(Code::Numpad8),
                "numpad9" | "num9" => key_code = Some(Code::Numpad9),
                "numpadadd" | "numadd" => key_code = Some(Code::NumpadAdd),
                "numpadsubtract" | "numsub" => key_code = Some(Code::NumpadSubtract),
                "numpadmultiply" | "nummul" => key_code = Some(Code::NumpadMultiply),
                "numpaddivide" | "numdiv" => key_code = Some(Code::NumpadDivide),
                "numpaddecimal" | "numdec" => key_code = Some(Code::NumpadDecimal),
                "numpadenter" | "numenter" => key_code = Some(Code::NumpadEnter),

                // Punctuation
                ";" | "semicolon" => key_code = Some(Code::Semicolon),
                "'" | "quote" => key_code = Some(Code::Quote),
                "," | "comma" => key_code = Some(Code::Comma),
                "." | "period" => key_code = Some(Code::Period),
                "/" | "slash" => key_code = Some(Code::Slash),
                "`" | "backquote" => key_code = Some(Code::Backquote),
                "[" | "bracketleft" => key_code = Some(Code::BracketLeft),
                "]" | "bracketright" => key_code = Some(Code::BracketRight),
                "\\\\" | "backslash" => key_code = Some(Code::Backslash),
                "-" | "minus" => key_code = Some(Code::Minus),
                "=" | "equal" => key_code = Some(Code::Equal),

                other => {
                    return Err(format!("Unknown key: {}", other).into());
                }
            }
        }

        let code = key_code.ok_or("No key specified in hotkey")?;
        Ok(HotKey::new(Some(modifiers), code))
    }
}

impl Drop for HotkeyManager {
    fn drop(&mut self) {
        if let Err(e) = self.manager.unregister(self.hotkey) {
            error!("Failed to unregister hotkey: {}", e);
        }
    }
}
```

## `src/inference/mod.rs`

```rust
//! Whisper speech-to-text inference.

mod whisper;

pub use whisper::WhisperEngine;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceStage {
    DownloadingModel,
    LoadingModel,
    Transcribing,
}

#[derive(Debug, Clone)]
pub struct InferenceProgress {
    pub stage: InferenceStage,
    pub message: String,
    pub percent: Option<u8>,
}
```

## `src/inference/whisper.rs`

```rust
//! Whisper speech-to-text inference engine.
//!
//! Uses whisper-rs (whisper.cpp bindings) for fast, accurate transcription.

use crate::config::Config;
use crate::inference::{InferenceProgress, InferenceStage};
use std::fs;
use std::path::PathBuf;
use tracing::{debug, info, warn};
use whisper_rs::{WhisperContext, WhisperContextParameters, FullParams, SamplingStrategy};

/// Whisper model size
#[derive(Debug, Clone, Copy)]
pub enum WhisperModel {
    Medium,  // ~1.5GB, good balance of speed/quality
}

impl WhisperModel {
    fn filename(&self) -> &'static str {
        match self {
            WhisperModel::Medium => "ggml-medium.bin",
        }
    }

    fn url(&self) -> &'static str {
        match self {
            WhisperModel::Medium => "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
        }
    }

    fn size_mb(&self) -> u64 {
        match self {
            WhisperModel::Medium => 1533,
        }
    }
}

/// Whisper inference engine
pub struct WhisperEngine {
    context: Option<WhisperContext>,
    model: WhisperModel,
    is_loaded: bool,
    use_gpu: bool,
}

impl WhisperEngine {
    /// Create a new Whisper engine (lazy loading)
    pub fn new(use_gpu: bool) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            context: None,
            model: WhisperModel::Medium,
            is_loaded: false,
            use_gpu,
        })
    }

    /// Create a new Whisper engine with config
    pub fn new_with_config(config: &Config) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new(config.use_gpu)
    }

    /// Check if model is loaded
    pub fn is_loaded(&self) -> bool {
        self.is_loaded
    }

    /// Unload model to free memory
    pub fn unload(&mut self) {
        if self.is_loaded {
            info!("Unloading Whisper model");
            self.context = None;
            self.is_loaded = false;
        }
    }

    /// Get models directory
    fn models_dir() -> Option<PathBuf> {
        Config::models_dir()
    }

    /// Ensure model is downloaded
    fn ensure_model(
        &self,
        progress: &mut Option<&mut dyn FnMut(InferenceProgress)>,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let models_dir = Self::models_dir().ok_or("Could not determine models directory")?;
        fs::create_dir_all(&models_dir)?;

        let model_path = models_dir.join(self.model.filename());

        if model_path.exists() {
            info!("Model already downloaded: {}", model_path.display());
            return Ok(model_path);
        }

        let download_message = format!(
            "Downloading Whisper {} model (~{}MB)...",
            format!("{:?}", self.model).to_lowercase(),
            self.model.size_mb()
        );
        info!("{}", download_message);
        if let Some(cb) = progress.as_deref_mut() {
            cb(InferenceProgress {
                stage: InferenceStage::DownloadingModel,
                message: download_message,
                percent: None,
            });
        }

        Self::download_file(self.model.url(), &model_path, progress)?;

        info!("Model downloaded: {}", model_path.display());
        Ok(model_path)
    }

    /// Download a file with progress
    fn download_file(
        url: &str,
        dest: &PathBuf,
        progress: &mut Option<&mut dyn FnMut(InferenceProgress)>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use std::io::{Read, Write};

        info!("Downloading: {}", url);

        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(7200))
            .build()?;

        let mut response = client.get(url).send()?;

        if !response.status().is_success() {
            return Err(format!("Download failed: HTTP {}", response.status()).into());
        }

        let total_size = response.content_length();
        if let Some(size) = total_size {
            info!("Download size: {:.1} MB", size as f64 / 1024.0 / 1024.0);
        }

        let temp_path = dest.with_extension("tmp");
        let mut file = fs::File::create(&temp_path)?;

        let mut downloaded: u64 = 0;
        let mut last_progress = 0;
        let mut buffer = [0u8; 131072];

        loop {
            let bytes_read = response.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            file.write_all(&buffer[..bytes_read])?;
            downloaded += bytes_read as u64;

            if let Some(total) = total_size {
                let progress_pct = (downloaded * 100 / total) as u32;
                if progress_pct >= last_progress + 5 {
                    info!("Download progress: {}% ({:.1} MB / {:.1} MB)",
                          progress_pct,
                          downloaded as f64 / 1024.0 / 1024.0,
                          total as f64 / 1024.0 / 1024.0);
                    last_progress = progress_pct;
                    if let Some(cb) = progress.as_deref_mut() {
                        cb(InferenceProgress {
                            stage: InferenceStage::DownloadingModel,
                            message: format!(
                                "Downloading model... {}% ({:.1}/{:.1} MB)",
                                progress_pct,
                                downloaded as f64 / 1024.0 / 1024.0,
                                total as f64 / 1024.0 / 1024.0
                            ),
                            percent: Some(progress_pct as u8),
                        });
                    }
                }
            }
        }

        file.flush()?;
        drop(file);
        fs::rename(&temp_path, dest)?;

        info!("Download complete: {}", dest.display());
        Ok(())
    }

    /// Ensure model is loaded
    fn ensure_loaded(
        &mut self,
        progress: &mut Option<&mut dyn FnMut(InferenceProgress)>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.is_loaded {
            return Ok(());
        }

        info!("Loading Whisper model...");

        let model_path = self.ensure_model(progress)?;
        if let Some(cb) = progress.as_deref_mut() {
            cb(InferenceProgress {
                stage: InferenceStage::LoadingModel,
                message: "Loading Whisper model...".to_string(),
                percent: None,
            });
        }

        let mut prefer_gpu = self.use_gpu;
        if prefer_gpu && !cfg!(feature = "cuda") {
            info!("GPU requested but CUDA feature is disabled; falling back to CPU");
            prefer_gpu = false;
        }

        let mut params = WhisperContextParameters::default();
        params.use_gpu(prefer_gpu);

        let ctx = match WhisperContext::new_with_params(
            model_path.to_str().ok_or("Invalid model path")?,
            params,
        ) {
            Ok(ctx) => ctx,
            Err(e) if prefer_gpu => {
                warn!("GPU init failed ({}); retrying on CPU", e);
                let mut cpu_params = WhisperContextParameters::default();
                cpu_params.use_gpu(false);
                WhisperContext::new_with_params(
                    model_path.to_str().ok_or("Invalid model path")?,
                    cpu_params,
                ).map_err(|e| format!("Failed to load Whisper model on CPU: {}", e))?
            }
            Err(e) => return Err(format!("Failed to load Whisper model: {}", e).into()),
        };

        self.context = Some(ctx);
        self.is_loaded = true;

        info!("Whisper model loaded!");
        Ok(())
    }

    /// Transcribe audio samples
    /// Input: f32 samples at 16kHz mono
    /// Output: transcribed text
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<String, Box<dyn std::error::Error>> {
        self.transcribe_with_progress(samples, None)
    }

    pub fn transcribe_with_progress(
        &mut self,
        samples: &[f32],
        mut progress: Option<&mut dyn FnMut(InferenceProgress)>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.ensure_loaded(&mut progress)?;

        let ctx = self.context.as_mut().ok_or("Whisper context not loaded")?;

        debug!("Transcribing {} samples ({:.2}s)",
               samples.len(),
               samples.len() as f32 / 16000.0);

        // Create state for this transcription
        let mut state = ctx.create_state()
            .map_err(|e| format!("Failed to create Whisper state: {}", e))?;

        // Configure transcription parameters
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

        // Set language to English for faster processing
        params.set_language(Some("en"));

        // Disable translation (we want transcription)
        params.set_translate(false);

        // Single segment mode for voice input
        params.set_single_segment(false);

        // Print progress
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        // Suppress non-speech tokens
        params.set_suppress_blank(true);
        params.set_suppress_nst(true);

        // Run inference
        if let Some(cb) = progress.as_deref_mut() {
            cb(InferenceProgress {
                stage: InferenceStage::Transcribing,
                message: "Transcribing audio...".to_string(),
                percent: None,
            });
        }
        let start = std::time::Instant::now();
        state.full(params, samples)
            .map_err(|e| format!("Whisper inference failed: {}", e))?;
        let elapsed = start.elapsed();

        debug!("Inference took {:.2}s", elapsed.as_secs_f32());

        // Collect all segments
        let num_segments = state.full_n_segments()
            .map_err(|e| format!("Failed to get segments: {}", e))?;

        let mut text = String::new();
        for i in 0..num_segments {
            if let Ok(segment) = state.full_get_segment_text(i) {
                text.push_str(&segment);
            }
        }

        let text = text.trim().to_string();
        debug!("Transcription: {}", text);

        Ok(text)
    }
}

impl Default for WhisperEngine {
    fn default() -> Self {
        Self::new(true).expect("Failed to create WhisperEngine")
    }
}
```

## `src/tray.rs`

```rust
//! System tray icon with menu.

use crate::app::{AppEvent, AppState};

use crossbeam_channel::Sender;
use tray_icon::{
    menu::{Menu, MenuEvent, MenuItem, PredefinedMenuItem},
    TrayIcon, TrayIconBuilder,
    Icon,
};
use std::sync::{Arc, Mutex};
use tracing::{debug, error};

/// Embedded icons (will be replaced with actual assets)
const ICON_IDLE: &[u8] = include_bytes!("../assets/icon_idle.png");
const ICON_RECORDING: &[u8] = include_bytes!("../assets/icon_recording.png");
const ICON_PROCESSING: &[u8] = include_bytes!("../assets/icon_processing.png");

/// Menu item IDs
const MENU_TOGGLE: &str = "toggle";
const MENU_HISTORY: &str = "history";
const MENU_SETTINGS: &str = "settings";
const MENU_QUIT: &str = "quit";

pub struct TrayManager {
    tray: TrayIcon,
    event_tx: Sender<AppEvent>,
    toggle_item: MenuItem,
    state: Arc<Mutex<AppState>>,
    status_detail: Arc<Mutex<Option<String>>>,
}

impl TrayManager {
    pub fn new(event_tx: Sender<AppEvent>) -> Result<Self, Box<dyn std::error::Error>> {
        // Load icons
        let icon_idle = Self::load_icon(ICON_IDLE)?;

        // Create menu items
        let toggle_item = MenuItem::with_id(MENU_TOGGLE, "Start Recording", true, None);
        let history_item = MenuItem::with_id(MENU_HISTORY, "Show History", true, None);
        let settings_item = MenuItem::with_id(MENU_SETTINGS, "Settings...", true, None);
        let separator = PredefinedMenuItem::separator();
        let quit_item = MenuItem::with_id(MENU_QUIT, "Quit Voclaude", true, None);

        // Build menu
        let menu = Menu::new();
        menu.append(&toggle_item)?;
        menu.append(&history_item)?;
        menu.append(&separator)?;
        menu.append(&settings_item)?;
        menu.append(&PredefinedMenuItem::separator())?;
        menu.append(&quit_item)?;

        // Build tray icon
        let tray = TrayIconBuilder::new()
            .with_tooltip("Voclaude - Ready")
            .with_icon(icon_idle)
            .with_menu(Box::new(menu))
            .build()?;

        let state = Arc::new(Mutex::new(AppState::Idle));
        let status_detail = Arc::new(Mutex::new(None));

        // Spawn menu event handler
        let event_tx_clone = event_tx.clone();
        let _state_clone = state.clone();
        std::thread::spawn(move || {
            let receiver = MenuEvent::receiver();
            loop {
                if let Ok(event) = receiver.recv() {
                    match event.id.0.as_str() {
                        MENU_TOGGLE => {
                            debug!("Toggle recording clicked");
                            let _ = event_tx_clone.send(AppEvent::HotkeyPressed);
                        }
                        MENU_SETTINGS => {
                            debug!("Settings clicked");
                            // TODO: Open settings window
                        }
                        MENU_HISTORY => {
                            debug!("History clicked");
                            let _ = event_tx_clone.send(AppEvent::ShowHistoryWindow);
                        }
                        MENU_QUIT => {
                            debug!("Quit clicked");
                            let _ = event_tx_clone.send(AppEvent::Quit);
                        }
                        _ => {}
                    }
                }
            }
        });

        Ok(Self {
            tray,
            event_tx,
            toggle_item,
            state,
            status_detail,
        })
    }

    fn load_icon(data: &[u8]) -> Result<Icon, Box<dyn std::error::Error>> {
        let img = image::load_from_memory(data)?;
        let rgba = img.to_rgba8();
        let (width, height) = rgba.dimensions();
        let icon = Icon::from_rgba(rgba.into_raw(), width, height)?;
        Ok(icon)
    }

    /// Update tray state (icon + menu text)
    pub fn set_state(&self, state: AppState) {
        // Update internal state
        if let Ok(mut s) = self.state.lock() {
            *s = state;
        }

        if state != AppState::Transcribing {
            if let Ok(mut detail) = self.status_detail.lock() {
                *detail = None;
            }
        }

        // Update icon and tooltip
        let detail = self
            .status_detail
            .lock()
            .ok()
            .and_then(|detail| detail.clone());
        let (icon_data, tooltip, menu_text) = match state {
            AppState::Idle => (
                ICON_IDLE,
                "Voclaude - Ready".to_string(),
                "Start Recording",
            ),
            AppState::Recording => (
                ICON_RECORDING,
                "Voclaude - Recording...".to_string(),
                "Stop Recording",
            ),
            AppState::Transcribing => (
                ICON_PROCESSING,
                detail
                    .map(|message| format!("Voclaude - {}", message))
                    .unwrap_or_else(|| "Voclaude - Processing...".to_string()),
                "Processing...",
            ),
        };

        // Update icon
        if let Ok(icon) = Self::load_icon(icon_data) {
            if let Err(e) = self.tray.set_icon(Some(icon)) {
                error!("Failed to set tray icon: {}", e);
            }
        }

        // Update tooltip
        if let Err(e) = self.tray.set_tooltip(Some(&tooltip)) {
            error!("Failed to set tooltip: {}", e);
        }

        // Update menu item text
        self.toggle_item.set_text(menu_text);

        // Disable toggle during transcription
        self.toggle_item.set_enabled(state != AppState::Transcribing);
    }

    pub fn set_progress(&self, message: &str) {
        if let Ok(mut detail) = self.status_detail.lock() {
            *detail = Some(message.to_string());
        }

        let state = self.state.lock().ok().map(|state| *state).unwrap_or(AppState::Idle);
        if state == AppState::Transcribing {
            let tooltip = format!("Voclaude - {}", message);
            if let Err(e) = self.tray.set_tooltip(Some(&tooltip)) {
                error!("Failed to set tooltip: {}", e);
            }
        }
    }
}
```

## `src/ui.rs`

```rust
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
```

---

## Unwired / Experimental Files (Tracked But Not Compiled)

The following files are tracked, but `src/inference/mod.rs` does not `mod` them, and `Cargo.toml` does not list their dependencies. They are included here because they may represent future/experimental work.

### `src/inference/model_manager.rs`

```rust
//! Model downloading and management for Parakeet TDT ONNX.
//!
//! Uses sherpa-onnx's well-tested ONNX export with encoder + decoder + joiner format.

use crate::config::Config;

use bzip2::read::BzDecoder;
use std::fs::{self, File};
use std::io::{Read, Write, BufReader};
use std::path::PathBuf;
use tar::Archive;
use tracing::info;

/// sherpa-onnx Parakeet TDT 0.6B v2 int8 quantized (encoder + decoder + joiner format)
const MODEL_TAR_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2";
const MODEL_DIR_NAME: &str = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8";

/// Model files (inside the extracted directory) - int8 quantized
const ENCODER_FILE: &str = "encoder.int8.onnx";
const DECODER_FILE: &str = "decoder.int8.onnx";
const JOINER_FILE: &str = "joiner.int8.onnx";
const TOKENS_FILE: &str = "tokens.txt";

pub struct ModelManager;

impl ModelManager {
    /// Get the models directory
    pub fn models_dir() -> Option<PathBuf> {
        Config::models_dir()
    }

    /// Ensure all models are downloaded and extracted
    fn ensure_models() -> Result<PathBuf, Box<dyn std::error::Error>> {
        let models_dir = Self::models_dir().ok_or("Could not determine models directory")?;
        fs::create_dir_all(&models_dir)?;

        let model_dir = models_dir.join(MODEL_DIR_NAME);
        let encoder_path = model_dir.join(ENCODER_FILE);

        // Check if already extracted
        if encoder_path.exists() {
            info!("Model already downloaded: {}", model_dir.display());
            return Ok(model_dir);
        }

        // Download and extract tar.bz2
        info!("Downloading Parakeet TDT 0.6B v2 model (~2.4GB)...");
        let tar_path = models_dir.join("model.tar.bz2");
        Self::download_file(MODEL_TAR_URL, &tar_path)?;

        info!("Extracting model files...");
        Self::extract_tar_bz2(&tar_path, &models_dir)?;

        // Clean up tar file
        fs::remove_file(&tar_path)?;

        info!("All model files ready: {}", model_dir.display());
        Ok(model_dir)
    }

    /// Download a file with progress
    fn download_file(url: &str, dest: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        info!("Downloading: {}", url);

        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(7200)) // 2 hours for large file
            .build()?;

        let mut response = client.get(url).send()?;

        if !response.status().is_success() {
            return Err(format!("Download failed: HTTP {}", response.status()).into());
        }

        let total_size = response.content_length();
        if let Some(size) = total_size {
            info!("Download size: {:.1} MB", size as f64 / 1024.0 / 1024.0);
        }

        let temp_path = dest.with_extension("tmp");
        let mut file = fs::File::create(&temp_path)?;

        let mut downloaded: u64 = 0;
        let mut last_progress = 0;
        let mut buffer = [0u8; 131072];

        loop {
            let bytes_read = response.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            file.write_all(&buffer[..bytes_read])?;
            downloaded += bytes_read as u64;

            if let Some(total) = total_size {
                let progress = (downloaded * 100 / total) as u32;
                if progress >= last_progress + 5 {
                    info!("Download progress: {}% ({:.1} MB / {:.1} MB)",
                          progress,
                          downloaded as f64 / 1024.0 / 1024.0,
                          total as f64 / 1024.0 / 1024.0);
                    last_progress = progress;
                }
            }
        }

        file.flush()?;
        drop(file);
        fs::rename(&temp_path, dest)?;

        info!("Download complete: {}", dest.display());
        Ok(())
    }

    /// Extract tar.bz2 archive
    fn extract_tar_bz2(tar_path: &PathBuf, dest_dir: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::open(tar_path)?;
        let reader = BufReader::new(file);
        let decompressor = BzDecoder::new(reader);
        let mut archive = Archive::new(decompressor);

        archive.unpack(dest_dir)?;

        info!("Extraction complete");
        Ok(())
    }

    /// Get path to the encoder ONNX model
    pub fn ensure_encoder() -> Result<PathBuf, Box<dyn std::error::Error>> {
        let model_dir = Self::ensure_models()?;
        let path = model_dir.join(ENCODER_FILE);
        if !path.exists() {
            return Err(format!("Encoder not found: {}", path.display()).into());
        }
        info!("Encoder: {}", path.display());
        Ok(path)
    }

    /// Get path to the decoder ONNX model
    pub fn ensure_decoder() -> Result<PathBuf, Box<dyn std::error::Error>> {
        let model_dir = Self::ensure_models()?;
        let path = model_dir.join(DECODER_FILE);
        if !path.exists() {
            return Err(format!("Decoder not found: {}", path.display()).into());
        }
        info!("Decoder: {}", path.display());
        Ok(path)
    }

    /// Get path to the joiner ONNX model
    pub fn ensure_joiner() -> Result<PathBuf, Box<dyn std::error::Error>> {
        let model_dir = Self::ensure_models()?;
        let path = model_dir.join(JOINER_FILE);
        if !path.exists() {
            return Err(format!("Joiner not found: {}", path.display()).into());
        }
        info!("Joiner: {}", path.display());
        Ok(path)
    }

    /// Get path to the tokens file
    pub fn ensure_tokens() -> Result<PathBuf, Box<dyn std::error::Error>> {
        let model_dir = Self::ensure_models()?;
        let path = model_dir.join(TOKENS_FILE);
        if !path.exists() {
            return Err(format!("Tokens not found: {}", path.display()).into());
        }
        info!("Tokens: {}", path.display());
        Ok(path)
    }
}
```

### `src/inference/parakeet.rs`

```rust
//! NVIDIA Parakeet-TDT-0.6B-V2 inference engine.
//!
//! Uses sherpa-onnx ONNX export with separate encoder + decoder + joiner.
//! Implements Token-and-Duration Transducer (TDT) greedy decoding.

use super::mel::MelSpectrogram;
use super::model_manager::ModelManager;

use ort::{
    execution_providers::CUDAExecutionProvider,
    session::Session,
    value::Tensor,
};
use std::collections::HashMap;
use std::fs;
use tracing::{debug, info};

/// TDT model constants - retrieved from model metadata
const PRED_RNN_LAYERS: usize = 2;  // Model has 2 LSTM layers in prediction network
const PRED_HIDDEN: usize = 640;

/// Parakeet TDT inference engine with CUDA acceleration
pub struct ParakeetEngine {
    encoder: Option<Session>,
    decoder: Option<Session>,
    joiner: Option<Session>,
    tokens: Option<HashMap<i64, String>>,
    vocab_size: usize,
    blank_id: i64,
    mel: MelSpectrogram,
    is_loaded: bool,
}

impl ParakeetEngine {
    /// Create a new engine (lazy loading - doesn't load model yet)
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            encoder: None,
            decoder: None,
            joiner: None,
            tokens: None,
            vocab_size: 0,
            blank_id: 0,
            mel: MelSpectrogram::new(),
            is_loaded: false,
        })
    }

    /// Check if model is loaded
    pub fn is_loaded(&self) -> bool {
        self.is_loaded
    }

    /// Unload model to free GPU memory
    pub fn unload(&mut self) {
        if self.is_loaded {
            info!("Unloading Parakeet model");
            self.encoder = None;
            self.decoder = None;
            self.joiner = None;
            self.tokens = None;
            self.is_loaded = false;
        }
    }

    /// Load tokens from tokens.txt
    /// Format: "token index" per line
    fn load_tokens(tokens_path: &std::path::Path) -> Result<(HashMap<i64, String>, usize), Box<dyn std::error::Error>> {
        let content = fs::read_to_string(tokens_path)?;
        let mut tokens = HashMap::new();
        let mut max_id: i64 = 0;

        for line in content.lines() {
            // Format: "token index" - split from the right
            if let Some(last_space) = line.rfind(' ') {
                let token = &line[..last_space];
                if let Ok(idx) = line[last_space + 1..].parse::<i64>() {
                    tokens.insert(idx, token.to_string());
                    if idx > max_id {
                        max_id = idx;
                    }
                }
            }
        }

        let vocab_size = (max_id + 1) as usize;
        info!("Loaded {} tokens, vocab_size={}", tokens.len(), vocab_size);
        Ok((tokens, vocab_size))
    }

    /// Ensure model is loaded (lazy loading)
    fn ensure_loaded(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.is_loaded {
            return Ok(());
        }

        info!("Loading Parakeet TDT model with CUDA...");

        // Ensure all model files are downloaded
        let encoder_path = ModelManager::ensure_encoder()?;
        let decoder_path = ModelManager::ensure_decoder()?;
        let joiner_path = ModelManager::ensure_joiner()?;
        let tokens_path = ModelManager::ensure_tokens()?;

        // Initialize ONNX Runtime
        ort::init()
            .with_name("voclaude")
            .commit()?;

        // Create encoder session with CUDA
        info!("Loading encoder...");
        let encoder = Session::builder()?
            .with_execution_providers([
                CUDAExecutionProvider::default().build(),
            ])?
            .commit_from_file(&encoder_path)?;
        info!("Encoder loaded");

        // Create decoder session with CUDA
        info!("Loading decoder...");
        let decoder = Session::builder()?
            .with_execution_providers([
                CUDAExecutionProvider::default().build(),
            ])?
            .commit_from_file(&decoder_path)?;
        info!("Decoder loaded");

        // Create joiner session with CUDA
        info!("Loading joiner...");
        let joiner = Session::builder()?
            .with_execution_providers([
                CUDAExecutionProvider::default().build(),
            ])?
            .commit_from_file(&joiner_path)?;
        info!("Joiner loaded");

        // Load tokens
        let (tokens, vocab_size) = Self::load_tokens(&tokens_path)?;
        let blank_id = (vocab_size - 1) as i64;  // Blank is last token

        self.encoder = Some(encoder);
        self.decoder = Some(decoder);
        self.joiner = Some(joiner);
        self.tokens = Some(tokens);
        self.vocab_size = vocab_size;
        self.blank_id = blank_id;
        self.is_loaded = true;

        info!("Parakeet TDT model loaded! vocab_size={}, blank_id={}", vocab_size, blank_id);
        Ok(())
    }

    /// Decode token IDs to text using vocabulary
    fn decode_tokens(&self, token_ids: &[i64]) -> String {
        let tokens = match &self.tokens {
            Some(t) => t,
            None => return String::new(),
        };

        let mut text = String::new();
        for &id in token_ids {
            // Skip blank and invalid tokens
            if id == self.blank_id || id < 0 || id as usize >= self.vocab_size {
                continue;
            }

            if let Some(token) = tokens.get(&id) {
                // Handle SentencePiece-style tokens (▁ prefix means space)
                if token.starts_with('▁') {
                    if !text.is_empty() {
                        text.push(' ');
                    }
                    text.push_str(&token[3..]); // Skip the ▁ character (3 bytes in UTF-8)
                } else if token == "<space>" {
                    text.push(' ');
                } else if !token.starts_with('<') {
                    // Skip special tokens like <blk>, <unk>
                    text.push_str(token);
                }
            }
        }

        text.trim().to_string()
    }

    /// Transcribe audio samples using TDT decoding
    /// Input: f32 samples at 16kHz mono
    /// Output: transcribed text
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<String, Box<dyn std::error::Error>> {
        self.ensure_loaded()?;

        // Run encoder and TDT decoding
        let output_tokens = self.run_inference(samples)?;

        // Decode tokens to text
        let text = self.decode_tokens(&output_tokens);
        debug!("Transcription: {}", text);

        Ok(text)
    }

    /// Helper to run decoder and extract outputs
    fn run_decoder(
        decoder: &mut Session,
        context: i64,
        state_h: &[f32],
        state_c: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
        let context_tensor = Tensor::from_array(([1_usize, 1_usize], vec![context as i32].into_boxed_slice()))?;
        let context_len_tensor = Tensor::from_array(([1_usize], vec![1_i32].into_boxed_slice()))?;
        let state_h_tensor = Tensor::from_array(([PRED_RNN_LAYERS, 1, PRED_HIDDEN], state_h.to_vec().into_boxed_slice()))?;
        let state_c_tensor = Tensor::from_array(([PRED_RNN_LAYERS, 1, PRED_HIDDEN], state_c.to_vec().into_boxed_slice()))?;

        // Get actual input names from model (clone to owned strings to avoid borrow conflict)
        let dec_in0 = decoder.inputs.get(0).map(|i| i.name.clone()).unwrap_or_else(|| "y".to_string());
        let dec_in1 = decoder.inputs.get(1).map(|i| i.name.clone()).unwrap_or_else(|| "y_lens".to_string());
        let dec_in2 = decoder.inputs.get(2).map(|i| i.name.clone()).unwrap_or_else(|| "state0".to_string());
        let dec_in3 = decoder.inputs.get(3).map(|i| i.name.clone()).unwrap_or_else(|| "state1".to_string());

        let outputs = decoder.run(ort::inputs![
            dec_in0.as_str() => context_tensor,
            dec_in1.as_str() => context_len_tensor,
            dec_in2.as_str() => state_h_tensor,
            dec_in3.as_str() => state_c_tensor,
        ])?;

        // Extract decoder output
        let dec_out_name = outputs.iter().next().map(|(n, _)| n.to_string());
        let dec_out = outputs.get("decoder_out")
            .or_else(|| dec_out_name.as_ref().and_then(|n| outputs.get(n)))
            .ok_or("Missing decoder output")?;
        let (_, dec_data) = dec_out.try_extract_tensor::<f32>()?;
        let decoder_out: Vec<f32> = dec_data.iter().cloned().collect();

        // Extract new states
        let mut new_h = state_h.to_vec();
        let mut new_c = state_c.to_vec();

        if let Some(h) = outputs.get("state0_out").or_else(|| outputs.get("out_state0")) {
            if let Ok((_, data)) = h.try_extract_tensor::<f32>() {
                for (i, &v) in data.iter().enumerate().take(PRED_RNN_LAYERS * PRED_HIDDEN) {
                    new_h[i] = v;
                }
            }
        }
        if let Some(c) = outputs.get("state1_out").or_else(|| outputs.get("out_state1")) {
            if let Ok((_, data)) = c.try_extract_tensor::<f32>() {
                for (i, &v) in data.iter().enumerate().take(PRED_RNN_LAYERS * PRED_HIDDEN) {
                    new_c[i] = v;
                }
            }
        }

        Ok((decoder_out, new_h, new_c))
    }

    /// Helper to run joiner and get logits
    fn run_joiner(
        joiner: &mut Session,
        enc_frame: Vec<f32>,
        dec_frame: Vec<f32>,
        enc_dim: usize,
        dec_dim: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let enc_tensor = Tensor::from_array(([1, enc_dim, 1], enc_frame.into_boxed_slice()))?;
        let dec_tensor = Tensor::from_array(([1, dec_dim, 1], dec_frame.into_boxed_slice()))?;

        // Get actual input names from model (clone to owned strings to avoid borrow conflict)
        let join_in0 = joiner.inputs.get(0).map(|i| i.name.clone()).unwrap_or_else(|| "encoder_out".to_string());
        let join_in1 = joiner.inputs.get(1).map(|i| i.name.clone()).unwrap_or_else(|| "decoder_out".to_string());

        let outputs = joiner.run(ort::inputs![
            join_in0.as_str() => enc_tensor,
            join_in1.as_str() => dec_tensor,
        ])?;

        let logit_name = outputs.iter().next().map(|(n, _)| n.to_string());
        let logits = outputs.get("logit")
            .or_else(|| logit_name.as_ref().and_then(|n| outputs.get(n)))
            .ok_or("Missing joiner output")?;
        let (_, logit_data) = logits.try_extract_tensor::<f32>()?;

        Ok(logit_data.iter().cloned().collect())
    }

    /// Run encoder and TDT greedy decoding
    fn run_inference(&mut self, samples: &[f32]) -> Result<Vec<i64>, Box<dyn std::error::Error>> {
        debug!("Transcribing {} samples ({:.2}s)", samples.len(), samples.len() as f32 / 16000.0);

        // Compute mel spectrogram: [n_mels, n_frames]
        let mel_spec = self.mel.compute(samples);
        let (n_mels, n_frames) = mel_spec.dim();
        debug!("Mel spectrogram: {} mels x {} frames", n_mels, n_frames);

        // Prepare encoder input: [batch=1, n_mels, time]
        let mel_vec: Vec<f32> = mel_spec.iter().cloned().collect();
        let mel_tensor = Tensor::from_array(([1, n_mels, n_frames], mel_vec.into_boxed_slice()))?;
        let length_tensor = Tensor::from_array(([1_usize], vec![n_frames as i64].into_boxed_slice()))?;

        // Run encoder - get input names dynamically
        debug!("Running encoder...");
        let encoder = self.encoder.as_mut().ok_or("Encoder not loaded")?;

        // Get actual input names from model (clone to owned strings to avoid borrow conflict)
        let enc_in0 = encoder.inputs.get(0).map(|i| i.name.clone()).unwrap_or_else(|| "x".to_string());
        let enc_in1 = encoder.inputs.get(1).map(|i| i.name.clone()).unwrap_or_else(|| "x_lens".to_string());
        debug!("Encoder input names: '{}', '{}'", enc_in0, enc_in1);

        let encoder_outputs = encoder.run(ort::inputs![
            enc_in0.as_str() => mel_tensor,
            enc_in1.as_str() => length_tensor,
        ])?;

        // Get encoder output - shape: [batch, dim, time]
        let (enc_data, enc_dim, enc_time) = {
            let enc_out_name = encoder_outputs.iter().next().map(|(n, _)| n.to_string());
            let enc_out = encoder_outputs.get("encoder_out")
                .or_else(|| enc_out_name.as_ref().and_then(|n| encoder_outputs.get(n)))
                .ok_or("Missing encoder output")?;
            let (enc_shape, enc_data_view) = enc_out.try_extract_tensor::<f32>()?;
            debug!("Encoder output shape: {:?}", enc_shape);

            // Parse encoder dimensions: [batch, dim, time]
            let enc_time = if enc_shape.len() == 3 {
                enc_shape[2] as usize
            } else {
                return Err(format!("Unexpected encoder shape: {:?}", enc_shape).into());
            };
            let enc_dim = enc_shape[1] as usize;

            // Copy encoder data so we can drop the session outputs
            let enc_data: Vec<f32> = enc_data_view.iter().cloned().collect();
            (enc_data, enc_dim, enc_time)
        };
        debug!("Encoder: {} frames x {} dim", enc_time, enc_dim);

        // TDT Greedy Decoding
        debug!("Starting TDT greedy decoding...");

        // Initialize decoder state: zeros for h and c
        let mut state_h = vec![0.0f32; PRED_RNN_LAYERS * PRED_HIDDEN];
        let mut state_c = vec![0.0f32; PRED_RNN_LAYERS * PRED_HIDDEN];

        // Initialize with blank context
        let mut context: i64 = self.blank_id;
        let mut output_tokens: Vec<i64> = Vec::new();

        // Run decoder once to get initial decoder_out (with blank context)
        let decoder = self.decoder.as_mut().ok_or("Decoder not loaded")?;
        let (initial_dec_out, state_h_next, state_c_next) = Self::run_decoder(decoder, context, &state_h, &state_c)?;

        let dec_dim = initial_dec_out.len();
        let mut decoder_out = initial_dec_out;
        let mut state_h_next = state_h_next;
        let mut state_c_next = state_c_next;

        debug!("Decoder output dim: {}", dec_dim);

        // Main decoding loop
        let mut t: usize = 0;
        while t < enc_time {
            // Extract encoder frame at time t
            let enc_frame: Vec<f32> = (0..enc_dim).map(|d| {
                // Data layout: [batch, dim, time] - batch=0, so index is d * enc_time + t
                let idx = d * enc_time + t;
                if idx < enc_data.len() { enc_data[idx] } else { 0.0 }
            }).collect();

            // Run joiner
            let joiner = self.joiner.as_mut().ok_or("Joiner not loaded")?;
            let logits = Self::run_joiner(joiner, enc_frame, decoder_out.clone(), enc_dim, dec_dim)?;

            // Split logits: first vocab_size are token logits, rest are duration logits
            let token_logits = &logits[..self.vocab_size.min(logits.len())];
            let duration_logits = if logits.len() > self.vocab_size {
                &logits[self.vocab_size..]
            } else {
                &[] as &[f32]
            };

            // Find best token
            let mut best_token: i64 = self.blank_id;
            let mut best_score = f32::NEG_INFINITY;
            for (i, &score) in token_logits.iter().enumerate() {
                if score > best_score {
                    best_score = score;
                    best_token = i as i64;
                }
            }

            // Find best duration (skip value)
            let skip: usize = if !duration_logits.is_empty() {
                let mut best_dur_idx: usize = 0;
                let mut best_dur_score = f32::NEG_INFINITY;
                for (i, &score) in duration_logits.iter().enumerate() {
                    if score > best_dur_score {
                        best_dur_score = score;
                        best_dur_idx = i;
                    }
                }
                best_dur_idx
            } else {
                1 // Default skip of 1 if no duration logits
            };

            if best_token != self.blank_id {
                // Non-blank token: emit token and update decoder state
                output_tokens.push(best_token);
                context = best_token;

                // Commit state update
                state_h = state_h_next;
                state_c = state_c_next;

                // Run decoder with new context to get new decoder_out
                let decoder = self.decoder.as_mut().ok_or("Decoder not loaded")?;
                let (new_dec_out, new_h, new_c) = Self::run_decoder(decoder, context, &state_h, &state_c)?;
                decoder_out = new_dec_out;
                state_h_next = new_h;
                state_c_next = new_c;
            }

            // Advance time by skip value (minimum 1 to ensure progress)
            t += skip.max(1);
        }

        debug!("Decoded {} tokens", output_tokens.len());
        Ok(output_tokens)
    }
}

impl Default for ParakeetEngine {
    fn default() -> Self {
        Self::new().expect("Failed to create ParakeetEngine")
    }
}
```

### `src/inference/mel.rs`

```rust
//! Mel spectrogram computation for Parakeet TDT 0.6B v2.
//!
//! The istupakov ONNX export expects:
//! - n_fft: 512
//! - hop_length: 160 (10ms at 16kHz)
//! - win_length: 400 (25ms at 16kHz)
//! - n_mels: 128 (FastConformer uses 128 mels)
//! - fmin: 0
//! - fmax: 8000

use ndarray::{Array1, Array2};
use realfft::{RealFftPlanner, RealToComplex};
use std::f32::consts::PI;
use std::sync::Arc;

/// Mel spectrogram parameters for Parakeet TDT 0.6B v2
pub const N_FFT: usize = 512;
pub const HOP_LENGTH: usize = 160;
pub const WIN_LENGTH: usize = 400;
pub const N_MELS: usize = 128;  // FastConformer encoder expects 128 mels
pub const SAMPLE_RATE: u32 = 16000;
pub const F_MIN: f32 = 0.0;
pub const F_MAX: f32 = 8000.0;

/// Mel spectrogram extractor
pub struct MelSpectrogram {
    fft: Arc<dyn RealToComplex<f32>>,
    mel_filterbank: Array2<f32>,
    window: Vec<f32>,
}

impl MelSpectrogram {
    pub fn new() -> Self {
        // Create FFT planner
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);

        // Create Hann window
        let window: Vec<f32> = (0..WIN_LENGTH)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / WIN_LENGTH as f32).cos()))
            .collect();

        // Create mel filterbank
        let mel_filterbank = Self::create_mel_filterbank();

        Self {
            fft,
            mel_filterbank,
            window,
        }
    }

    /// Convert Hz to Mel scale
    fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    /// Convert Mel to Hz scale
    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }

    /// Create mel filterbank matrix
    fn create_mel_filterbank() -> Array2<f32> {
        let n_fft_bins = N_FFT / 2 + 1;
        let mut filterbank = Array2::<f32>::zeros((N_MELS, n_fft_bins));

        let mel_min = Self::hz_to_mel(F_MIN);
        let mel_max = Self::hz_to_mel(F_MAX);

        // Create mel points
        let mel_points: Vec<f32> = (0..=N_MELS + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (N_MELS + 1) as f32)
            .collect();

        // Convert to Hz and then to FFT bin indices
        let hz_points: Vec<f32> = mel_points.iter().map(|&m| Self::mel_to_hz(m)).collect();
        let bin_points: Vec<usize> = hz_points
            .iter()
            .map(|&hz| ((N_FFT as f32 + 1.0) * hz / SAMPLE_RATE as f32).floor() as usize)
            .collect();

        // Create triangular filters
        for m in 0..N_MELS {
            let left = bin_points[m];
            let center = bin_points[m + 1];
            let right = bin_points[m + 2];

            // Rising edge
            for k in left..center {
                if center > left {
                    filterbank[[m, k]] = (k - left) as f32 / (center - left) as f32;
                }
            }

            // Falling edge
            for k in center..right {
                if right > center {
                    filterbank[[m, k]] = (right - k) as f32 / (right - center) as f32;
                }
            }
        }

        filterbank
    }

    /// Compute mel spectrogram from audio samples
    /// Input: f32 samples at 16kHz
    /// Output: [n_mels, n_frames] mel spectrogram
    pub fn compute(&self, samples: &[f32]) -> Array2<f32> {
        let n_frames = (samples.len().saturating_sub(WIN_LENGTH)) / HOP_LENGTH + 1;
        if n_frames == 0 {
            return Array2::zeros((N_MELS, 1));
        }

        let n_fft_bins = N_FFT / 2 + 1;
        let mut spectrogram = Array2::<f32>::zeros((N_MELS, n_frames));

        // Prepare FFT buffers
        let mut fft_input = vec![0.0f32; N_FFT];
        let mut fft_output = self.fft.make_output_vec();

        for frame_idx in 0..n_frames {
            let start = frame_idx * HOP_LENGTH;
            let end = (start + WIN_LENGTH).min(samples.len());

            // Clear and fill FFT input with windowed samples
            fft_input.fill(0.0);
            for (i, &sample) in samples[start..end].iter().enumerate() {
                fft_input[i] = sample * self.window[i];
            }

            // Compute FFT
            self.fft
                .process(&mut fft_input, &mut fft_output)
                .expect("FFT failed");

            // Compute power spectrum
            let power_spectrum: Vec<f32> = fft_output
                .iter()
                .map(|c| c.norm_sqr())
                .collect();

            // Apply mel filterbank
            for mel_idx in 0..N_MELS {
                let mut mel_energy = 0.0f32;
                for bin_idx in 0..n_fft_bins.min(power_spectrum.len()) {
                    mel_energy += self.mel_filterbank[[mel_idx, bin_idx]] * power_spectrum[bin_idx];
                }
                // Log mel spectrogram (add small epsilon to avoid log(0))
                spectrogram[[mel_idx, frame_idx]] = (mel_energy + 1e-10).ln();
            }
        }

        // Normalize per feature (mean=0, std=1)
        self.normalize(&mut spectrogram);

        spectrogram
    }

    /// Per-feature normalization
    fn normalize(&self, spec: &mut Array2<f32>) {
        let (n_mels, n_frames) = spec.dim();
        if n_frames == 0 {
            return;
        }

        for mel_idx in 0..n_mels {
            // Compute mean
            let mut sum = 0.0f32;
            for frame_idx in 0..n_frames {
                sum += spec[[mel_idx, frame_idx]];
            }
            let mean = sum / n_frames as f32;

            // Compute std
            let mut var_sum = 0.0f32;
            for frame_idx in 0..n_frames {
                let diff = spec[[mel_idx, frame_idx]] - mean;
                var_sum += diff * diff;
            }
            let std = (var_sum / n_frames as f32).sqrt().max(1e-10);

            // Normalize
            for frame_idx in 0..n_frames {
                spec[[mel_idx, frame_idx]] = (spec[[mel_idx, frame_idx]] - mean) / std;
            }
        }
    }
}

impl Default for MelSpectrogram {
    fn default() -> Self {
        Self::new()
    }
}
```

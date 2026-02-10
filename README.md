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

## Qwen3-ASR Smoke Test

A Python-first Qwen3-ASR validation harness is available in `tools/qwen3_asr_smoke/`.
Use it to verify Russian/English behavior and runtime viability before Rust integration.

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

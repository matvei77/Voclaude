# Voclaude

Voice input anywhere — local-first, GPU-accelerated speech-to-text.

## Features

- **Local-first**: Uses Whisper.cpp for on-device transcription
- **GPU accelerated**: CUDA/Metal support for fast inference
- **Tiered models**: Auto-selects model based on your hardware (tiny/base/small/medium)
- **Lazy loading**: Model loads on first use, unloads after idle
- **System tray**: Minimal UI, stays out of your way
- **Global hotkey**: Win+C (configurable) to start/stop recording

## Installation

### Prerequisites

- Rust 1.70+ (`rustup` recommended)
- For GPU: CUDA toolkit (NVIDIA) or Metal (macOS)

### Build

```bash
# CPU only
cargo build --release

# With CUDA
cargo build --release --features cuda

# With Metal (macOS)
cargo build --release --features metal
```

### Run

```bash
cargo run --release
```

Or copy `target/release/voclaude.exe` to your preferred location.

## Usage

1. Launch Voclaude — it appears in your system tray
2. Press **Win+C** to start recording (or click tray icon → "Start Recording")
3. Speak
4. Press **Win+C** again to stop and transcribe
5. Text is copied to clipboard — paste with Ctrl+V

## Configuration

Config file location:
- Windows: `%APPDATA%\voclaude\Voclaude\config.toml`
- macOS: `~/Library/Application Support/com.voclaude.Voclaude/config.toml`
- Linux: `~/.config/voclaude/config.toml`

```toml
# Hotkey to toggle recording
hotkey = "Super+C"

# Model tier: tiny, base, small, medium
model_tier = "base"

# Auto-select model based on hardware
auto_select_model = true

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
```

## Model Tiers

| Tier | VRAM | Quality | Speed |
|------|------|---------|-------|
| tiny | ~75MB | Decent | ~1s |
| base | ~150MB | Good | ~2s |
| small | ~500MB | Great | ~3s |
| medium | ~1.5GB | Excellent | ~5s |

Models are downloaded automatically on first use from Hugging Face.

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
│  - Auto-tier selection          │
└─────────────────────────────────┘
```

## License

MIT

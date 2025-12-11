# Voclaude

Voice input anywhere — local-first, GPU-accelerated speech-to-text that runs in your system tray.

Press a hotkey, speak, and your words are instantly transcribed and copied to your clipboard. All processing happens locally on your machine using OpenAI's Whisper model.

## Features

- **Cross-platform** — Windows and Linux support
- **Local-first** — All transcription happens on-device, no cloud required
- **GPU accelerated** — Optional CUDA support for fast inference
- **Lazy loading** — Model loads on first use, auto-unloads after idle to save VRAM
- **System tray** — Minimal UI, stays out of your way
- **Global hotkey** — Configurable hotkey (default: F4) works from any application

## Quick Start

```bash
# Install Rust (if needed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone and build
git clone https://github.com/matvei77/Voclaude.git
cd Voclaude
cargo build --release

# Run
./target/release/voclaude
```

## Installation

### Prerequisites

- Rust 1.70+ ([rustup](https://rustup.rs) recommended)
- For GPU acceleration: CUDA toolkit (NVIDIA)

### Linux Dependencies

```bash
# Ubuntu/Debian
sudo apt install libasound2-dev libgtk-3-dev libayatana-appindicator3-dev \
    libxkbcommon-dev libxdo-dev pkg-config cmake clang

# Fedora
sudo dnf install alsa-lib-devel gtk3-devel libayatana-appindicator-gtk3-devel \
    libxkbcommon-devel libX11-devel cmake clang

# Arch
sudo pacman -S alsa-lib gtk3 libappindicator-gtk3 libxkbcommon cmake clang
```

### Build

```bash
# CPU only (default)
cargo build --release

# With CUDA support (requires CUDA toolkit)
cargo build --release --features cuda
```

### Run

```bash
./target/release/voclaude      # Linux
target\release\voclaude.exe    # Windows
```

## Usage

1. **Launch** — Voclaude appears in your system tray
2. **Press F4** (or your configured hotkey) to start recording
3. **Speak** — your voice is captured locally
4. **Press F4 again** to stop and transcribe
5. **Paste** (Ctrl+V) — transcribed text is in your clipboard

The first time you use Voclaude, it will download the Whisper model (~1.5GB). This only happens once.

## Configuration

Config file locations:
- **Linux:** `~/.config/voclaude/config.toml`
- **Windows:** `%APPDATA%\voclaude\Voclaude\config.toml`

```toml
# Hotkey to toggle recording
hotkey = "F4"                    # Also: "Super+C", "Ctrl+Shift+Space", etc.

# Add trailing space after transcription
add_trailing_space = true

# Capitalize first letter
capitalize_first = true

# Unload model after N seconds idle (saves VRAM)
idle_unload_seconds = 300
```

### Supported Hotkeys

- **Modifiers:** `Ctrl`, `Alt`, `Shift`, `Super` (Win key)
- **Keys:** A-Z, 0-9, F1-F12, Space, Enter, Tab, Escape, etc.
- **Examples:** `F4`, `Super+C`, `Ctrl+Shift+Space`, `Alt+V`

## How It Works

```
┌─────────────────────────────────────────┐
│            System Tray                   │
│  [Idle] → [Recording] → [Transcribing]  │
└─────────────────┬───────────────────────┘
                  │
    ┌─────────────▼─────────────┐
    │     Audio Capture         │
    │  - cpal (cross-platform)  │
    │  - 16kHz mono resampling  │
    └─────────────┬─────────────┘
                  │
    ┌─────────────▼─────────────┐
    │    Whisper Inference      │
    │  - whisper.cpp bindings   │
    │  - CPU or CUDA            │
    │  - ~1.5GB model (medium)  │
    └─────────────┬─────────────┘
                  │
    ┌─────────────▼─────────────┐
    │       Clipboard           │
    │  - Auto-formatted text    │
    │  - Ready to paste         │
    └───────────────────────────┘
```

## Performance

| Mode | Transcription Speed | Memory |
|------|---------------------|--------|
| CPU | ~10-15s for 10s audio | ~2GB RAM |
| CUDA | ~1-2s for 10s audio | ~1.5GB VRAM |

## Troubleshooting

### Linux: Hotkey not working on Wayland
Global hotkeys may require X11 compatibility. Try running with:
```bash
GDK_BACKEND=x11 ./target/release/voclaude
```

### Model download fails
The Whisper model is downloaded from Hugging Face. Ensure you have internet access and ~2GB free disk space.

### No audio input detected
Check that your microphone is working and set as the default input device in your system settings.

## Building from Source

```bash
# Clone
git clone https://github.com/matvei77/Voclaude.git
cd Voclaude

# Build (CPU)
cargo build --release

# Build (CUDA) - requires CUDA toolkit
cargo build --release --features cuda

# Run tests
cargo test
```

## License

MIT

## Acknowledgments

- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) — Fast C++ Whisper implementation
- [whisper-rs](https://github.com/tazz4843/whisper-rs) — Rust bindings for whisper.cpp
- [cpal](https://github.com/RustAudio/cpal) — Cross-platform audio I/O

# Voclaude

Voice input anywhere — local-first, GPU-accelerated speech-to-text that runs in your system tray.

Press a hotkey, speak, and your words are instantly transcribed and copied to your clipboard. All processing happens locally using [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) via pure Rust inference (no Python needed).

## Features

- **Single binary** — pure Rust, no Python or external runtimes
- **Local-first** — all transcription happens on-device, nothing leaves your machine
- **GPU accelerated** — CUDA support for fast inference (~2s for 10s of audio)
- **Lazy loading** — model downloads on first use, auto-unloads after idle to save VRAM
- **System tray** — lives in your tray, stays out of your way
- **HUD overlay** — recording/transcribing status appears as a small overlay
- **History window** — browse, search, and copy past transcriptions (Ctrl+Shift+H)
- **Global hotkey** — configurable hotkey (default: F4) works from any application
- **Session recovery** — if the app crashes mid-transcription, it recovers on restart

## Quick Start (download)

If someone gave you a `voclaude-vX.Y.Z-gpu.zip`:

1. **Unzip** the archive anywhere (e.g. `C:\Voclaude\`)
2. **Run** `voclaude.exe` — it appears in your system tray
3. **Press F4** to record, **F4 again** to stop — transcription is copied to your clipboard
4. **Paste** (Ctrl+V) anywhere

The ASR model (~4.5 GB) downloads automatically from Hugging Face on first use. After that, everything works offline.

### Requirements (download)

- **Windows 10/11** (64-bit)
- **NVIDIA GPU** with recent drivers (the zip includes CUDA runtime DLLs, no toolkit install needed)
- ~4.5 GB free disk space for the model

For CPU-only builds (no NVIDIA GPU needed), ask for the `-cpu` zip or build from source (see below).

## Quick Start (build from source)

### Prerequisites

- **Rust** 1.70+ ([rustup.rs](https://rustup.rs))
- **NVIDIA GPU** with [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 12.x+ (for GPU mode)
- ~4.5 GB disk space for the model (downloaded automatically on first run)

### Build and Run

```bash
git clone https://github.com/matvei77/Voclaude.git
cd Voclaude
cargo build --release
```

```bash
# Windows
target\release\voclaude.exe

# Linux
./target/release/voclaude
```

### CPU-only Build

If you don't have an NVIDIA GPU:

```bash
cargo build --release --no-default-features --features cpu
```

### Packaging for Distribution

To build a distributable zip with CUDA DLLs bundled:

```powershell
.\package.ps1          # GPU build -> dist\voclaude-vX.Y.Z-gpu.zip
.\package.ps1 -Cpu     # CPU build -> dist\voclaude-vX.Y.Z-cpu.zip
```

The GPU zip includes `voclaude.exe`, the required CUDA DLLs, and `config.example.toml`. Recipients only need NVIDIA GPU drivers — no CUDA Toolkit or Rust toolchain.

### Linux Dependencies

```bash
# Ubuntu/Debian
sudo apt install libasound2-dev libgtk-3-dev libayatana-appindicator3-dev \
    libxkbcommon-dev libxdo-dev pkg-config cmake clang
```

## Usage

1. **Launch** — Voclaude appears in your system tray
2. **Press F4** to start recording — a HUD overlay appears top-right
3. **Speak** — your voice is captured locally
4. **Press F4 again** to stop — the HUD shows "Transcribing...", then "Copied to clipboard"
5. **Paste** (Ctrl+V) — transcribed text is in your clipboard

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `F4` | Start/stop recording |
| `Ctrl+Shift+H` | Toggle history window |

### Tray Menu

Right-click the tray icon for: Show History, Open Transcripts, Settings, Quit.

## Configuration

Config is created automatically on first run.

**Location:** `%APPDATA%\voclaude\Voclaude\config\config.toml`

```toml
hotkey = "F4"
history_hotkey = "Ctrl+Shift+H"
add_trailing_space = true
capitalize_first = true
idle_unload_seconds = 30
use_gpu = true
model = "Qwen/Qwen3-ASR-1.7B"
```

See [`config.example.toml`](config.example.toml) for all options.

### Supported Hotkeys

- **Modifiers:** `Ctrl`, `Alt`, `Shift`, `Super`/`Win`
- **Keys:** A-Z, 0-9, F1-F12, Space, Enter, Tab, Escape, Numpad keys, punctuation
- **Examples:** `F4`, `Super+C`, `Ctrl+Shift+Space`, `Alt+V`

## Architecture

```
System Tray ─── Global Hotkey (F4)
    │                │
    │    ┌───────────▼───────────┐
    │    │    Audio Capture      │
    │    │  cpal + 16kHz resamp  │
    │    └───────────┬───────────┘
    │                │
    │    ┌───────────▼───────────┐
    │    │   Qwen3-ASR Candle    │
    │    │   Pure Rust inference  │
    │    │   CPU or CUDA          │
    │    │   ~4.5 GB model        │
    │    └───────────┬───────────┘
    │                │
    │    ┌───────────▼───────────┐
    │    │     Clipboard         │
    │    │  Formatted + copied   │
    │    └───────────────────────┘
    │
    ├── HUD Overlay (recording/transcribing status)
    └── History Window (searchable transcript archive)
```

## Troubleshooting

### Model download is slow
The Qwen3-ASR model (~4.5 GB) is downloaded from Hugging Face on first use. Ensure you have a stable internet connection and sufficient disk space.

### No audio input detected
Check that your microphone is set as the default input device in your system sound settings.

### CUDA errors
- If using the prebuilt zip: make sure you have an NVIDIA GPU and up-to-date drivers (`nvidia-smi` should work)
- If building from source: ensure CUDA Toolkit 12.x+ is installed
- Try CPU mode: rebuild with `--no-default-features --features cpu`

### Hotkey not working
- Make sure no other application has registered the same global hotkey (F4)
- Try a different hotkey in the config file (e.g. `hotkey = "Ctrl+Shift+Space"`)

### Linux: Hotkey not working on Wayland
Global hotkeys may require X11 compatibility:
```bash
GDK_BACKEND=x11 ./target/release/voclaude
```

## License

MIT

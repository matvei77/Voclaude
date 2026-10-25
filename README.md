# Voclaude

Voice input anywhere — local-first, GPU-accelerated speech-to-text that runs in your system tray.

Press a hotkey, speak, and your words are instantly transcribed and copied to your clipboard. All processing happens locally: by default with [Parakeet Ultra](https://huggingface.co/handy-computer/parakeet-ultra-gguf) (0.6B, 25 European languages including Russian) through [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp), or with [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) through pure Rust inference (candle). No Python needed.

## Features

- **No Python** — a Rust app plus the transcribe.cpp/ggml DLLs next to it
- **Local-first** — all transcription happens on-device, nothing leaves your machine
- **Fast on GPU and CPU** — Parakeet runs at ~145x realtime on an RTX 3060 (Vulkan) and ~26x on an i5-13600K CPU; the Qwen engine runs on CUDA at ~30x
- **Russian and English in one recording** — segments are cut at pauses and Parakeet picks the language of each one
- **Transcribes while you speak** — the recording is cut into segments at pauses and transcribed in the background; after you stop, only the last few seconds remain, so the text is on the clipboard within about a second
- **Never loses a word** — audio is streamed to disk and synced every 5 s; every finished segment's text is journaled; a crash or power cut resumes from the last finished segment on relaunch, and the tray has "Recover Last Recording"
- **Tiny when idle** — inference runs in a child process that exits after 60 s idle: 0 VRAM and ~20 MB of RAM for the tray
- **Lazy loading** — model downloads on first use; loads while you are still speaking
- **System tray** — lives in your tray, stays out of your way
- **HUD overlay** — recording/transcribing status appears as a small overlay
- **History window** — browse, search, and copy past transcriptions (Ctrl+Shift+H)
- **Global hotkey** — configurable hotkey (default: F4) works from any application
- **Session recovery** — if the app crashes mid-transcription, it recovers on restart

## Quick Start (download)

1. Download `voclaude-vX.Y.Z-gpu-setup.exe` from the [Releases](https://github.com/matvei77/Voclaude/releases) page and run it.
   It installs for the current user (no admin prompt), can start with Windows, and launches the app. A tray icon appears.
   Prefer no installer? `voclaude-vX.Y.Z-<hash>-gpu.zip` unzips anywhere; run `voclaude.exe` from the folder.
2. Press **F4**, speak, press **F4** again. The text is on your clipboard; paste it with Ctrl+V.

On the first run the app downloads the model from Hugging Face (Parakeet: about 0.6 GB; Qwen:
about 3.4 GB). Later starts load in about 1.5 s. The very first Parakeet run on a GPU also compiles
the Vulkan shaders once (~10 s); the driver caches them.

To start it with Windows: press Win+R, type `shell:startup`, and put a shortcut to `voclaude.exe` there.

### Requirements (download)

- Windows 10/11, 64-bit.
- **Parakeet (default):** any GPU with a Vulkan driver and ~1 GB free VRAM, or just the CPU
  (~26x realtime on a desktop CPU; the right ggml CPU module is picked at runtime). See
  `docs/bench/2026-10-23-model-survey.md`.
- **Qwen (`engine = "qwen"`):** an NVIDIA GPU from the GTX 16xx / RTX 20xx generation or newer
  with 6 GB or more VRAM and a driver from 2025 or newer (the zip bundles the CUDA DLLs). On the
  CPU it runs at about 2x realtime for the 1.7B model and 3.4x for 0.6B (see `docs/bench/README.md`).
  Needs a CPU with AVX2 (2013+).
- Microphone. Windows: Settings → Privacy → Microphone → allow desktop apps.

## Quick Start (build from source)

### Prerequisites

- **Rust** 1.70+ ([rustup.rs](https://rustup.rs))
- **NVIDIA GPU** with [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 12.x+ (for GPU mode)
- MSVC build tools (for `fetch-transcribe.ps1`, which turns the official transcribe.cpp DLLs into a link prefix)
- ~1 GB disk space for the Parakeet model, ~4.5 GB for Qwen (downloaded automatically on first run)

### Build and Run

```bash
git clone https://github.com/matvei77/Voclaude.git
cd Voclaude
.\fetch-transcribe.ps1   # once: transcribe.cpp 0.4.1 runtime (CPU + Vulkan) into vendor/
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

On Windows, run `fetch-transcribe.ps1` in the same PowerShell session as Cargo.
The script exports `TRANSCRIBE_DIR` for that shell. The prebuilt prefix contains
Windows DLLs and must not be used on Linux or macOS. On those platforms Cargo
builds transcribe.cpp from source and links it statically. The `metal` feature
also enables transcribe.cpp's Metal backend. These non-Windows builds have not
been revalidated with the Parakeet integration.

## Usage

1. **Launch** — Voclaude appears in your system tray
2. **Press F4** to start recording — a HUD overlay appears top-right
3. **Speak** — your voice is captured locally
4. **Press F4 again** to stop — segments were already transcribed while you spoke; the last one finishes and the HUD shows "Copied to clipboard"
5. **Paste** (Ctrl+V) — transcribed text is in your clipboard

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `F4` | Start/stop recording |
| `Ctrl+Shift+H` | Toggle history window |

### Tray Menu

Right-click the tray icon for: Show History, Open Last Transcript, Open Transcripts Folder, Recover Last Recording, Settings, Quit.

## Local API (for scripts and agents)

While the app runs it serves `http://127.0.0.1:7770/v1/audio/transcriptions`, an
OpenAI-compatible transcription endpoint that decodes audio and video files
(wav, mp3, flac, ogg, m4a/mp4/mov, mkv; anything else via `ffmpeg` if installed)
and uses the same loaded model as dictation:

```
voclaude transcribe meeting.mp4            # prints the text
curl -F file=@clip.wav http://127.0.0.1:7770/v1/audio/transcriptions
```

See [docs/API.md](docs/API.md). Disable with `api_enabled = false`.

## Configuration

Config is created automatically on first run.

**Location:** `%APPDATA%\voclaude\Voclaude\config\config.toml`

```toml
hotkey = "F4"
history_hotkey = "Ctrl+Shift+H"
add_trailing_space = true
capitalize_first = true
idle_unload_seconds = 60
use_gpu = true
engine = "parakeet"          # or "qwen"
parakeet_model = "handy-computer/parakeet-ultra-gguf/parakeet-ultra-Q6_K.gguf"
model = "Qwen/Qwen3-ASR-1.7B" # Qwen engine only
quantization = "q8_0"        # Qwen engine only; "none" for F16 weights
streaming = true             # transcribe segments while recording
# segment_min_seconds = 6    # default 6 for Parakeet, 20 for Qwen
segment_max_seconds = 45
segment_pause_seconds = 0.5
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
    │    │  Parakeet (default)   │
    │    │  transcribe.cpp/ggml  │
    │    │  Vulkan or CPU, 0.7GB │
    │    │  or Qwen3-ASR (candle)│
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

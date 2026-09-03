## Codebase Overview

Voclaude is a pure-Rust desktop voice-to-text dictation tool that runs as a Windows system tray application. It captures microphone audio via CPAL, computes mel spectrograms, runs Qwen3-ASR-1.7B inference through Candle 0.9 on GPU (CUDA) or CPU, and copies the transcribed text to the clipboard.

**Stack**: Rust, Candle 0.9 (ML), egui/eframe 0.27 (UI), CPAL (audio), crossbeam (channels), global-hotkey + tray-icon (OS integration), HuggingFace hub (model download)

**Structure**: `src/app.rs` is the event loop and state machine. `src/inference/candle_backend.rs` is the ML kernel (~12k tokens, largest file). Audio capture is in `src/audio/`. Config, session recovery, and history are standalone modules.

For detailed architecture, module guide, data flow diagrams, and navigation guides, see [docs/CODEBASE_MAP.md](docs/CODEBASE_MAP.md).

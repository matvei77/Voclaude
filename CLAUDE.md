## Codebase Overview

Voclaude is a pure-Rust desktop voice-to-text dictation tool that runs as a Windows system tray application. It captures microphone audio via CPAL, cuts it into segments at pauses, transcribes them with Parakeet Ultra through transcribe.cpp (default; Vulkan GPU or CPU, `src/inference/parakeet.rs`) or with Qwen3-ASR-1.7B through Candle 0.9 (NVIDIA GPU with the `cuda` feature, or CPU), and copies the text to the clipboard. Inference runs in a `--worker` child process. A local OpenAI-compatible HTTP API on `127.0.0.1:7770` (`src/api.rs`) shares the loaded model.

**Build**: run `.\fetch-transcribe.ps1` once per shell (it exports `TRANSCRIBE_DIR`), then `cargo build --release`. The default build needs no CUDA toolkit. `--features cuda` adds the Qwen engine on NVIDIA GPUs. `package.ps1 [-Cuda] [-Installer]` writes the zip and the Inno Setup installer to `dist/`.

**Stack**: Rust, transcribe.cpp 0.4.1 (prebuilt DLLs via `fetch-transcribe.ps1`), Candle 0.9 (ML), egui/eframe 0.27 (UI), CPAL (audio), crossbeam (channels), global-hotkey + tray-icon (OS integration), HuggingFace hub (model download)

**Structure**: `src/app.rs` is the event loop and state machine. `src/inference/candle_backend.rs` is the ML kernel (~12k tokens, largest file). Audio capture and the pause segmenter are in `src/audio/`. Config, session recovery, and history are standalone modules. Benchmarks and model decisions are in `docs/bench/`. Superseded plans are in `docs/history/`.

For detailed architecture, module guide, data flow diagrams, and navigation guides, see [docs/CODEBASE_MAP.md](docs/CODEBASE_MAP.md).

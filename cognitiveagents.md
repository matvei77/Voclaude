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

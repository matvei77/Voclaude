# Voclaude Qwen Runtime

Full replacement test app with the same UX flow (tray + global hotkey + record/transcribe/paste),
but inference is Qwen via Python/CUDA instead of whisper.cpp.

This project is isolated from the main app and uses separate app data paths.

## Behavior

- F4 to start recording.
- F4 again to stop and transcribe.
- On first transcription: spawns a persistent Python server that loads the Qwen model into VRAM.
- Subsequent transcriptions reuse the running server — no model reload, near-instant inference.
- Server auto-restarts if it crashes; shuts down cleanly on app exit or idle unload.

## Default backend settings

Config lives in:
- Windows: `%APPDATA%\voclaude\VoclaudeQwenRuntime\config\config.toml`

Important defaults:
- `qwen_model = "Qwen/Qwen3-ASR-1.7B"`
- `qwen_require_gpu = true`
- `qwen_chunk_seconds = 60.0`
- `qwen_chunk_overlap_seconds = 2.0`

Set these paths in config for deterministic startup:
- `qwen_python_path`
- `qwen_script_path`

Example:
```toml
qwen_python_path = "<path-to-repo>/tools/qwen3_asr_smoke/.venv/Scripts/python.exe"
qwen_script_path = "<path-to-repo>/tools/qwen3_asr_smoke/transcribe.py"
```

## Build and run

```powershell
cd next/voclaude_qwen_runtime
cargo run --release
```

The app now performs backend preflight at startup and exits immediately with a clear message if:
- Python path/script path is invalid
- CUDA is required but unavailable

## Test mode

```powershell
cargo run --release -- --test "<path-to-audio-file>"
```

## Troubleshooting

If transcription does not copy to clipboard:
1. Check app startup preflight error in terminal.
2. Verify config file path from startup logs.
3. Confirm these values in config:
   - `qwen_python_path`
   - `qwen_script_path`
   - `qwen_require_gpu = true`
   - `qwen_device = "cuda:0"`
4. Run `--test` first to validate inference independently from tray/hotkey flow.

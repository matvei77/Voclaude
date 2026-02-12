# Voclaude Frontier (Parallel App)

This is an isolated, parallel runtime for next-gen local speech-to-text.

Goals:
- keep the existing Voclaude app untouched
- run in parallel during migration
- enforce GPU-first, Qwen 1.7B chunked transcription
- keep a clean architecture with pluggable inference backends

## Status

Current backend:
- `qwen_python`: Rust orchestrator + Python Qwen3-ASR harness (CUDA required by default config)

Execution model (current):
- one-shot inference process per transcription request
- Rust spawns Python, Python loads model, transcribes, returns JSON, process exits
- no always-resident model in GPU/CPU memory between requests

This is intentionally separated from the main app while we validate reliability/performance.

## Layout

- `src/main.rs`: CLI entrypoint
- `src/config.rs`: app/backend/chunking config
- `src/audio.rs`: input normalization helpers (`.wav`, `.f32`)
- `src/backend/`: backend abstraction + Qwen implementation
- `config/default.toml`: default runtime settings

## Quick Start

1. Make sure the Qwen smoke environment from this repo is installed:
   - `tools/qwen3_asr_smoke/.venv/...`
2. Build:
   - `cargo build --release`
3. Health check:
   - `cargo run -- healthcheck`
4. Transcribe:
   - `cargo run -- transcribe --input "C:\Users\matvei\Documents\Sound Recordings\Recording (2).wav"`

## Notes

- This runtime intentionally fails fast if CUDA is not available when `require_gpu=true`.
- CPU-only validation is possible only with an explicit temporary override config.
- Default model is `Qwen/Qwen3-ASR-1.7B` with chunking enabled.

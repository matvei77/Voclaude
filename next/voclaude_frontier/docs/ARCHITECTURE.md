# Architecture

## What is Rust vs Python here

- The product binary is Rust (`voclaude-frontier`).
- The inference runtime is currently delegated to Python (`qwen-asr`) for model execution.
- CUDA is used by Python/Torch during inference.

So this is a **Rust orchestrator + Python CUDA inference backend**.

## Request Flow (current)

1. Rust CLI receives input audio path.
2. Rust prepares input (`.wav` passthrough, `.f32` converted to temp wav).
3. Rust spawns Python `transcribe.py` as a one-shot process.
4. Python loads Qwen model, runs transcription (chunked if configured), prints JSON.
5. Rust parses JSON result, writes optional output, exits.
6. Python process exits -> model memory is released by process teardown.

## GPU policy

- Default config (`config/default.toml`) sets:
  - `require_gpu=true`
  - `device="cuda:0"`
- Healthcheck fails fast if CUDA is unavailable.
- A temporary CPU config can be used only for pipeline validation.

## Why `cargo check` still matters

`cargo check` validates Rust compile/integration correctness:
- config parsing
- process orchestration
- JSON contract handling
- error paths

It does **not** validate CUDA runtime availability by itself.

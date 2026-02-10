# Qwen3-ASR Integration Plan (Python -> Rust)

Date: 2026-02-10

## Objective

Replace single-backend Whisper-only transcription with a production-capable Qwen3-ASR path while preserving reliability of the existing desktop app.

## Current State

- Production app backend: `whisper-rs` only
- New smoke harness: `tools/qwen3_asr_smoke/` (Python, validated locally)

## Constraints

- Public repository quality standards
- Windows and Linux binaries remain first-class targets
- Russian and English must both be supported by one model
- Keep rollback path to Whisper until Qwen path is proven stable

## Phase 1: Controlled Python Validation (done)

Deliverables:
- isolated Python runner
- repeatable setup docs
- sample runs proving model load + inference path

Acceptance:
- transcribes local English and Russian files
- emits deterministic machine-readable JSON output (`--json`)

## Phase 2: Rust Backend Abstraction

Deliverables:
- define inference backend trait in Rust
- keep existing Whisper implementation as one backend
- add backend selection in config (`whisper` / `qwen_python`)

Acceptance:
- no behavior regression for Whisper mode
- app compiles and tests pass on current targets
- backend can be switched without code changes

## Phase 3: Qwen Python Bridge in App

Deliverables:
- add a Rust adapter that executes the Python harness as a child process
- map audio path -> JSON transcript result
- handle timeouts, process errors, malformed output

Acceptance:
- end-to-end app transcription works with `qwen_python` backend
- clear user-visible error messages on dependency/runtime failures
- telemetry/logging includes backend, load time, inference time

## Phase 4: Production Hardening

Deliverables:
- cache/warm model process strategy to reduce repeated load latency
- add watchdog/restart logic for child-process crashes
- benchmark RU/EN latency + quality against Whisper baseline

Acceptance:
- stable operation over long sessions
- no leaked subprocesses
- quantified quality and performance tradeoffs documented

## Phase 5: Native Runtime Options (optional, future)

Potential directions:
- ONNX Runtime based implementation in Rust when a stable export path exists
- external inference server path for deployment flexibility

Exit criteria:
- retire Python bridge only after a native path matches reliability targets

## Quality Gates for Every Phase

- keep experimental code under dedicated paths
- avoid committing local config/user artifacts
- include docs for setup and rollback
- keep production defaults conservative (Whisper fallback until proven)

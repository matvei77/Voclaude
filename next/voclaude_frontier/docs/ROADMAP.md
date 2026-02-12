# Voclaude Frontier Roadmap

## Product Principle

Local-first speech-to-text that is:
- GPU-first
- memory-bounded
- resilient under long sessions
- easy to rollback

## Parallel Deployment Model

- Keep legacy app as-is.
- Run this app in parallel as an isolated binary under `next/voclaude_frontier/`.
- Compare outputs and latency before promoting default backend behavior.

## Milestones

1. M0: Isolated runtime scaffold (complete)
- standalone Rust binary
- backend abstraction
- CUDA-mandatory Qwen backend path
- chunking controls and JSON artifacts

2. M1: One-shot backend process hardening (default mode)
- keep per-request spawn model (no persistent loaded model)
- enforce process cleanup and memory release after each request
- watchdog/timeouts for stuck inference processes

Optional (not default): warm worker mode
- lower latency by reusing loaded model
- only acceptable behind explicit opt-in because of VRAM residency

3. M2: Reliability hardening
- request timeout + cancellation
- structured error classes
- automatic fallback strategy (policy-based)

4. M3: Performance + memory envelope
- bounded chunk queue
- overlap-aware text stitching
- memory usage telemetry and alert thresholds

5. M4: Production quality gate
- batch regression over preserved recordings
- human-scored qualitative review
- promotion decision for default backend

## Non-Goals

- modifying existing `src/` app path during validation
- replacing Whisper until objective quality/stability gates are met

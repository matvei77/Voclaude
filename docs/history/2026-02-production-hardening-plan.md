# Voclaude Production Hardening Plan

## Phase 1 — Inference Performance (Highest Impact)

### 1.1 Precompute MRoPE cos/sin tables on GPU
**File:** `src/inference/candle_backend.rs` (lines 445-506)
**Problem:** `MRoPEEmbedding::forward()` pulls `inv_freq` to CPU via `to_vec1::<f64>()`, runs triple-nested loop over `batch * seq_len * half_dim`, pushes result back to GPU — on EVERY decode step across all 28 layers.
**Fix:**
- At `TextDecoder::load()` time, precompute `cos_table[max_seq_len, head_dim]` and `sin_table[max_seq_len, head_dim]` on GPU as persistent F16 tensors
- Store them in `MRoPEEmbedding` struct
- During decode, index with `Tensor::narrow(0, position, 1)` — zero CPU involvement
- Max positions: ~2500 (2048 generated + ~450 audio prefix)
- Validate transcription quality after change (f64→f16 precision shift)

### 1.2 Replace KV-cache concatenation with pre-allocated buffers
**File:** `src/inference/candle_backend.rs` (lines 609-617)
**Problem:** `Tensor::cat(&[prev_k, &k], 2)` every step = O(N^2) total memory traffic, 5600 alloc/free cycles for 200 tokens across 28 layers.
**Fix:**
- Pre-allocate `(batch, n_kv_heads, max_seq_len, head_dim)` tensors per layer at model load
- Track `cache_len: usize` counter per layer
- Each step: write new K/V into `cache[..., cache_len, ...]` via narrow+copy or slice_assign
- Attention uses `cache.narrow(seq_dim, 0, cache_len)` view
- Add `clear_cache()` method for between-transcription reset

### 1.3 Optimize GQA head expansion
**File:** `src/inference/candle_backend.rs` (lines 621-632)
**Problem:** `unsqueeze + expand + reshape + contiguous` forces GPU memory copy every step.
**Fix:**
- For single-token decode (seq_len=1): restructure attention matmul to work with compact KV shape directly
- Use `repeat_interleave` or reshape the Q heads to match KV groups instead of expanding KV

### 1.4 Skip causal mask for seq_len=1 decode steps
**File:** `src/inference/candle_backend.rs` (lines 643-651)
**Problem:** Full causal mask built on CPU every forward pass, even for single-token decode where no mask is needed.
**Fix:**
- Check `if seq_len == 1 { None }` for mask — single-token decode is trivially causal
- Only build mask during prefill (seq_len > 1)

### 1.5 Batch audio embedding replacement
**File:** `src/inference/candle_backend.rs` (lines 881-896)
**Problem:** N sequential narrow+cat operations to replace audio embeddings one at a time.
**Fix:**
- Collect all audio positions into an index tensor
- Use a single batched scatter/index_put operation

---

## Phase 2 — VRAM Management

### 2.1 Remove eager post-transcription model unload
**File:** `src/app.rs` (line 631)
**Problem:** `InferenceCommand::Unload` sent immediately after every `TranscriptionComplete`, forcing full 3.3GB reload on next request.
**Fix:**
- Remove the `inference_tx.send(InferenceCommand::Unload)` at line 631
- The existing `idle_unload_seconds` timer (lines 311-320, default 30s) already handles idle VRAM reclamation
- This alone eliminates the reload latency between consecutive transcriptions

### 2.2 Investigate CUDA memory pool retention
**Problem:** User reports 5GB constant VRAM even after unload. CUDA allocators retain memory pools.
**Fix:**
- Add VRAM logging: print `nvidia-smi` equivalent before/after load and unload
- Test whether candle-core's `Drop` for CUDA tensors actually frees memory
- If CUDA pool retained: investigate `cuMemPoolTrimTo` via FFI, or `cudaDeviceReset`
- If candle doesn't expose pool flush: consider filing upstream issue or wrapping via raw CUDA FFI
- Document findings regardless

### 2.3 Drop audio encoder intermediates before decode
**File:** `src/inference/candle_backend.rs` (in `transcribe()`)
**Problem:** Audio encoder intermediate tensors may be held alive through the decode phase.
**Fix:**
- Ensure `audio_features` is the only tensor surviving from the encode phase
- Explicitly drop intermediate conv outputs, attention matrices after encoding completes
- Scope audio encoding in a block so intermediates drop at block exit

### 2.4 Predictable peak VRAM with pre-allocated KV-cache
**Depends on:** Phase 1.2
**Result:** Pre-allocated KV-cache gives a known, fixed VRAM ceiling instead of unbounded growth.
- 28 layers × 8 KV heads × 2500 max_seq × 128 head_dim × 2 bytes (F16) × 2 (K+V) = ~286MB fixed
- Total peak: ~3.3GB model + ~286MB KV + ~200MB intermediates + ~500MB CUDA context ≈ 4.3GB

---

## Phase 3 — UI Stability

### 3.1 Fix history window viewport lifecycle
**File:** `src/ui.rs`
**Problems:**
- Close events only fire inside `show_viewport_deferred` closure during active repaints
- `ViewportCommand::Focus` sent before deferred viewport exists on first open
- `history_visible` AtomicBool desynchronizes from actual OS window state
**Fix:**
- Track viewport existence separately from desired visibility
- Queue focus commands to be sent from WITHIN the deferred callback after confirming viewport exists
- Use a command-queue architecture: all viewport state changes serialized through `UiCommand` channel
- Add `ViewportState` enum: `Hidden | Creating | Visible | Closing`

### 3.2 Fix GlobalHotKeyEvent receiver contention
**File:** `src/hotkey.rs` and wherever history hotkey is handled
**Problem:** `GlobalHotKeyEvent::receiver()` returns a global static receiver. Both recording-hotkey and history-hotkey listener threads compete for events — one may consume the other's event.
**Fix:**
- Use a single hotkey listener thread that receives ALL hotkey events
- Dispatch to appropriate handler based on `hotkey_id`
- Send dispatched events via separate channels to recording and history handlers

### 3.3 Fix settings management
**File:** `src/app.rs` (OpenSettings handler), `src/config.rs`
**Minimum fix:**
- After the user closes the external editor, re-read and validate the TOML
- On parse error: show error message in HUD with the specific error
- On success: hot-reload applicable settings without app restart
**Ideal fix:**
- In-app egui settings panel for common settings (hotkey, language, idle timeout, GPU toggle)
- Save with validation before write

### 3.4 Fix root viewport position hack
**File:** `src/ui.rs`
**Problem:** Root viewport parked at (-32000, -32000) — may be a valid coordinate on multi-monitor setups.
**Fix:**
- Use `with_visible(false)` on the root viewport builder instead of off-screen positioning
- Or set window size to (0, 0) with no decorations

---

## Phase 4 — Production Readiness

### 4.1 Remove legacy artifacts
- Delete `legacy/whisper/` directory (keep in git history)
- Remove `dist/` pre-built zips from the repository
- Clean up any dead code paths flagged by `#[allow(dead_code)]`

### 4.2 Config versioning and validation
**File:** `src/config.rs`
- Add `config_version: u32` field (default 1)
- Validate all numeric fields at load time (bounds checking)
- Fail-fast on startup with user-visible error dialog if config is invalid
- Document config schema as frozen API for deployment

### 4.3 Fix write_atomic for Windows
**Files:** `src/history.rs`, `src/session.rs`
- Before rename: call `FlushFileBuffers` on the temp file handle
- Handle `ACCESS_DENIED` on rename with retry logic
- Consider using `ReplaceFileW` Win32 API for truly atomic replacement

### 4.4 Add test infrastructure
**New files:** `tests/` directory
- Unit tests:
  - MRoPE correctness (compare precomputed GPU values against CPU reference)
  - KV-cache pre-allocation and cursor behavior
  - Config parsing, migration, validation
  - History store: append, retention, corrupt recovery
  - Session state machine transitions
- Integration tests:
  - Full pipeline: load model → transcribe reference WAV → verify output text matches expected
  - Timing regression: `--test` mode should report per-phase timing (mel, encode, decode)

### 4.5 Inference worker watchdog
**File:** `src/app.rs`
- Detect if inference worker thread has panicked
- On panic: log error, show HUD message, respawn worker thread
- Prevents permanent app failure from CUDA OOM or other runtime errors

### 4.6 Model integrity and distribution
- Pin model files with SHA256 checksums
- Support `--model-dir` flag for IT to pre-stage models (no HF download in production)
- Verify checksums at load time, fail-fast if mismatch

### 4.7 Build and packaging
- Version-stamp binary: embed `CARGO_PKG_VERSION` + git hash in HUD and `--version` output
- Add `--validate` CLI flag: check GPU, model, audio device, config without starting app
- Improve `package.ps1`: deterministic, reproducible builds with version in artifact name
- Consider MSI installer via `cargo-wix` for enterprise SCCM/Intune deployment

### 4.8 Structured logging
- Add file-based logging to `%LOCALAPPDATA%\Voclaude\logs\`
- Log rotation (keep last 5 files, 10MB each)
- Log all errors, model load/unload events, transcription timing, VRAM usage
- Essential for post-deployment diagnostics when "no ability to change"

---

## Execution Order

```
Phase 1.1 (MRoPE precompute)     ← highest single impact, do first
Phase 1.2 (KV-cache pre-alloc)   ← second highest, enables Phase 2.4
Phase 2.1 (remove eager unload)  ← trivial, one-line change
Phase 1.4 (skip causal mask)     ← small, easy win
Phase 1.3 (GQA optimization)     ← moderate effort
Phase 1.5 (batch embeddings)     ← moderate effort
Phase 2.2 (CUDA memory investigation) ← research task
Phase 2.3 (drop intermediates)   ← small change
Phase 3.1 (history viewport)     ← critical stability fix
Phase 3.2 (hotkey contention)    ← critical stability fix
Phase 3.3 (settings validation)  ← important UX fix
Phase 3.4 (root viewport)        ← small fix
Phase 4.1 (cleanup)              ← quick wins
Phase 4.2 (config versioning)    ← important for deployment
Phase 4.3 (write_atomic)         ← important for reliability
Phase 4.4 (tests)                ← ongoing, start with MRoPE tests
Phase 4.5-4.8 (deployment infra) ← final polish
```

## Success Metrics
- Token generation: 5-10x measured speedup (validate with --test timing)
- VRAM idle: <500MB (model unloaded)
- VRAM active: <4GB peak (F16 model + pre-allocated KV)
- History window: 100% reliable open/close/restore
- Settings: validated, no silent failures
- Tests: coverage for all critical paths
- Build: single reproducible artifact with version stamp

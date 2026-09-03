# Voclaude Goal — "Stop, and the text is already there"

**Budget:** 10 hours of autonomous work. **Hardware:** RTX 3060 12 GB, Windows 11.

## The goal in one sentence

Press the hotkey to stop, and the finished transcript is on the clipboard within 3 seconds,
no matter how long you dictated, in whatever mix of Russian and English you spoke,
with every word already safe on disk before you ever pressed stop, and with nothing
resident on the GPU or in RAM when you are not dictating.

## Baseline (measured today, 2026-09-03, from the live app's logs)

| Metric | Today |
|---|---|
| Stop → clipboard, 4.5-min dictation (3172 chars) | 35.4 s (2.2 s model load + 1.5 s encode + ~31 s decode) |
| Stop → clipboard, 3.5-min dictation | 28.5 s |
| Stop → clipboard, 15-min dictation | ~2 min extrapolated (3 sequential 5-min chunks) |
| Stop → clipboard, 5-second clip, model cold | ~3.5 s (dominated by model load) |
| Decode speed | ~25 tokens/s (Qwen3-ASR-1.7B, F16, unquantized) |
| VRAM while model loaded | ~5.7 GB |
| VRAM idle | 0 (model unloads 300 s after last use, CUDA context dropped) |
| RAM idle | 138 MB working set, 1.0 GB private bytes |
| Loss on crash mid-dictation | Audio is safe on disk. Transcript: nothing exists until the whole recording is re-run. |
| Multilingual quality | Unmeasured. Server research rated Qwen3-ASR-1.7B "retire it": slowest but one, worst on proper nouns. No WER data for RU/EN mixing anywhere. |

The whole wait is decode time that happens *after* you stop. Speech is ~9x faster than
realtime to transcribe, so the fix is to transcribe *while* you are still talking.

## Targets (acceptance criteria)

| Metric | Target | How verified |
|---|---|---|
| Stop → clipboard, any dictation length up to 30 min | ≤ 3 s (p95) | 15-min live dictation, stopwatch + log timestamps |
| Stop → clipboard, short clip (< 15 s), model already loaded | ≤ 1.5 s | log timestamps |
| Stop → clipboard, short clip, model cold | ≤ 4 s | log timestamps |
| Decode speed | ≥ 45 tokens/s | `--bench` harness on stored recordings |
| VRAM while loaded | ≤ 3.5 GB | nvidia-smi during transcription |
| VRAM idle | 0 within 60 s of last use (default `idle_unload_seconds` = 60) | nvidia-smi |
| RAM idle | ≤ 60 MB working set | Task Manager / Get-Process |
| Loss on power cut / kill mid-dictation | All completed segments' text already on disk; at most the last ~30 s of speech needs re-transcription; recovery is automatic on next start | `taskkill /F` during a 5-min dictation, then relaunch |
| Loss on transcription failure (OOM, panic) | Zero. Failed segment retried; if still failing, audio + partial text preserved and surfaced in tray | fault injection in the inference worker |
| Multilingual quality | Measured WER on a 20-clip personal set (RU / EN / mixed) for every candidate; default model chosen by WER first, speed second; language switching inside one dictation keeps the right script | bench harness output committed to `docs/bench/` |

Single-digit-MB idle RAM is **not** a target. It requires replacing egui/eframe with raw Win32,
which is a separate project. 60 MB is the realistic floor with the current UI stack.

## Plan (10 h)

### 0. Ground truth (0.5 h)
- Build and test the current uncommitted work (chunking, model tiers, audio retention), commit it.
  The running binary is from Aug 26 and predates it.
- Add `voclaude --bench <dir>`: runs every stored `.f32` recording, prints per-phase timing
  (load / mel / encode / prefill / decode tok/s / total) and WER against a reference text
  where one exists. This is the yardstick for every later change.
- Assemble the 20-clip personal test set from existing recordings + history (44 Russian,
  10 mixed entries already on disk).

### 1. Streaming transcription (4 h) — the UX win
- **Preload on record start.** Hotkey → recording starts *and* model load starts in parallel.
  Load (2–4 s) is hidden behind the first seconds of speech.
- **Segmenter.** Writer thread tracks RMS energy; cuts a segment at the first pause ≥ 0.5 s
  after 20 s of audio, hard-cuts at 30 s. Each segment is a byte range of the same `.f32`
  file already on disk.
- **Incremental inference.** Inference worker transcribes segments as they close, while
  recording continues. Previous segment's text is passed as the Qwen context/system slot so
  language and vocabulary carry across segment boundaries.
- **Tail-only finish.** On stop, only the last open segment is transcribed, then all segment
  texts are joined, formatted, copied. This is where 35 s becomes ≤ 3 s.
- **Progress in the HUD/tray:** "Recording · 4:12 · 3 segments done".

### 2. Decode speed and VRAM (2 h)
- Quantize decoder weights to Q8_0 on first load (Candle's CUDA quantized kernels are
  already compiled in and unused). Cached as GGUF next to the safetensors. Expected:
  ~2x decode speed, ~3 GB VRAM. Fall back to F16 automatically if the bench shows WER regression.
- Skip causal-mask construction for single-token steps; remove unnecessary host syncs in the
  decode loop.

### 3. Idle footprint (1 h)
- Default `idle_unload_seconds` 300 → 60. Verify 0 VRAM and no lingering CUDA host
  allocations after unload (1 GB private bytes today).
- Trim idle RAM: stop the 100 ms keepalive repaint when nothing is visible, cap the in-process
  log ring, drop decoded tray icons that aren't in use.

### 4. Never lose a word (1.5 h)
- Each finished segment's text is appended to `transcripts/<session>.txt` immediately
  (atomic append, fsync). The session journal records which segments are done.
- Startup recovery resumes from the last completed segment instead of re-running the whole file.
- Per-segment retry on failure (OOM → unload, reload, retry once). If a segment still fails,
  the session is kept, text so far is copied to the clipboard, and the tray shows
  "Recover last recording" which re-runs only the missing segments.
- Test: kill the process during a 5-min dictation, relaunch, confirm full text recovered.

### 5. Model decision (1 h)
- Run the bench on: Qwen3-ASR-1.7B F16, Qwen3-ASR-1.7B Q8, Qwen3-ASR-0.6B.
  If time allows, Whisper large-v3-turbo (the only alternative with a Candle implementation;
  server research found it fastest among punctuated models at ~4.8 GB).
- Pick the default by WER on the mixed RU/EN clips, then speed. Record results in `docs/bench/`.
- Note: GigaAM (best Russian on the server) is Russian-only and unpunctuated; not a fit for a
  bilingual dictation tool without a second model. Out of scope for 10 h.

## Non-goals
- New UI, settings panel, installer.
- Sub-10 MB idle RAM (egui rewrite).
- Porting a non-Candle model architecture.

## Order of work
0 → 1 → 2 → 4 → 3 → 5. Streaming first because it removes the wait regardless of model speed;
quantization second because it makes segments finish with margin and halves VRAM; durability
before idle tuning because it is a core promise; model bake-off last because it needs the
bench and the streaming pipeline to be meaningful.

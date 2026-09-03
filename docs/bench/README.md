# Bench results

All runs: RTX 3060 12 GB, Windows 11, Qwen3-ASR-1.7B via Candle 0.9, greedy
decoding, 12 personal dictation recordings (29 min total: English, Russian and
mixed), `voclaude --bench bench_data ...`. There is no hand-labelled
reference, so "quality" below means agreement with the unquantized F16
transcript plus a manual read of the differences. Hypotheses live next to the
audio as `<file>.<tag>.hyp.txt` (audio is not in the repo).

## 2026-09-04

| Tag | Change | Decode tok/s | Whole-set speed | Load | Text vs F16 official |
|---|---|---|---|---|---|
| `legacy_prompt` | pre-existing prompt ("You are a helpful assistant" + instruction) | 33.6 | 16.7x realtime | 2.0 s | baseline (identical bar 6 words on 3 files) |
| `official_prompt` | model's own chat template, empty system slot | 33.8 | 16.3x | 2.0 s | fixes "Free"→"three", "unallocate"→"allocate" |
| `fused` | fused rms-norm / rope / softmax, merged QKV and gate-up matmuls, no K-cache copy | **72.0** | 26.7x | 2.1 s | identical (2 one-word diffs) |
| `q8` | + Q8_0 decoder weights (ggml kernels) | **98.3** | 31.9x | 9.8 s → 4.5 s after CPU-side quantization | punctuation-level diffs only |
| `q4k` | Q4_K decoder weights | 127.5 | 42.5x | 38 s | **rejected**: lost most of a 10-min file, "you"→"we" swaps |
| `q8b` / `dyn` | Q8_0 quantized on the CPU from the mmapped weights, uploaded once; CUDA libraries loaded at runtime | 98.4–99.0 | 31.9x | 4.4 s | identical to `q8` |
| `gguf` | quantized projections cached as GGUF under `%LOCALAPPDATA%oclaude\Voclaude\cache\weights` | 105 | – | **1.4 s** (5.8 s on the run that writes the cache) | identical |

### Model bake-off: Whisper large-v3-turbo (`--engine whisper`, F32, 30 s windows, per-window language detection)

| | Qwen3-ASR-1.7B Q8_0 | Whisper large-v3-turbo |
|---|---|---|
| Whole set | 31.9x realtime, 99 tok/s | 27.7x realtime, 149 tok/s (encoder dominates: every window is padded to 30 s) |
| Mixed RU/EN recording (4.5 min with a Russian passage) | Russian passage transcribed correctly, script switches back to English seamlessly | **Russian passage lost** (0 Cyrillic characters in the output; whole window detected as "en") |
| English dictation | full casing and punctuation | some windows come out lowercase and unpunctuated; drops short phrases ("top right", "a bit"); "Vo Cloud" → "war cloud" |
| Silence / non-speech file | empty output | hallucinated "you you you you" |
| Load | 1.4 s from the GGUF cache (4.4 s when quantizing) | 0.75 s |

Verdict: Qwen3-ASR-1.7B stays the default. Whisper turbo is faster to load and decode but fails the core
requirement of this app (seamless Russian/English mixing) and is noisier on English dictation.
Whisper support remains bench-only (`src/inference/whisper_engine.rs`).

Stream simulation (`--stream`, segments 20–45 s cut at pauses, F16 unfused):

| Variant | Stop-to-result latency (mean / max) | Notes |
|---|---|---|
| no context | **1.24 s / 2.50 s** | default |
| previous-segment context in system slot | 1.28 s / 2.72 s | fixed a few names ("Vermeer") but injected a fabricated 40-word sentence at one boundary and turned "three" into "free" — off by default |

Segmented transcription also punctuates long recordings better than one-shot
whole-file decoding, which tended to drop punctuation for long stretches.

## Live app (same day, final build)

| Metric | Before | After |
|---|---|---|
| Stop → clipboard, 4.5-min dictation | 35 s | 0.3–2.4 s measured live (1.2 s mean / 2.7 s max in the stream simulation over the set) |
| Decode speed | 33.6 tok/s | 99 tok/s |
| VRAM while loaded | ~5.9 GB | ~3.6 GB (Q8_0) |
| VRAM idle | 0 after 300 s | 0 after 60 s (worker process exits) |
| RAM idle (tray process) | 138 MB WS / 1.0 GB private after unload | **20 MB WS / 8 MB private** (79 MB with the history window open) |
| Crash mid-dictation | transcript lost until full re-run | finished segments journaled; relaunch resumes from the last one (tested with a hard kill) |
| Inference crash | took the app down | child process; app restarts it on the next segment (tested by killing the child mid-recording) |

Details:

- 4.5-minute dictation before this work: 35 s from stop to clipboard.
- After: model loads while recording starts (2–4.5 s, hidden), segments are
  transcribed during recording, stop-to-result 0.3–2.4 s in live tests.
- Kill test: process killed 48 s into a recording; on relaunch the two finished
  segments were reused, the tail was segmented and transcribed, clipboard
  ready 2.4 s after launch including model load.

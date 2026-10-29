# Roadmap and assessment (2026-10-29)

This document records where Voclaude stands after the Parakeet Ultra switch and
which improvements are worth the effort. Every claim with a number points to a
measurement in `docs/bench/` or to the live app's logs. Estimates are marked.

## Where the app stands

**Speed is at the floor.** The user-visible wait is the time to transcribe the
last open segment after the stop key. On the RTX 3060 the logs of 2026-10-29
show 0.04 to 0.56 s for 40 dictations, with one outlier at 1.32 s. The Parakeet
decoder costs about 70 ms per segment on the GPU and the encoder about 46 ms
per 12 s of audio (`docs/bench/2026-10-23-model-survey.md`). Faster hardware
cannot remove much more. On the CPU (i5-13600K, 6 threads) a 20 s segment
takes about 0.8 s. Model load (0.8 to 2.1 s on Vulkan) overlaps the first
seconds of speech because the worker starts when the recording starts.

**Quality is good for dictation, not state of the art.** On FLEURS read speech
Parakeet Ultra Q6_K scores 5.1 (ru), 4.7 (en), 4.7 (de), 5.0 (fr) and 7.3 (nl)
percent word error rate. Cloud models score 2 to 4 times lower on the same
clips. In a blind reading of a real 58-minute Russian and English meeting,
readers preferred Parakeet over Qwen3-ASR-1.7B in 45 segments, Qwen in 24,
and could not tell in 54. Mixed Russian and English inside one recording is
the feature that no cloud model handles reliably, and that is what the
segmenter plus `parakeet_recheck` are built for.

**The tray is the only Windows feedback surface.** Notifications are debug
logs on Windows. The History window shows status, but only when open.

## Improvements, in the order they pay off

### 1. Cleanup pass with a small local language model

The model survey measured this on 2026-10-23. Spokenly's `cleanup-1`
(Qwen3-0.6B fine-tune, Q8_0) removed fillers, false starts and stutters in
English and Russian, kept English terms inside Russian, left questions and
injected commands alone, and tripped none of the output guards. Cost: 115 ms
median on the RTX 3060 (0.5 s for 80+ words), 0.53 s on the CPU (2.6 s for
80+ words). Generic chat models (Qwen3 1.7B to 4B, GigaChat) translated mixed
sentences or obeyed injected instructions and are not suitable.

Design that fits the current pipeline:

- Run the cleanup per segment while the user is still speaking, exactly like
  transcription. After the stop key only the last segment is cleaned, so the
  added wait is about 0.1 s on the GPU and 0.5 s on the CPU.
- Guards from the open-source dictation apps: at least 75 percent of output
  words must come from the input, length between 0.5 and 1.3 times the input,
  Cyrillic share within 15 points. A guard failure pastes the raw transcript.
- Pure Rust: `candle-transformers` 0.9.2 ships `quantized_qwen3`, so the GGUF
  runs in the existing worker without a new native dependency. Memory: about
  0.7 GB extra while loaded.
- A config toggle (`cleanup = true`) and a tray check item. Off by default on
  CPU-only machines until the CPU numbers are confirmed on a laptop.

Risk: Spokenly states Russian quality below English. The one Russian change the
survey saw was a capitalization error. A hand-corrected set of 20 mixed
dictations (see item 6) is the way to decide the default.

### 2. Paste at the cursor

Today the text lands on the clipboard and the user presses Ctrl+V. Most
dictation tools type the text into the focused window. Implementation on
Windows: after the clipboard write, send Ctrl+V with `SendInput`, then restore
the previous clipboard content. One config switch (`paste_on_finish`) and one
tray check item. Risk: terminals and some editors map Ctrl+V differently; keep
the switch off by default for the first release.

### 3. Settings window

The 2026-09 priority list asked for a small control panel. The History window
already runs as a deferred egui viewport, so a second viewport with the same
pattern is the cheapest route. Fields: record hotkey, history hotkey, engine,
GPU or CPU, Parakeet quantization (Q4_K_M, Q6_K, Q8_0), `parakeet_recheck`,
capitalization and trailing space, streaming, API enabled and port, idle
unload, start with Windows (the `Run` registry key the installer writes).
Save writes `config.toml` and sends `ConfigReloaded`, which the app already
applies without a restart for most fields.

### 4. API for other programs

The local API already accepts files. Three additions make it useful for agents
and other desktop tools:

- `GET /v1/live` as server-sent events: every transcribed segment of the
  current dictation as it finishes, plus the final text. This is the "hook my
  own tool into dictation" case.
- `GET /v1/history?limit=N`: the entries the History window shows.
- Optional LAN binding (`api_bind = "0.0.0.0"`) with `api_token` required when
  the bind address is not loopback, so a laptop can use the desktop's GPU.

Compatibility stays: the OpenAI transcription route and `voclaude transcribe`
do not change.

### 5. Language-boundary cuts

transcribe.cpp exports a language-ID module. The survey showed that cutting
where language ID sees a Russian to English change fixes Qwen's translation of
the real mixed passage and costs Parakeet nothing (177x instead of 182x).
Language ID must not force the language: on accented English it labels many
segments as Russian, German or French. This is a segmenter change plus one
config switch.

### 6. A hand-corrected code-switching set

`parakeet_recheck` rests on one real positive example. Twenty natural Russian
and English mixed dictations with corrected transcripts would decide the
re-check thresholds, the cleanup default, and whether language-boundary cuts
help on real speech. The recordings and the probe scripts already exist under
`D:\voclaude-model-eval` (local only).

### 7. Visible feedback on Windows

Toast notifications need an Application User Model ID on the Start Menu
shortcut. The installer can set it, and `notify-rust` supports Windows toasts.
Until then every tray action either shows something (a window, a file, a
folder) or is disabled, which is what the 2026-10-29 menu rewrite did.

### 8. Code signing

The installer is unsigned, so SmartScreen shows "Windows protected your PC" on
first run. A code-signing certificate removes that prompt for non-technical
users. This is a purchase and a `signtool` step in `package.ps1`, not code.

### 9. macOS

The Metal build compiles in CI and the Qwen engine was measured on an M1
runner (4.6x realtime for 1.7B Q8_0). The tray, hotkey and egui code still
needs a main-thread run loop on macOS, and the Parakeet path through
transcribe.cpp's Metal backend is not measured. This is a separate project.

### 10. Laptop numbers

The CPU figures come from a desktop i5-13600K. A Core Ultra laptop (CPU, Arc
iGPU through Vulkan, NPU) was not available. The survey's estimate is 10 to
15x realtime on the CPU. Measuring it decides the default `cpu_threads`.

## Performance headroom in the current engine

- `cpu_threads`: 6 threads beat transcribe.cpp's default on the hybrid
  i5-13600K (20x against 18x). The default is still 0 (automatic). Setting the
  default to the number of physical performance cores needs a CPU-topology
  query; the config field already exists for manual tuning.
- `idle_unload_seconds` (60): the worker exits after a minute, which costs the
  0.8 to 2.1 s model load on the next dictation, hidden behind speech. Users who
  dictate in short bursts can raise it; the cost is about 1 GB of VRAM or RAM
  while loaded.
- Quantization: Q6_K equals Q8_0 on FLEURS and in the blind reading, and runs
  at the same CPU speed as Q4_K_M (40 percent faster than Q8_0). Nothing
  faster with equal quality was found.
- `parakeet_recheck` costs about 5 percent of GPU time on real recordings. The
  CPU cost is not measured.

## Not recommended (evidence in `docs/bench/`)

- Whisper large-v3-turbo: dropped the Russian passage entirely, hallucinated
  on silence (2026-09-04).
- Qwen3-ASR Q4_K: lost most of a 10-minute file (2026-09-04).
- `segment_context` for Qwen: fabricated a sentence at a segment boundary.
- Forcing the language from language ID: mislabels accented English.

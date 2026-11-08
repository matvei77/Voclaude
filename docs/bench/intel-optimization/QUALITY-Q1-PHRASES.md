# Quality candidate Q1-phrases-b3: names and terms for Parakeet

2026-11-08, desktop Opus quality track. Status: **merged to main and installed on the desktop** (see the ack).
Rebased onto main 0.6.0 (`4ea1ad4`); the outputs are unchanged. Model weights and precision are
unchanged (Parakeet Ultra Q6_K).

## What changes

- **New setting `parakeet_phrases`** (default empty: no change). Each entry is a term, optionally
  with spoken forms: `"Hemingway = Hemengwe, Megway"`. Spoken forms are spellings Parakeet writes
  for the term when it mishears it.
- **Native: `transcribe.dll` B3** (patch `native/parakeet-phrase-bias-b3.patch` on
  transcribe-cpp-sys 0.4.1). Parakeet TDT greedy decoding now honors the existing C API field
  `RunOptions::vocabulary`. No ABI change. The bias is bounded:
  1. It acts only at a step where the acoustic argmax is not blank. Blank decisions and durations stay acoustic.
  2. It only re-ranks the model's own top-8 tokens. A token that continues a list spelling gets +4 logits.
  3. A spelling starts only where the acoustic best token also starts a word with the same first-letter case. A start piece other than the acoustic best needs 2 or more letters.
  4. A partial match ends after 3 encoder frames without progress, counted from the end of the last emitted token.
  5. The logits are not modified. The reported token confidence `p` stays the entropy of the unbiased distribution.
- **App: phrase re-check** in `src/inference/parakeet.rs`, after the Russian re-check:
  1. Candidates are Latin first-pass words that are shaky (the Russian re-check rule) or have
     `p < 0.97` and letter similarity >= 0.6 to a list spelling. A segment without candidates gets no extra decode.
  2. One extra decode of the same segment runs with the list as `vocabulary`.
  3. An edit replaces 1-3 candidate words with the term only when one of these is true:
     - The biased decode spells the term, or a spoken form, over the same time span, and the old words are >= 0.4 similar to that spelling.
     - The old words are exactly a spoken form.
  4. Russian re-check replacements take precedence. A change of case alone is never made. The term keeps its exact spelling.
  5. With the official DLL (no vocabulary support) only exact spoken forms apply, and the app logs one warning.

## Invariants verified

| Check | Result |
|---|---|
| Unpatched rebuild B0 vs official `transcribe.dll` | byte-identical tokens, times and `p`, 153 pieces, Vulkan and CPU |
| B3 with empty vocabulary vs official | byte-identical, Vulkan and CPU |
| Q1 exe, no phrases vs Q0 exe (master `7fe28c3` + official DLL), Russian re-check on | byte-identical text: 296/296 pieces on Vulkan, 153/153 on CPU |
| Rust worker vs Python prototype of the acceptance rules | 296/296 pieces |
| Q1 no phrases vs Q0, whole clips through the app's own segmentation (`segment_all`) | identical text: 195/195 clips |
| Unit tests (`cargo test --release`, on main 0.6.0) | 50 passed (7 new) |

## Results

Sets (private, see manifest below):

- **names55:** 55 personal clips, 60 min, 153 pieces. 25 labelled misspellings of 7 list terms.
- **neg140:** 40 FLEURS-en, 40 FLEURS-ru, 20 synthetic RU/EN switch clips (human references), and
  40 personal clips without list terms. 31 min, 143 pieces.

The list has 16 terms. Spoken forms come from the user's history, leave-one-clip-out: a clip never
uses a form seen only in its own history entry.

| Config | names55 fixed (GPU) | names55 fixed (CPU) | False edits names55 | Edits on 120 human-reference clips | Edits on 20 personal negatives |
|---|---|---|---|---|---|
| Q0 | 0/25 | 0/26 | - | - | - |
| Q1, official DLL (exact spoken forms only) | 15/25 | not run | 0 | 0 | 0 |
| **Q1, B3 (bias + spoken forms)** | **21/25** | **20/26** | **0** | **0** | **1 (a correct fix)** |

The rows above use the manifest pieces. Through the app's own segmentation (whole clips, `segment_all`) on the GPU:

- names55: 19 edits, 17/20 labelled misspellings fixed, 0 false edits.
- neg140: 1 edit, the same correct fix.

The first pass differs between the two segmentations, so the edit sets differ a little.

Per term on the GPU (pieces): Qwen 10/10, Codex 4/4, Claude 2/2, a person name 1/1, Hemingway 4/6,
Parakeet 0/1, Krea 0/1.

- **Mixed EN/RU:** Cyrillic text is never replaced. FLEURS-ru and all 20 switch clips have no edit, so
  their error rates are unchanged.
- **False edits:** 0. Opus judged every edit from context. The user has not reviewed the labels.
- **Remaining misses and causes:**
  - 2 Hemingway misspellings: the bias did not reach the term, and leave-one-out removed the only exact spoken form.
  - Parakeet: its only spoken form comes from the same clip.
  - Krea: the audio starts with "cr", and the case rule blocks the change of first-letter case.
- **What the bias alone cannot do:** when the user says a term unlike its spelling (Hemingway said
  "Hemengway"), the needed token is about 6 logits below the model's choice. A spoken form fixes these cases.

## Cost (coarse, for the laptop timing track; not certified)

- The extra decode is a full run of the segment (encoder and decoder). It ran on 53 % of the
  names55 audio and 12 % of the ordinary neg140 audio.
- RTX 3060 desktop, segment wall time through the worker:

  | Set | Device | Without list | With list |
  |---|---|---|---|
  | names55 | Vulkan | 36.4 s | 56.6 s |
  | neg140 | Vulkan | 20.2 s | 21.9 s |
  | names55 | CPU | 165.2 s | 260.9 s |

- A decode-only second pass from the first pass's encoder output would remove the encoder cost.
  It needs a native API addition. It is not built.
- Per-request profiling: set `VOCLAUDE_WORKER_PROFILE=1` for the worker.
  - The worker then logs at DEBUG and writes `voclaude-profile {json}` records with no transcript text.
  - Each record has sample counts and the times of the first decode, the Russian re-check and the phrase re-check.
  - The record schema is in `OPUS-QUALITY-ACK.md`.
- Other telemetry, debug level only:
  - App: `Phrase re-check: N edit(s), biased decode run|skipped, X ms`.
  - Native: `parakeet phrase bias: ... boosted steps=…, flips=…`.
  - Evaluation-only environment overrides: `TRANSCRIBE_PARAKEET_BIAS_BONUS`, `_TOPK`, `_GAP`, `_TRACE`.

## Reproduction (desktop)

```powershell
# Native B3: patch transcribe-cpp-sys 0.4.1 (crates.io source) and build transcribe.dll
git -C <src> apply docs\bench\intel-optimization\native\parakeet-phrase-bias-b3.patch
docs\bench\intel-optimization\native\build-transcribe-dll.cmd <src> <build>   # MSVC 2022 Build Tools, Ninja
# Use <build>\bin\transcribe.dll with the OFFICIAL ggml*.dll from vendor\transcribe-prebuilt\bin

# App: separate target dir; TRANSCRIBE_DIR must be set or cargo rebuilds transcribe.cpp from source
$env:TRANSCRIBE_DIR = "$PWD\vendor\transcribe-prebuilt"; $env:CARGO_TARGET_DIR = "$PWD\target\quality"
cargo build --release      # 0.6.0 default build: no CUDA
cargo test  --release
```

The evaluation scripts and private data are in `D:\voclaude-eval` on the desktop:
`worker_eval.py`, `token_dump.py`, `accept.py`, `make_neg.py`, `make_manifest.py` and `make_results.py`.

## Open items

0. Builds (hashes in the ack):
   - The evaluated build was `9a929490…` with B3 `4234e7ae…`.
   - The merged build on main 0.6.0 is `003072cc…`, with the `build-transcribe-patched.ps1` DLL `a3158db3…`.
   - `config.example.toml` and the README document `parakeet_phrases`.
   - `build-transcribe-patched.ps1` (repository root) builds the patched DLL from the crate source and this patch.
     Every build has a new DLL hash: the MSVC link timestamp and the source paths are compiled in. Two script builds
     (`a3158db3…`, `530f68e8…`) gave outputs byte-identical to B3, and with an empty list to the official DLL.
   - `package.ps1 -PatchedTranscribe` bundles that DLL in the zip and installer. Without the switch, the official DLL ships.

1. The user reviews the term labels and the edit list (private results file).
2. The user's list is the draft from their own history: 13 entries, 7 with spoken forms. It made no edit on
   the 140 negative/reference clips and fixed all 25 labelled misspellings, which are in-sample.
   Planned next: commands to add, check and suggest entries.
3. Russian speech that contains an English name is not handled. Shaky runs need a mostly-Latin segment.
4. The laptop measures Arc/CPU cost. The laptop also confirms the B3 identity on Arc, comparing B3 with
   an empty list against the official DLL.

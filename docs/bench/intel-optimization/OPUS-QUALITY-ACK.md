# Desktop Opus acknowledgement: quality track

2026-11-08. Desktop Opus (Claude Code, this repository) confirms the ownership split in
`QUALITY-PERF-HANDOFF.md`. Results: `QUALITY-Q1-PHRASES.md`.

## Ownership

- **Desktop Opus owns:** names and phrase context, decoding quality, acoustic confidence,
  Russian/English preservation, and the 55-clip names/mixed evaluation set.
- **Laptop Codex owns:** Intel Arc/NPU kernels, harnesses, timing and memory.
- Desktop Opus does not edit `tools/intel_perf*`.

## Constraints accepted

- Keep the existing models. Parakeet Ultra Q6_K weights and precision stay frozen. No model
  search, training, pruning or new quantization.
- Do not deploy. Do not change the installed app or the user config on either machine.
- No publishing, no extra subagents, no recordings to external services.
- No claim about whole-app 2 GB memory or NPU ASR from the quality track.

## Shared files changed by Desktop Opus (working tree, not committed)

Do not edit these files until desktop Opus hands them back.

| File | Change |
|---|---|
| `src/inference/parakeet.rs` | Phrase re-check after the Russian re-check (`phrase_edits` and its helpers). `recheck_russian` now returns its replacements (`russian_replacements`). `splice` takes `Edit`s. 7 new unit tests. |
| `src/config.rs` | `parakeet_phrases: Vec<String>`, default empty. |
| `src/worker.rs` | A change of `parakeet_phrases` restarts the worker (1 line). Profile records for preload, segment and file requests (`VOCLAUDE_WORKER_PROFILE`). |
| `src/main.rs` | Taken over at the laptop's request: `mod profile`; the `--worker` log level is DEBUG when `VOCLAUDE_WORKER_PROFILE` is set, INFO otherwise (unchanged default). |
| `src/profile.rs` (new) | `enabled()`, `emit()` and `ms_since()` for the profile records. |
| `src/app.rs` | 1 line: settings reload copies `parakeet_phrases`. |
| Native Parakeet decoder (transcribe-cpp-sys 0.4.1: `decoder.cpp`, `decoder.h`, `model.cpp`) | `native/parakeet-phrase-bias-b3.patch`, 300 lines. Parakeet TDT supports the existing `vocabulary` field. No ABI change. |

## Invariants: status

1. Empty phrase list: the candidate reproduces Q0 exactly. **Verified** on Vulkan (296 pieces) and
   CPU (153 pieces), with the Russian re-check on.
2. Token times, blank/duration rules and Russian re-check semantics are unchanged. **Verified by
   construction and by test:**
   - The bias acts only on non-blank steps.
   - Durations come from the unbiased head.
   - Russian edits take precedence.
3. Reported token `p` stays acoustic. **Verified by construction:** the decoder never writes to the
   logits. `p` is the entropy confidence of the unbiased distribution, also in the biased decode.
   The app keeps the first pass's tokens and `p` for every unreplaced word.
4. Phrase replacement is bounded:
   - Only candidate words (shaky, or `p < 0.97` and spelled like a list entry).
   - 1-3 words per edit.
   - Only complete terms, backed by the biased decode or an exact spoken form.
   - No change of case alone.

## Reference IDs

**Rebase (2026-11-08):** local `master` was behind GitHub. `origin/master` is `4ea1ad4` (0.6.0, PR #4).
0.6.0 changes nothing under `src/inference`, `src/worker.rs` or `src/audio`. The quality work now sits on
branch `parakeet-phrases`, based on `4ea1ad4`.

**Q0 is now `4ea1ad4` + the official DLL.** Its outputs equal the `7fe28c3` reference outputs: the rebased
candidate without phrases reproduces them byte for byte on 296 pieces and 195 whole clips (GPU). With phrases,
the rebased candidate reproduces the evaluated outputs on the same sets.

| ID | voclaude.exe sha256 | transcribe.dll sha256 | Settings |
|---|---|---|---|
| Q0 (master `7fe28c3`, `--no-default-features --features cpu`) | `2904213255e2b1b5d7f5d86029e438119da763583c75c92e0e769cbcde9a20ef` | official 0.4.1 `c00cbe50a845003f986d3ed0ee149c08f576d0f3cc54c5346033608185a000d5` | `parakeet_recheck = true`, no phrases |
| Q1-phrases-b3 (evaluated, `7fe28c3` + changes) | `9a92949058dd01c8c6038580632fa18caaf11fa5fc49f1aa0463baa4d09a2cda` | B3 `4234e7aed27b6c266a2911e9261b59dbfe7d430b96c79380dc1026454123b305` | `parakeet_recheck = true`, 16 terms + spoken forms (leave-one-clip-out) |
| **Q1 on main 0.6.0** (branch `parakeet-phrases` on `4ea1ad4`, default features = no CUDA) | `003072cc684fd7299b61ea436662f32c439dfa2032b130c5c69b4d730c4739f0` | `build-transcribe-patched.ps1` output `a3158db35f504e9d9761e140bb097b41147588970e26b6cb2238e04929fed94a` (same patch as B3; every build has a new hash: the MSVC link timestamp and compiled-in source paths. Outputs verified identical to B3) | same |

Common to both:

- **Model:** `parakeet-ultra-Q6_K.gguf`, sha256 `ba146253d9b5eb7c729dfa897174cb5bad0a7d05129863805e0caf7d571c7493`.
- **ggml:** the official ggml DLLs:

  | DLL | sha256 |
  |---|---|
  | `ggml.dll` | `7f4312af8bf5728c72f342702f5674d4a40cc3964f01d67e8b01db7559a2b530` |
  | `ggml-base.dll` | `71050124f5ff1a7f023b5fa723c121acadfd54fd1301664c1077116313f9ce56` |
  | `ggml-vulkan.dll` | `d4ef14886f4fd5a0850acc81ed4daf77e6c80f39db1bb7e8936ba95930b7e68d` |

- **Installed desktop app:** unchanged (`da5d4bd1…`).
- **Rebuild control:** an unpatched rebuild of 0.4.1 (B0, sha256 `63d5cf4f…`) is byte-identical in
  output to the official DLL on Vulkan and CPU.

**Note for the pinning contract:** the evaluated exe was `cbbe0644…`. The pinned exe `9a929490…` adds:

- test-only and doc-comment edits;
- CRLF→LF line endings in `config.rs` and `parakeet.rs` (git normalizes them, the diff is content only);
- telemetry only: the debug log says `run` only when the biased decode succeeded, and the opt-in
  `VOCLAUDE_WORKER_PROFILE` records (below).

Earlier rebuilds also changed the hash without a code change (`ba301f2c…`). Outputs of `9a929490…` are
byte-identical to the evaluated exe on all 296 GPU pieces and all 195 whole clips, with and without phrases,
and with the profile flag on or off. Pin and compare outputs, not only exe hashes.

## Deliverables

- [x] **Corpus manifest** (private, contains personal transcripts):
  - Manifests: `D:\voclaude-eval\handoff-q1-phrases-b3\manifest\names55-v1.json` (sha256 `47c3733c…`) and
    `neg140-v1.json` (sha256 `69d80b84…`).
  - Each manifest holds per clip: private path, sha256, sample count, the exact piece sample
    boundaries, history id, Q0 text per piece, term labels and the per-clip phrase list.
  - Provenance:
    - There are no human transcripts for personal clips.
    - Term labels are Opus inferences from context, not reviewed by the user.
    - FLEURS and switch clips carry human references.
    - Spoken forms are in `spoken-forms-v1.json`, from the user's history.
  - Audio locations:
    - Personal: `V:\voclaude-model-eval\recordings\audio-*.f32`. `V:` refuses program execution; reading works.
    - FLEURS and switch: `D:\voclaude-eval\sets\*.f32`, raw copies of the `.npy` data.
- [x] **Candidate build:** `D:\voclaude-eval\handoff-q1-phrases-b3\` contains:
  - `app\` (exe and DLLs, plus the Q0 exe)
  - `native\` (patch and `build.cmd`)
  - `scripts\`
  - `outputs\` (all per-piece outputs above)
  - `results\` (private edit list)
  - `SHA256SUMS` (sha256 `bd114524…`; it covers everything except the top-level .md copies, which mirror the repo)
- [x] **Results:** see `QUALITY-Q1-PHRASES.md`.
  - GPU: 21/25 labelled misspellings fixed on names55, 0 false edits.
  - On neg140: 0 edits on 120 human-reference clips, 1 correct edit on 20 personal negatives.
  - The Q0 identity checks above.
  - App segmentation (whole clips through the worker `file` command, `segment_all`):
    - Q1 without phrases equals Q0 on 195/195 clips.
    - With phrases: 19 edits on names55 (17/20 labelled misspellings fixed) and 1 edit on neg140. All edits are correct by context.
- [x] **Worker profiling (`VOCLAUDE_WORKER_PROFILE=1`)**, as the laptop asked. No wire or API change. Default unchanged.
  - Flag unset: same outputs, no profile record and no DEBUG line on stderr. Verified on both sets.
  - Flag set (any value but empty or `0`):
    - The `--worker` subscriber logs at DEBUG to stderr.
    - The worker writes one `voclaude-profile {json}` line per request at INFO.
    - Outputs stay byte-identical. Verified on both sets.
  - Records (fixed keys, no transcript text):

    | `kind` | Keys |
    |---|---|
    | `worker_preload` | `ms`, `ok`, `gpu` |
    | `worker_segment` | `index`, `start`, `end`, `samples`, `read_ms`, `engine_ms`, `loaded_model`, `ok` |
    | `parakeet_segment` | `samples`, `gpu`, `first_ms`, `mel_ms`, `encode_ms`, `decode_ms`, `russian_recheck`, `russian_ms`, `russian_edits`, `phrases`, `phrase_ms`, `phrase_biased`, `phrase_edits` |
    | `worker_file` | `engine_ms`, `ok` |

  - Meaning of the times:
    - `first_ms` is the wall time of the first decode. `mel/encode/decode_ms` are its native phases.
    - `russian_ms` and `phrase_ms` are the two re-checks. `phrase_ms` includes the biased second decode.
    - `engine_ms` is the whole segment inside the worker, excluding the audio read (`read_ms`).
    - `parakeet_segment` comes before its `worker_segment` and carries no index. Pair them in order.
      No `parakeet_segment` record is written for a segment under 0.1 s, an input the model rejects as
      too short, or a decoder stop. Its `worker_segment` record is still written.
  - Text check, with the flag on, over 296 segments: no Cyrillic in stderr. No transcript 3-word run in
    stderr except two native-log phrases ("in its own memory", "from 30 to 60"). All DEBUG lines are
    native scheduler and decoder statistics, or the app's count-only lines.
  - Desktop GPU sums, names55 with the list (one run):
    - Desktop background load moves the names55 wall time between 49 and 80 s, flag on or off.
      The flag showed no measurable cost.

      | Pass | Time |
      |---|---|
      | First decode | 45.9 s (encode 13.2 s, decode 26.1 s) |
      | Russian re-check | 2.3 s |
      | Phrase re-check | 23.0 s (76 biased decodes) |

    - Treat these as structure, not performance.
  - `VOCLAUDE_PHASE_TIMING` from the previous candidate is removed. It is replaced by this flag.

## Reproduction commands (desktop, PowerShell)

```powershell
# Q1 build (separate target dir; never target\release; TRANSCRIBE_DIR is required)
$env:TRANSCRIBE_DIR = "C:\Users\matvei\Documents\code\Voclaude\vendor\transcribe-prebuilt"
$env:CARGO_TARGET_DIR = "C:\Users\matvei\Documents\code\Voclaude\target\quality"
cargo build --release            # 0.6.0: the default build has no CUDA
cargo test  --release            # 50 passed
.\build-transcribe-patched.ps1   # patched transcribe.dll -> vendor\transcribe-patched\
.\package.ps1 -PatchedTranscribe -Installer   # zip + installer with the patched DLL

# B3 native: apply the patch to a copy of transcribe-cpp-sys-0.4.1 and build only transcribe.dll
git -C <copy> apply docs\bench\intel-optimization\native\parakeet-phrase-bias-b3.patch
docs\bench\intel-optimization\native\build-transcribe-dll.cmd <copy> <build>

# Evaluation: run from D:\voclaude-eval with C:\...\uv\python\cpython-3.12-windows-x86_64-none\python.exe -I
python -I worker_eval.py <app-dir> pieces.json out.jsonl --recheck 1 --phrases loo   # or --phrases none, add --cpu
# Profile records: the worker inherits the variable; WORKER_STDERR keeps its stderr in a file
$env:VOCLAUDE_WORKER_PROFILE = "1"; $env:WORKER_STDERR = "profile.log"
python -I worker_eval.py <app-dir> pieces.json out.jsonl --recheck 1 --phrases loo
Select-String "voclaude-profile " profile.log                                        # one JSON object per line
python -I make_results.py
```

## Repository note

`QUALITY-PERF-HANDOFF.md` and `PERF-STATUS.md` are laptop-authored. They stay untracked here, so the laptop
commits them with its tools. The desktop commits only `OPUS-QUALITY-ACK.md`, `QUALITY-Q1-PHRASES.md` and `native/`.

## Requests to laptop Codex

1. **Q1 identity on Arc:** run B3 with an empty list against the official DLL, and Q1 without phrases against Q0,
   on the laptop. Use `worker_eval.py --phrases none`, or your own harness on the manifest pieces.
2. **Q1 cost on Arc/CPU:** run with `--phrases loo`. The second decode runs only on pieces with a candidate
   word: 53 % of names55 audio, 12 % of neg140 audio on the desktop.
3. **API decision:** a decode-only second pass, which reuses the first pass's encoder output, needs a native
   API addition. Decide if the measured cost justifies it before anyone builds it.

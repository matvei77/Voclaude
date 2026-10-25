# Model survey, 2026-10-23: small multilingual ASR + LLM cleanup

Question: which small, fast, multilingual speech model should Voclaude use on the RTX 3060
and on an Intel Core Ultra laptop CPU, how does Wispr Flow produce clean text, and does a
small local LLM for cleanup work for Russian/English dictation?

Everything below was measured on this desktop (RTX 3060 12 GB, i5-13600K, DDR4-3600) on
2026-10-23. The GPU was the desktop's display GPU, shared with ~40 other processes (7.6 GB VRAM in
use before the runs). The laptop was not available, so laptop numbers are estimates. Harness, sets and
raw outputs: `D:\voclaude-model-eval\` (local only: it holds personal transcripts).

## Test sets

| Set | What | Reference |
|---|---|---|
| FLEURS ru/en/de/fr | 40 validation utterances per language (6–8 min each) | human transcripts, WER after lowercasing and stripping punctuation |
| switch | 20 synthetic clips, EN–RU–EN or RU–EN–RU FLEURS sentences joined by 0.4 s pauses (7 min) | joined transcripts; also "how many clips lose one language" (<50 % of its words survive) |
| personal | the 12 `bench_data` dictations (29 min, accented English, one RU passage) | no labels: agreement with the stored Qwen3-ASR-1.7B F16 transcript, fillers removed. This favours Qwen. |
| cleanup | 32 real history entries (14 EN, 14 RU, 4 mixed) + 9 traps (questions, "ignore previous instructions", self-corrections, in-sentence RU/EN) | automatic guards + reading every diff |

## ASR results (transcribe.cpp 0.4.1, Q8_0 GGUF, single stream)

| | Parakeet Ultra | Parakeet v3 | Nemotron 3.5 streaming | Qwen3-ASR-0.6B | Qwen3-ASR-1.7B |
|---|---|---|---|---|---|
| Params / Q8 file | 0.6B / 0.74 GB | 0.6B / 0.74 GB | 0.6B / 0.75 GB | 0.9B / 0.85 GB | 2.0B / 2.19 GB |
| Licence | CC-BY-4.0 | CC-BY-4.0 | OpenMDW-1.1 | Apache-2.0 | Apache-2.0 |
| WER ru / en / de / fr | **5.1** / 4.7 / **4.7** / 5.3 | 5.5 / 5.1 / 5.3 / 5.3 | 10.2 / 7.9 / 9.1 / 10.9 | 12.1 / 3.6 / 6.8 / 8.1 | 5.4 / **3.3** / 5.3 / **4.4** |
| RTX 3060 (Vulkan), FLEURS clips | 99× | 103× | 47× | 39× | 25× |
| RTX 3060 (Vulkan), dictation in ≤30 s chunks | **182×** | 179× | 95× | 82× | 50× |
| i5-13600K CPU, 6 threads (default thread count) | **20×** (18×) | (17×) | (15×) | 7× (6×) | (3×) |
| Your Russian passage (real mixed recording) | **correct, Cyrillic** | garbled | garbled | gibberish in Latin letters | **translated into English** ("…in Polish") |
| Differences vs Qwen-1.7B F16 on your English | 7.1 % | 6.9 % | 11.6 % | 6.5 % | 3.8 % |
| Silent file | empty | empty | empty | empty | empty |

For comparison, the current app (candle, Qwen3-ASR-1.7B Q8_0) runs at 32× on CUDA and 2.3× on this CPU.

The Russian passage is an English sentence, then two Russian clauses, then English again. Ultra
kept all three parts in their own language and script. Qwen3-ASR-0.6B in the app had heard the same
Russian verbs as Ultra, so Ultra is probably more faithful here than the 1.7B, which heard different ones.

What Ultra's remaining differences on your English are: kept fillers ("uh", "um" — it is verbatim),
numbers as digits ("500 euros", "27%"), and some real slips on fast accented speech ("mail fee",
"weapon", "anallocate"). Qwen-1.7B is the better English recogniser (FLEURS-en 3.3 vs 4.7); Ultra
is equal or better on Russian and German, 3.6× faster on the GPU and 7× faster on the CPU.

### Code-switching

| Chunking | Clips losing a language (of 20) | Switch WER |
|---|---|---|
| Whole clip | 15–19 for **every** model | 48–50 % |
| Split at pauses (VAD, each piece decides its own language) | 3 for every model | 16–23 % |
| Same, spans padded by 250 ms (Ultra and Qwen-1.7B only) | 3 | 13.1 % (Ultra), 13.7 % (Qwen-1.7B) |
| Split where language ID sees a change, model still decides the language | 3 for every model | 17–18 % |

- Whole-clip failures are mostly **silent drops** of the other-language sentence, and for Qwen
  also **translation**. No model, Qwen-1.7B included, switches reliably inside one chunk.
- Language-boundary chunking fixes Qwen-1.7B's translation of your real passage (18 Cyrillic words
  instead of 0) and costs Ultra nothing (177× instead of 182×).
- Pad pause-based spans by ~250 ms: Silero's tight boundaries clip first sounds (WER 16.3 → 13.1 for Ultra).
  The personal-set chunks above were unpadded too.
- Do not let language ID *force* the language: on your accented English it labels many segments
  Russian, German or French.
- The synthetic clips switch speaker as well as language, and there is only one real mixed recording.
  This is the weakest part of the evidence (see next steps).

### Runtime findings

- On the 3060, transcribe.cpp's **Vulkan backend is 2.5× faster than its CUDA backend** for Parakeet
  (encoder 46 ms vs 242 ms for 12 s of audio). The CPU+Vulkan native build is 18 MB with no CUDA DLLs.
- The CPU backend must be the per-ISA build (`ggml-cpu-alderlake` etc.): the generic `ggml-cpu.dll`
  shipped in the CUDA wheel ran Parakeet at 3× instead of 18×.
- 6 threads beat transcribe.cpp's default thread count on this hybrid CPU (20× vs 18×).
- The Parakeet transducer decoder costs ~0.85 ms per step on the GPU (~70 ms per clip); it is a
  latency floor, not a throughput problem.

## How Wispr Flow does it (public sources, Oct 2026)

- Two cloud stages: speech recognition, then a fine-tuned **Llama** that formats the text, served
  on Baseten with TensorRT-LLM. Budget: ASR <200 ms + LLM <200 ms + network <200 ms, text within
  700 ms p99 ([engineering post](https://wisprflow.ai/post/technical-challenges),
  [Baseten case study](https://www.baseten.co/resources/customers/wispr-flow/)).
- ASR since 17 Sep 2026 is their own model **Canto** (SFT + RL with a 1−WER reward, personal
  dictionary at runtime; English evaluations only) ([post](https://wisprflow.ai/canto)).
- Context sent to the LLM: app name/type, text before/after the cursor, dictionary, your name,
  optionally page text and a screenshot ([API schema](https://api-docs.wisprflow.ai/request_schema)).
  Casing/spacing matching to the surrounding text runs on the device.
- "Transcription always occurs on the cloud" ([data controls](https://wisprflow.ai/data-controls)).
- "Flow detects **one language per dictation**, not per word"
  ([docs](https://docs.wisprflow.ai/articles/3191899797-use-flow-with-multiple-languages)).
  For RU/EN mixing, a local pipeline with language-boundary chunking can do better than Wispr.

Local equivalents: Superwhisper's S1-mini and Spokenly's `cleanup-1` are both Qwen3-0.6B
fine-tunes; Handy (Rust/Tauri, Windows) uses Parakeet v3 via transcribe-rs.

## LLM cleanup results (llama.cpp b11836, greedy, same prompt for all chat models)

| Model | Size | 3060: median / ≥80 words | CPU: median / ≥80 words | Guard flags (of 41) | What went wrong |
|---|---|---|---|---|---|
| **Spokenly cleanup-1** | 0.6B, Q8_0 | **115 ms / 0.5 s** | **0.53 s / 2.6 s** (2.0 s with n-gram drafts) | **0** | one Russian word changed ("картинки" → "Картины") |
| Qwen3.5-0.8B | 0.8B, Q8_0 | 159 ms / 0.6 s | 0.77 s / 3.7 s | 1 | translated a Russian clause inside an English sentence into English |
| Qwen3-1.7B | 1.7B, Q4_K_M | 182 ms / 0.8 s | 1.0 s / 4.7 s (1.9 s with n-gram drafts) | 1 | obeyed "ignore previous instructions … banana" |
| Qwen3.5-2B | 2B, Q4_K_M | 205 ms / 0.8 s | 1.2 s / 5.5 s | 2 | banana + translated the mixed sentence |
| Qwen3-4B-Instruct-2507 | 4B, Q4_K_M | 328 ms / 1.5 s (0.7 s with n-gram drafts) | – | 2 | banana + translated the mixed sentence |
| Qwen3.5-4B | 4B, Q4_K_M | 407 ms / 1.7 s | – | 3 | **translated two English dictations into Russian**; lectured instead of cleaning |
| GigaChat 3.1 Lightning | 10B MoE (1.8B active), Q4_K_M | 231 ms / 1.0 s | 2.1 s / 9.6 s | 3 | banana; translated a mixed message to English |

Caveat: every chat model ran under one system prompt, written in English with Russian filler words in
its rule list. Mixed-language prompts are known to induce translation, so the translation failures of the
chat models are partly prompt-confounded; a tuned prompt per model would do better. cleanup-1 needs no
prompt at all, which is part of why it wins. The n-gram draft rows are Qwen3-1.7B on CPU and Qwen3-4B on GPU.

Guards (from the open-source dictation apps): output words must mostly come from the input
(≥75 %), length 0.5–1.3× the input, Cyrillic share within ±15 points. They caught every
failure listed above; on a flag the app should paste the raw transcript.

cleanup-1 on real dictations removed fillers and false starts, applied self-corrections in both
languages (only the corrected phrase is kept), removed stutters ("не не" → "не"), kept
in-sentence English inside Russian, and left questions and injected commands alone. Its prompt is
plain `text: …\nout:` (optional `app`, `tone`, `before`, `after` keys). Spokenly states Russian
quality is lower than English.

## Recommendation

1. **ASR: Parakeet Ultra (Q8_0) through transcribe.cpp** (ggml, Rust bindings), Vulkan on the
   3060, per-ISA CPU build on the laptop. 182× realtime on the GPU, 20× on this CPU; expected
   ~10–15× on a Core Ultra laptop (estimate). Same model and file on both targets.
2. **Chunking:** keep the current streaming segmenter (cut at pauses, ≤30 s), and also cut where
   language ID sees RU↔EN change; never force the language from language ID.
3. **Cleanup: cleanup-1 (0.6B)** per segment while you are still speaking, so only the last segment
   is cleaned after you stop (~0.1 s on the 3060, ~0.5 s on CPU). Guards + raw fallback, with a toggle.
4. Keep Qwen3-ASR-1.7B as an optional "accurate English" tier on the 3060, with language-boundary
   chunking; it is the better English recogniser but 2B and 3× realtime on CPU.

## Decision (2026-10-23, after the follow-up tests)

**Parakeet Ultra Q6_K (611 MB) through transcribe.cpp is the local model**, on the GPU (Vulkan)
and on the CPU, served by Voclaude's own OpenAI-compatible API (`127.0.0.1:7770`). Qwen3-ASR stays
available as `engine = "qwen"` but is no longer the default. Everything below was run through the
Voclaude build itself (`voclaude.exe --worker`). Installed as the live app on 2026-10-24.

Error rate on FLEURS (read speech, 40 clips per language, human references; lower is better):

| | ru | en | de | fr | nl | avg |
|---|---|---|---|---|---|---|
| Parakeet Ultra Q8_0 | 5.1 | 4.7 | 4.7 | 5.3 | 7.5 | 5.5 |
| **Parakeet Ultra Q6_K** | 5.1 | 4.7 | 4.7 | 5.0 | 7.3 | 5.4 |
| Parakeet Ultra Q4_K_M | 4.8 | 5.1 | 4.5 | 5.3 | 7.5 | 5.5 |
| Qwen3-ASR-1.7B (the app before) | 5.0 | 3.6 | 5.3 | 4.4 | 7.4 | 5.1 |
| Qwen3-ASR-0.6B | 11.4 | 4.1 | 7.3 | 8.1 | 14.2 | 9.0 |
| *cloud: Voxtral Small Transcribe 3* | 0.7 | 1.7 | 1.0 | 2.5 | 1.2 | 1.4 |
| *cloud: Gemini 4 Argon* | 2.1 | 3.3 | 3.2 | 3.0 | 3.5 | 3.0 |
| *cloud: ElevenLabs Scribe v2* | 2.0 | 2.4 | 2.3 | 4.4 | 4.8 | 3.2 |

Speed on a real 58-minute RU/EN meeting (the app's own 170 live segments):

| | GPU (RTX 3060) | CPU (i5-13600K) | Wait per 20 s segment, CPU |
|---|---|---|---|
| Parakeet Q8 / Q6 / Q4 | 24–25 s for 58 min (~145×) | 3.1 / 2.2 / 2.2 min (18.5× / 26× / 26×) | 1.1 / 0.8 / 0.8 s |
| Qwen3-ASR-1.7B | 120 s (29×) | 23 min (2.5×); the app silently swaps to 0.6B on CPU | 8.0 s |
| Qwen3-ASR-0.6B | 85 s (41×) | 12 min (4.8×) | 4.3 s |

Blind reading of the same meeting (no human reference exists): for the 143 segments where the
transcripts differ, readers saw the versions shuffled and unlabelled and picked the one most likely
to be what was said. Two readers on the same packet agreed on 22 of 24 verdicts.

| | Parakeet Q8 better | Qwen-1.7B better | tie / can't tell |
|---|---|---|---|
| All segments | 45 | 24 | 54 |
| Russian | 36 | 13 | 35 |
| English | 7 | 9 | 18 |

Errors counted by the readers: Qwen 109 (97 in Russian; plus 3 Chinese hallucinations), Parakeet
Q8 48, Q6 52, Q4 50. Q4 vs Q8 came out 18 : 17, so the quantisations are indistinguishable by
reading. On a 34 s EN–RU–EN clip of the user's own dictation, all three Parakeet files matched the
cloud consensus word for word, while Qwen-1.7B translated the Russian sentence into English.

Why Q6_K: output within 0.8 % of Q8's words, FLEURS equal to Q8, the same CPU speed as Q4 (40 %
faster than Q8), 129 MB smaller than Q8. Q4_K_M is an equal choice by every measure here if
size matters more.

The honest gap: on read speech Parakeet makes about four times the errors of Voxtral Small
Transcribe 3 and twice those of Gemini 4 Argon, in every language. FLEURS is public, so the cloud
models may have trained on these clips; their margin on unseen dictation is not measured. No cloud model is a safe
reference for RU/EN switching inside one segment (Voxtral and Scribe drop a language on 8–10 of 20
mixed clips; GPT Transcribe drops 2 but has no zero-data-retention endpoint). Raw outputs:
`D:\voclaude-model-eval\out\app`, `cloud`, `judge`, `side-by-side` (local only).

## Not measured / next steps

- **Segment length:** the meeting runs above used the app's old 20–45 s segments. The app now cuts
  Parakeet segments at 6–45 s (`segment_min_seconds` default 6) so each language part gets its own
  segment; speed and accuracy at that length are not measured yet.

- **Real code-switching set:** dictate 15–20 natural RU/EN mixes, correct the transcripts by hand,
  rerun. This decides whether Ultra's single real success generalises.
- **The laptop:** CPU, Arc iGPU (Vulkan) and NPU numbers. Its LPDDR5x has about twice the memory
  bandwidth of this desktop's DDR4-3600, so LLM decoding may be faster there; prefill and the ASR
  encoder will be slower.
- A cleanup LoRA trained on your own dictations (logged, labelled by a frontier model), as
  [hourliert](https://github.com/hourliert/VoiceInk-Qwen3.5-2B-FT/blob/master/docs/BLOG_POST.md) did.

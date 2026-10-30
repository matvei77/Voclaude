# Qwen3-ASR Integration Plan (Python -> Rust)

Updated: 2026-02-12

## Current State

Working app: `next/voclaude_qwen_runtime/` — full voice dictation with Qwen3-ASR, persistent server, polished UI. Phases 1-4 are complete.

### What is in Rust (voclaude_qwen_runtime)

- Audio capture (cpal, 16kHz resampling)
- WAV encoding (hound)
- System tray with icons (tray-icon)
- Global hotkeys F4 / Ctrl+Shift+H (global-hotkey)
- UI: HUD overlay, history window, dark theme, level bars (eframe/egui)
- Window management (Hidden/Hud/History mode switching)
- Clipboard paste (arboard)
- History storage (JSON on disk)
- Session recovery (crash-safe)
- Config loading (TOML)
- Process orchestration: spawning, managing, and communicating with Python server
- JSON protocol over stdin/stdout pipes

### What is still in Python (transcribe.py)

- **Model loading**: `Qwen3ASRModel.from_pretrained()` via `qwen_asr` library
- **Inference**: `model.transcribe(audio=...)` — the actual neural network forward pass
- **Audio normalization**: `normalize_audio_input()` for reading audio files into tensors
- **Chunking**: splitting long audio into overlapping segments for the model
- **CUDA/torch runtime**: PyTorch manages GPU memory, CUDA kernels, tensor ops

Dependencies that Python brings:
- `torch` (~4.3 GB install)
- `qwen_asr` (HuggingFace model wrapper)
- `transformers` (tokenizer, model architecture)
- Python 3.10+ runtime (~100 MB)
- CUDA toolkit (via torch)

**Total Python dependency footprint: ~5.2 GB**

### Current resource usage

| Component | Size |
|-----------|------|
| Rust binary | ~7 MB |
| Python venv | ~5,256 MB |
| Model weights (bf16 safetensors) | ~4,485 MB |
| VRAM at runtime | ~7,343 MB of 12,288 MB |

## Completed Phases

### Phase 1: Controlled Python Validation ✓

- Isolated Python runner (`tools/qwen3_asr_smoke/transcribe.py`)
- JSON output mode, chunking, multi-language support
- Batch comparison tooling

### Phase 2: Rust Backend Abstraction ✓

- Backend trait in `voclaude_frontier` (InferenceBackend with health_check/transcribe_path)
- Skipped Whisper dual-backend — went directly to Qwen-only app

### Phase 3: Qwen Python Bridge in App ✓

- Full end-to-end: F4 → record → transcribe → clipboard
- JSON protocol between Rust and Python
- Error handling, timeouts, progress reporting

### Phase 4: Production Hardening ✓

- Persistent Python server mode (model loads once, stays in VRAM)
- Auto-restart on server crash
- Graceful shutdown with timeout
- stderr drain thread (prevents deadlocks)
- No console window flash on Windows (CREATE_NO_WINDOW)
- Taskbar hiding, always-on-top HUD
- Session recovery for interrupted transcriptions

## Phase 5: Eliminate Python Dependency

Goal: single Rust binary + model weights, no Python/torch install required.

### Qwen3-ASR Architecture (from source code analysis)

The model (`Qwen3-ASR-1.7B`) has **3 components** with **no cross-attention**:

#### 1. Audio Encoder (AuT) — ~300M params

- **Input**: 128-bin log-mel spectrogram (WhisperFeatureExtractor, FFT=400, hop=160)
- **Conv stem** (8x temporal downsampling → 12.5 Hz tokens):
  - `conv2d1`: 1→480 channels, kernel=3, stride=2, GELU
  - `conv2d2`: 480→480, kernel=3, stride=2, GELU
  - `conv2d3`: 480→480, kernel=3, stride=2, GELU
  - `conv_out`: Linear projection → d_model=1024
- **Positional embedding**: Sinusoidal (not learned)
- **Encoder stack**: 24 transformer layers
  - d_model=1024, 16 attention heads, FFN=4096
  - **GELU** activation, **LayerNorm with bias** on all projections
- **Projector**: `proj1` (Linear+bias) → GELU → `proj2` (Linear+bias), output_dim=2048

#### 2. Bridge — masked_scatter (No Cross-Attention!)

Audio features directly **replace** `<|audio_pad|>` placeholder tokens in the text embedding
sequence via `masked_scatter`. There is no cross-attention adapter. This is architecturally
simpler than Whisper.

#### 3. Qwen3 LLM Decoder — ~1.4B params

- 28 layers, hidden=2048, 16 attention heads, 8 KV heads (GQA 2:1)
- SwiGLU MLP (gate + up + down), **SiLU** activation
- **RMSNorm without bias** (different from encoder)
- MRoPE with interleaved sections `[24, 20, 20]` (temporal/height/width)
- Autoregressive generation, vocab=151,936 (Qwen2Tokenizer BPE)

#### Weight layout

- 2 safetensors shards: `model-00001-of-00002.safetensors` (4.0 GB) + `model-00002-of-00002.safetensors` (457 MB)
- 708 weight tensors, all bf16
- Prefixed: `thinker.audio_tower.*` (encoder) and `thinker.model.*` (decoder)

### Research Results (completed 2026-02-12)

#### Option A: ONNX Runtime (`ort` crate) — NOT VIABLE TODAY

Research findings:
- **No ONNX export exists** for Qwen3-ASR (official or community)
- HF Optimum does **not** support `qwen3_asr` model type
- Community ONNX exports exist for text-only Qwen3, but **none for ASR variant**
- Flash attention with dynamic windows cannot be exported to standard ONNX ops
- Would need 3 separate ONNX graphs (encoder, projector, decoder-with-past)
- Without KV-cache in decoder ONNX, autoregressive decoding is ~4x slower
- The `ort` crate itself is production-ready (v2.0.0-rc.11, CUDA 12, bf16, `load-dynamic`)

Verdict: Rust-side tooling is ready, but the ONNX export doesn't exist and would take
multi-week custom engineering with high risk of quality/perf regression.

#### Option B: Candle (Pure Rust ML) — RECOMMENDED PATH

Research findings:
- **Qwen3 LLM decoder already exists** in `candle-transformers` (`qwen3.rs`)
- **Whisper mel spectrogram already exists** in Candle's `audio.rs` (same 128-bin params)
- **Safetensors loading is native** via `VarBuilder::from_mmaped_safetensors()`
- **CUDA support is full** via `features = ["cuda"]` (uses cuBLAS)
- **Tokenizer**: `tokenizers` crate loads Qwen2Tokenizer BPE natively
- **Precedent**: `qwen3-tts-rs` project already built a Qwen3 audio model in pure Candle/Rust

What needs to be implemented from scratch:

| Component | Lines (est.) | Difficulty |
|-----------|-------------|------------|
| Conv2D downsampling stem | ~80 | Easy |
| Sinusoidal position embedding | ~30 | Easy |
| Audio encoder layer (attention + FFN) | ~200 | Medium |
| Audio encoder stack (24 layers) | ~150 | Medium |
| Projector (proj1 + proj2) | ~30 | Easy |
| masked_scatter injection | ~50 | Easy |
| MRoPE interleaved sections | ~150 | Medium-Hard |
| Token expansion logic | ~80 | Easy |
| Autoregressive decode loop | ~100 | Medium |
| Weight name mapping | ~100 | Easy-Medium |
| **Total new code** | **~1000-1500** | |

Estimated effort: **2-4 weeks** for a working prototype.

Deployment would look like:
```
voclaude.exe                              (~15-25 MB, pure Rust)
models/Qwen3-ASR-1.7B/
  config.json
  model-00001-of-00002.safetensors       (4.0 GB)
  model-00002-of-00002.safetensors       (457 MB)
  tokenizer.json
  preprocessor_config.json
```

Net savings vs current: eliminates ~5.2 GB of Python/torch dependencies.

#### Option C: llama.cpp / GGUF — NOT VIABLE TODAY

Research findings:
- **No GGUF conversion exists** for Qwen3-ASR
- llama.cpp audio support is "highly experimental" (only Ultravox + Qwen2.5-Omni)
- `convert_hf_to_gguf.py` does not support `Qwen3ASRModel`
- Feature request exists (Issue #17634) for Qwen-Omni, but not Qwen3-ASR specifically
- Timeline: 3-6+ months before this might be usable, if ever

#### Option D: PyInstaller Bundle — FALLBACK

- Bundle Python + torch + model into standalone .exe
- ~4+ GB bundled, 10-30s cold start
- Not truly Rust-native but removes user-facing Python dependency
- Use only if Candle path fails

### Community Watch

| Project | Status | Relevance |
|---------|--------|-----------|
| [qwen3-asr-swift](https://github.com/ivan-digital/qwen3-asr-swift) | Working (MLX, Apple Silicon only) | Proves architecture is portable to non-Python runtimes |
| [qwen3-tts-rs](https://github.com/TrevorS/qwen3-tts-rs) | Working (Candle, CUDA) | Proves Qwen3 audio models work in pure Rust/Candle |
| [llama.cpp #17634](https://github.com/ggml-org/llama.cpp/issues/17634) | In progress | Qwen-Omni ASR endpoint being developed |
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | No Qwen3-ASR | May add support in future |

### Recommended path: Candle (Option B)

Why Candle wins:
1. Most code already exists (Qwen3 decoder, mel extraction, safetensors, CUDA, tokenizer)
2. Only ~1000-1500 lines of new Rust needed (AuT encoder is a standard transformer)
3. No cross-attention — the masked_scatter bridge is dead simple
4. Proven pattern — qwen3-tts-rs already did this for the TTS variant
5. Single binary — no Python, no ONNX export step, loads safetensors directly

Risks and mitigations:

| Risk | Impact | Mitigation |
|------|--------|------------|
| MRoPE interleaving bug | Garbage output | Compare against Python reference with same audio |
| Conv output length mismatch | Wrong alignment | Unit test `_get_feat_extract_output_lengths` |
| bf16 precision differences | Quality drop | Compare WER on test set vs Python |
| Candle CUDA perf vs PyTorch | Potentially slower | Candle uses cuBLAS; within ~1.5x |
| Flash attention not in Candle | Slower long audio | Use standard attention (correct, slower) |

Implementation sequence:
1. Prototype AuT encoder in Candle (conv stem + 1 layer, verify shapes)
2. Port MRoPE interleaved and validate against Python reference
3. Wire up existing Qwen3 decoder from candle-transformers
4. Integration test: same audio through Python and Rust, compare output
5. Keep Python server as fallback until Rust path matches quality

### Exit criteria

- Single binary deployment (no Python, no pip, no venv)
- Same or better transcription quality as Python path
- GPU inference with comparable latency
- Russian + English support maintained

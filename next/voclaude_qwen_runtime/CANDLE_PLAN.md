# Candle Native Inference — Execution Plan

Replace the Python server backend (`qwen.rs`) with a pure-Rust Candle inference engine
that loads Qwen3-ASR-1.7B safetensors directly and runs on CUDA via candle-core.

Branch: `feature/candle-native-inference`

## Architecture Overview

Current: `QwenEngine` → spawns Python subprocess → JSON over stdin/stdout → Python loads torch + model
Target:  `QwenEngine` → loads safetensors directly via candle → CUDA inference in-process

The public API of `QwenEngine` stays the same (`prepare`, `transcribe`, `transcribe_with_progress`,
`unload`, `model_label`, etc). Only the internals change. `app.rs` and the rest of the codebase
should need zero or minimal changes.

## File Plan

### New files to create

```
src/inference/
  candle_backend.rs      — Core Candle model: audio encoder, projector, decoder, generate loop
  candle_audio.rs        — Mel spectrogram computation (port from Candle's whisper/audio.rs)
  candle_tokenizer.rs    — Tokenizer loading and chat template formatting
```

### Files to modify

```
Cargo.toml              — Add candle-core, candle-nn, candle-transformers, safetensors,
                           tokenizers, hf-hub, half. Add cuda/cpu features.
src/inference/mod.rs    — Add mod declarations for new files
src/inference/qwen.rs   — Replace Python subprocess with Candle model calls
src/config.rs           — Add model_path config field (path to safetensors dir)
```

### Files that should NOT change

```
src/app.rs              — Calls QwenEngine through existing public API (no changes needed)
src/main.rs             — No changes needed
src/ui.rs               — No changes needed
src/audio/*             — No changes needed (still captures at 16kHz, gives f32 samples)
src/tray.rs             — No changes needed
```

---

## Step-by-Step Execution

### Step 1: Add Candle dependencies to Cargo.toml

Add these dependencies:

```toml
# ML inference (Candle)
candle-core = "0.9"
candle-nn = "0.9"
candle-transformers = "0.9"
safetensors = "0.7"
tokenizers = "0.22"
hf-hub = "0.4"
half = "2.4"
num-traits = "0.2"

[features]
default = ["cuda"]
cuda = ["candle-core/cuda", "candle-nn/cuda"]
cpu = []
```

Remove `tempfile` (no longer needed — we don't write WAV files to disk for Python).
Remove `hound` (no longer needed for the same reason).

Verify: `cargo check` succeeds (even if inference code is not yet wired up).

### Step 2: Create `candle_audio.rs` — Mel spectrogram

Port the mel spectrogram computation. This converts raw f32 PCM samples (16kHz mono) into
a 128-bin log-mel spectrogram tensor.

Reference implementation: `candle-transformers/src/models/whisper/audio.rs`

Key constants (must match WhisperFeatureExtractor config):
- `SAMPLE_RATE = 16000`
- `N_FFT = 400`
- `HOP_LENGTH = 160`
- `N_MELS = 128`
- `CHUNK_LENGTH = 30` (seconds)

Public API:

```rust
/// Compute 128-bin log-mel spectrogram from PCM samples.
/// Returns tensor of shape (1, n_mels, n_frames).
pub fn pcm_to_mel(samples: &[f32], device: &Device) -> Result<Tensor>

/// Load mel filterbank weights (128 filters for 201 FFT bins).
/// Can be computed or loaded from file.
pub fn mel_filters(device: &Device) -> Result<Tensor>
```

The mel computation is: STFT with Hann window → power spectrum → mel filterbank → log scale.
This is pure math, no model weights involved.

Verify: Write a unit test that computes mel for a short sine wave and checks the output shape
is `(1, 128, expected_frames)` where `expected_frames = (n_samples / HOP_LENGTH)`.

### Step 3: Create `candle_tokenizer.rs` — Tokenizer and prompt formatting

Load the Qwen2 BPE tokenizer and construct the chat template for ASR.

Public API:

```rust
pub struct Qwen3ASRTokenizer {
    tokenizer: tokenizers::Tokenizer,
}

impl Qwen3ASRTokenizer {
    /// Load from model directory (reads tokenizer.json).
    pub fn load(model_dir: &Path) -> Result<Self>

    /// Build the full input_ids for an ASR request.
    /// Returns (input_ids, audio_token_positions) where audio_token_positions
    /// are the indices that should be replaced with audio features.
    pub fn encode_asr_prompt(
        &self,
        n_audio_tokens: usize,
        language: Option<&str>,
    ) -> Result<(Vec<u32>, Vec<usize>)>

    /// Decode token IDs back to text.
    pub fn decode(&self, token_ids: &[u32]) -> Result<String>
}
```

The prompt template is:
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|audio_start|><|audio_pad|>...(N times)...<|audio_pad|><|audio_end|>
Transcribe the audio to text.<|im_end|>
<|im_start|>assistant
```

Special token IDs (from tokenizer_config.json):
- `<|audio_start|>` = 151669
- `<|audio_end|>` = 151670
- `<|audio_pad|>` = 151676 (the placeholder that gets replaced with audio features)
- `<|im_start|>` = 151644
- `<|im_end|>` = 151645
- `<|endoftext|>` = 151643

The number of `<|audio_pad|>` tokens = output of `_get_feat_extract_output_lengths(mel_frames)`.

Implement `get_feat_extract_output_lengths`:
```rust
fn get_feat_extract_output_lengths(input_lengths: usize) -> usize {
    let leave = input_lengths % 100;
    let feat = (leave - 1) / 2 + 1;
    ((feat - 1) / 2 + 1 - 1) / 2 + 1 + (input_lengths / 100) * 13
}
```

Verify: Unit test that `get_feat_extract_output_lengths` matches Python for several input values.
Unit test that `encode_asr_prompt` produces correct token sequence.

### Step 4: Create `candle_backend.rs` — The model

This is the biggest file. It implements the three model components.

#### 4a: Audio Encoder (AuT)

```rust
pub struct AudioEncoder {
    conv2d1: Conv2d,         // 1 → 480, kernel=3, stride=2
    conv2d2: Conv2d,         // 480 → 480, kernel=3, stride=2
    conv2d3: Conv2d,         // 480 → 480, kernel=3, stride=2
    conv_out: Linear,        // flattened → d_model (1024), no bias
    positional_embedding: Tensor,  // sinusoidal, (max_source_positions, d_model)
    layers: Vec<AudioEncoderLayer>,
    ln_post: LayerNorm,      // with bias
    proj1: Linear,           // d_model → d_model, with bias
    proj2: Linear,           // d_model → output_dim (2048), with bias
    n_window: usize,
    n_window_infer: usize,
    conv_chunksize: usize,
}
```

Each `AudioEncoderLayer`:
```rust
struct AudioEncoderLayer {
    self_attn_layer_norm: LayerNorm,  // with bias
    self_attn: AudioAttention,        // q/k/v/out_proj all with bias
    final_layer_norm: LayerNorm,      // with bias
    fc1: Linear,                      // with bias
    fc2: Linear,                      // with bias
    activation: candle_nn::Activation, // GELU
}
```

`AudioAttention`:
```rust
struct AudioAttention {
    q_proj: Linear,   // with bias
    k_proj: Linear,   // with bias
    v_proj: Linear,   // with bias
    out_proj: Linear,  // with bias
    num_heads: usize,
    head_dim: usize,
}
```

Forward pass:
1. Chunk input mel into windows of `n_window * 2` frames
2. Pad chunks to equal length, run through conv2d{1,2,3} with GELU
3. Flatten conv output, project through `conv_out`
4. Add sinusoidal positional embeddings
5. Pack into sequences with cu_seqlens for windowed attention
6. Run through 24 encoder layers
7. `ln_post` → `proj1` → GELU → `proj2`

Weight prefix: `thinker.audio_tower.*`

#### 4b: Text Decoder (reuse candle-transformers Qwen3)

The Qwen3 decoder already exists in `candle-transformers`. However, we may need a custom
version because:
- The existing `qwen3.rs` uses standard RoPE, but Qwen3-ASR uses MRoPE with interleaved
  sections [24, 20, 20]
- We need to feed mixed audio+text embeddings (not just token IDs)

Options:
1. **Fork the candle Qwen3 model and modify it** — add MRoPE support and `inputs_embeds` path
2. **Write a thin wrapper** that calls into candle-transformers internals

Recommended: Write our own decoder based on candle-transformers' qwen3.rs, modified for:
- MRoPE with `mrope_section = [24, 20, 20]` and interleaved frequency layout
- Accept `inputs_embeds: Tensor` instead of `input_ids: Tensor` for the first forward pass
- QK-norm (RMSNorm on q and k projections per-head) — verify candle's qwen3.rs already has this
- GQA with 16 query heads, 8 KV heads

```rust
pub struct TextDecoder {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    rotary_emb: MRoPEEmbedding,  // custom MRoPE
    device: Device,
    dtype: DType,
}
```

Weight prefix: `thinker.model.*` and `thinker.lm_head.*`

#### 4c: MRoPE (Multi-dimensional Rotary Position Embedding)

```rust
pub struct MRoPEEmbedding {
    inv_freq: Tensor,
    mrope_section: [usize; 3],  // [24, 20, 20]
    attention_scaling: f64,
}

impl MRoPEEmbedding {
    /// Compute cos/sin for position_ids of shape (3, batch, seq_len).
    /// Returns (cos, sin) each of shape (batch, seq_len, head_dim).
    pub fn forward(&self, x: &Tensor, position_ids: &Tensor) -> Result<(Tensor, Tensor)>
}
```

The interleaving converts chunked `[TTT...HHH...WWW]` to `[THWTHW...]`:
```rust
fn apply_interleaved_mrope(freqs: &Tensor, mrope_section: &[usize; 3]) -> Result<Tensor> {
    // freqs shape: (3, bs, seq_len, head_dim/2)
    // For each of H and W dimensions, interleave into T's frequency slots
    // Result shape: (bs, seq_len, head_dim/2)
}
```

#### 4d: Top-level model and generate loop

```rust
pub struct Qwen3ASRModel {
    audio_encoder: AudioEncoder,
    text_decoder: TextDecoder,
    config: ModelConfig,
    device: Device,
    dtype: DType,
}

impl Qwen3ASRModel {
    /// Load from a directory containing safetensors + config.json.
    pub fn load(model_dir: &Path, device: &Device, dtype: DType) -> Result<Self>

    /// Transcribe audio samples to text.
    pub fn transcribe(
        &mut self,
        samples: &[f32],
        language: Option<&str>,
        tokenizer: &Qwen3ASRTokenizer,
    ) -> Result<String>
}
```

The `transcribe` method:
1. Compute mel spectrogram from samples
2. Run audio encoder → get audio features (seq_len, 2048)
3. Compute `n_audio_tokens = get_feat_extract_output_lengths(n_mel_frames)`
4. Build prompt via tokenizer: get `input_ids` and `audio_token_positions`
5. Embed text tokens via `embed_tokens`
6. Replace audio positions with audio features (`masked_scatter` equivalent)
7. Compute position_ids (handle rope_deltas for audio vs text)
8. Run decoder forward pass with mixed embeddings
9. Autoregressive generation loop:
   - Get logits from decoder output
   - Argmax to get next token
   - Stop on EOS token (151645 or 151643)
   - Feed next token ID back into decoder (with KV cache)
10. Decode token IDs to text via tokenizer
11. Strip language tags / ASR markers from output

Verify: Load model, transcribe a test WAV, compare output to Python.

### Step 5: Wire up `qwen.rs` to use Candle backend

Replace the Python server internals with Candle model calls.

```rust
pub struct QwenEngine {
    model: Option<Qwen3ASRModel>,
    tokenizer: Option<Qwen3ASRTokenizer>,
    device: Device,
    dtype: DType,
    // ... keep existing config fields for model_path, language, etc.
}
```

- `prepare()` → loads model and tokenizer from disk (replaces `ensure_server_ready`)
- `transcribe(samples)` → calls `model.transcribe(samples, language, tokenizer)`
- `unload()` → drops the model (frees VRAM)

Remove: `ServerProcess`, `ServerReady`, `ServerResponse`, `ServerResult`,
`resolve_python`, `resolve_script`, `send_transcribe_request`, `write_wav_mono_16k`,
`load_f32_file` (or keep load_f32_file if --test mode still uses it).

Remove `tempfile` and `hound` from dependencies.

### Step 6: Update config.rs

Add a new config field:

```rust
pub qwen_model_path: Option<String>,  // Path to local safetensors directory
```

If not set, fall back to `~/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/...`
or use `hf-hub` crate to download.

Remove or deprecate: `qwen_python_path`, `qwen_script_path` (no longer needed).

### Step 7: Update Cargo.toml features and build

Final Cargo.toml feature setup:

```toml
[features]
default = ["cuda"]
cuda = ["candle-core/cuda", "candle-nn/cuda"]
cpu = []
```

Build commands:
- `cargo build --release` — CUDA build (default)
- `cargo build --release --no-default-features --features cpu` — CPU-only build

### Step 8: Test and validate

1. `cargo check` — zero errors
2. `cargo build --release` — succeeds
3. `cargo run --release -- --test "path/to/test.wav"` — transcribes correctly
4. Compare output against Python server for the same audio files
5. Measure: inference latency, VRAM usage, binary size

---

## Dependency Summary

| Crate | Purpose | Size Impact |
|-------|---------|-------------|
| `candle-core` | Tensor ops, CUDA backend | ~2 MB (+ CUDA libs) |
| `candle-nn` | Neural network layers (Linear, Conv2d, etc.) | Minimal |
| `candle-transformers` | Existing Qwen3 model code (reference/reuse) | Minimal |
| `safetensors` | Load model weights | Minimal |
| `tokenizers` | BPE tokenizer | ~1 MB |
| `hf-hub` | Download model from HuggingFace | Minimal |
| `half` | bf16/f16 support | Minimal |
| `num-traits` | Float trait for mel computation | Minimal |

Removed:
| `tempfile` | No longer writing temp WAV files |
| `hound` | No longer encoding WAV for Python |

## Weight Loading Map

Safetensors weight prefixes → Rust struct fields:

```
thinker.audio_tower.conv2d{1,2,3}.{weight,bias}     → AudioEncoder.conv2d{1,2,3}
thinker.audio_tower.conv_out.weight                  → AudioEncoder.conv_out
thinker.audio_tower.positional_embedding.*           → AudioEncoder.positional_embedding
thinker.audio_tower.layers.{0-23}.self_attn.*        → AudioEncoderLayer.self_attn
thinker.audio_tower.layers.{0-23}.self_attn_layer_norm.* → AudioEncoderLayer.self_attn_layer_norm
thinker.audio_tower.layers.{0-23}.fc1.*              → AudioEncoderLayer.fc1
thinker.audio_tower.layers.{0-23}.fc2.*              → AudioEncoderLayer.fc2
thinker.audio_tower.layers.{0-23}.final_layer_norm.* → AudioEncoderLayer.final_layer_norm
thinker.audio_tower.ln_post.*                        → AudioEncoder.ln_post
thinker.audio_tower.proj1.*                          → AudioEncoder.proj1
thinker.audio_tower.proj2.*                          → AudioEncoder.proj2

thinker.model.embed_tokens.weight                    → TextDecoder.embed_tokens
thinker.model.layers.{0-27}.self_attn.q_proj.weight  → DecoderLayer.self_attn.q_proj
thinker.model.layers.{0-27}.self_attn.k_proj.weight  → DecoderLayer.self_attn.k_proj
thinker.model.layers.{0-27}.self_attn.v_proj.weight  → DecoderLayer.self_attn.v_proj
thinker.model.layers.{0-27}.self_attn.o_proj.weight  → DecoderLayer.self_attn.o_proj
thinker.model.layers.{0-27}.self_attn.q_norm.weight  → DecoderLayer.self_attn.q_norm
thinker.model.layers.{0-27}.self_attn.k_norm.weight  → DecoderLayer.self_attn.k_norm
thinker.model.layers.{0-27}.mlp.gate_proj.weight     → DecoderLayer.mlp.gate_proj
thinker.model.layers.{0-27}.mlp.up_proj.weight       → DecoderLayer.mlp.up_proj
thinker.model.layers.{0-27}.mlp.down_proj.weight     → DecoderLayer.mlp.down_proj
thinker.model.layers.{0-27}.input_layernorm.weight   → DecoderLayer.input_layernorm
thinker.model.layers.{0-27}.post_attention_layernorm.weight → DecoderLayer.post_attention_layernorm
thinker.model.norm.weight                            → TextDecoder.norm
thinker.lm_head.weight                               → TextDecoder.lm_head
```

## Config Values (from config.json of 1.7B)

### Audio Encoder
- `d_model = 1024`
- `encoder_layers = 24`
- `encoder_attention_heads = 16`
- `encoder_ffn_dim = 4096`
- `num_mel_bins = 128`
- `max_source_positions = 1500`
- `n_window = 50`
- `n_window_infer = 800`
- `conv_chunksize = 500`
- `downsample_hidden_size = 480`
- `output_dim = 2048`
- `activation_function = "gelu"`
- `dropout = 0.0`

### Text Decoder
- `hidden_size = 2048`
- `num_hidden_layers = 28`
- `num_attention_heads = 16`
- `num_key_value_heads = 8`
- `head_dim = 128`
- `intermediate_size = 6144`
- `vocab_size = 151936`
- `max_position_embeddings = 65536`
- `rope_theta = 1000000.0`
- `rope_scaling.mrope_section = [24, 20, 20]`
- `hidden_act = "silu"`
- `rms_norm_eps = 1e-6`
- `attention_bias = false`

### Special Token IDs
- `audio_token_id = 151676` (audio_pad — the one replaced by audio features)
- `audio_start_token_id = 151669`
- `eos_token_id = [151645, 151643]`
- `pad_token_id = 151643`

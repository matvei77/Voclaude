//! Candle-based Qwen3-ASR model: audio encoder, text decoder, MRoPE, generate loop.
//!
//! Loads safetensors weights from a local directory and runs inference entirely
//! in Rust via `candle-core`.

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Conv2d, Conv2dConfig, Embedding, LayerNorm, LayerNormConfig, Linear, Module, VarBuilder};
use serde::Deserialize;
use std::path::Path;

use crate::inference::candle_audio;
use crate::inference::candle_tokenizer::{
    get_feat_extract_output_lengths, EOS_TOKEN_IDS, Qwen3ASRTokenizer,
};

// ═══════════════════════════════════════════════════════════════════════════
// Config structs (parsed from config.json in model directory)
// ═══════════════════════════════════════════════════════════════════════════

/// Top-level config.json structure for Qwen3-ASR.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    #[serde(default)]
    pub thinker_config: ThinkerOuterConfig,
}

/// The `thinker_config` block contains both `audio_config` and `text_config`.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct ThinkerOuterConfig {
    #[serde(default)]
    pub audio_config: AudioConfig,
    #[serde(default)]
    pub text_config: ThinkerConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioConfig {
    #[serde(default = "default_d_model")]
    pub d_model: usize,
    #[serde(default = "default_encoder_layers")]
    pub encoder_layers: usize,
    #[serde(default = "default_encoder_attention_heads")]
    pub encoder_attention_heads: usize,
    #[serde(default = "default_encoder_ffn_dim")]
    pub encoder_ffn_dim: usize,
    #[serde(default = "default_num_mel_bins")]
    pub num_mel_bins: usize,
    #[serde(default = "default_max_source_positions")]
    pub max_source_positions: usize,
    #[serde(default = "default_n_window")]
    pub n_window: usize,
    #[serde(default = "default_n_window_infer")]
    pub n_window_infer: usize,
    #[serde(default = "default_conv_chunksize")]
    pub conv_chunksize: usize,
    #[serde(default = "default_downsample_hidden_size")]
    pub downsample_hidden_size: usize,
    #[serde(default = "default_output_dim")]
    pub output_dim: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ThinkerConfig {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub rope_scaling: Option<RopeScalingConfig>,
}

impl ThinkerConfig {
    pub fn mrope_section(&self) -> Vec<usize> {
        self.rope_scaling
            .as_ref()
            .map(|rs| rs.mrope_section.clone())
            .unwrap_or_else(default_mrope_section)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScalingConfig {
    #[serde(default = "default_mrope_section")]
    pub mrope_section: Vec<usize>,
}

// Defaults matching Qwen3-ASR-1.7B config.json
fn default_d_model() -> usize { 1024 }
fn default_encoder_layers() -> usize { 24 }
fn default_encoder_attention_heads() -> usize { 16 }
fn default_encoder_ffn_dim() -> usize { 4096 }
fn default_num_mel_bins() -> usize { 128 }
fn default_max_source_positions() -> usize { 1500 }
fn default_n_window() -> usize { 50 }
fn default_n_window_infer() -> usize { 800 }
fn default_conv_chunksize() -> usize { 500 }
fn default_downsample_hidden_size() -> usize { 480 }
fn default_output_dim() -> usize { 2048 }
fn default_hidden_size() -> usize { 2048 }
fn default_num_hidden_layers() -> usize { 28 }
fn default_num_attention_heads() -> usize { 16 }
fn default_num_key_value_heads() -> usize { 8 }
fn default_head_dim() -> usize { 128 }
fn default_intermediate_size() -> usize { 6144 }
fn default_vocab_size() -> usize { 151936 }
fn default_rope_theta() -> f64 { 1_000_000.0 }
fn default_rms_norm_eps() -> f64 { 1e-6 }
fn default_mrope_section() -> Vec<usize> { vec![24, 20, 20] }

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            d_model: default_d_model(),
            encoder_layers: default_encoder_layers(),
            encoder_attention_heads: default_encoder_attention_heads(),
            encoder_ffn_dim: default_encoder_ffn_dim(),
            num_mel_bins: default_num_mel_bins(),
            max_source_positions: default_max_source_positions(),
            n_window: default_n_window(),
            n_window_infer: default_n_window_infer(),
            conv_chunksize: default_conv_chunksize(),
            downsample_hidden_size: default_downsample_hidden_size(),
            output_dim: default_output_dim(),
        }
    }
}

impl Default for ThinkerConfig {
    fn default() -> Self {
        Self {
            hidden_size: default_hidden_size(),
            num_hidden_layers: default_num_hidden_layers(),
            num_attention_heads: default_num_attention_heads(),
            num_key_value_heads: default_num_key_value_heads(),
            head_dim: default_head_dim(),
            intermediate_size: default_intermediate_size(),
            vocab_size: default_vocab_size(),
            rope_theta: default_rope_theta(),
            rms_norm_eps: default_rms_norm_eps(),
            rope_scaling: Some(RopeScalingConfig {
                mrope_section: default_mrope_section(),
            }),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Audio Encoder
// ═══════════════════════════════════════════════════════════════════════════

struct AudioAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl AudioAttention {
    fn load(vb: VarBuilder, d_model: usize, num_heads: usize) -> Result<Self> {
        let head_dim = d_model / num_heads;
        let q_proj = candle_nn::linear(d_model, d_model, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(d_model, d_model, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(d_model, d_model, vb.pp("v_proj"))?;
        let out_proj = candle_nn::linear(d_model, d_model, vb.pp("out_proj"))?;
        Ok(Self { q_proj, k_proj, v_proj, out_proj, num_heads, head_dim })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((b, seq, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape((b, seq, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape((b, seq, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;

        let scale = (self.head_dim as f64).sqrt();
        let in_dtype = q.dtype();
        let attn = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)? / scale)?;
        let attn = candle_nn::ops::softmax(&attn, D::Minus1)?.to_dtype(in_dtype)?;
        let out = attn.matmul(&v)?;

        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, seq, self.num_heads * self.head_dim))?;
        self.out_proj.forward(&out)
    }
}

struct AudioEncoderLayer {
    self_attn_layer_norm: LayerNorm,
    self_attn: AudioAttention,
    final_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl AudioEncoderLayer {
    fn load(vb: VarBuilder, d_model: usize, num_heads: usize, ffn_dim: usize) -> Result<Self> {
        let ln_cfg = LayerNormConfig { eps: 1e-5, ..Default::default() };
        let self_attn_layer_norm = candle_nn::layer_norm(d_model, ln_cfg, vb.pp("self_attn_layer_norm"))?;
        let self_attn = AudioAttention::load(vb.pp("self_attn"), d_model, num_heads)?;
        let final_layer_norm = candle_nn::layer_norm(d_model, ln_cfg, vb.pp("final_layer_norm"))?;
        let fc1 = candle_nn::linear(d_model, ffn_dim, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(ffn_dim, d_model, vb.pp("fc2"))?;
        Ok(Self { self_attn_layer_norm, self_attn, final_layer_norm, fc1, fc2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.self_attn_layer_norm.forward(x)?;
        let x = self.self_attn.forward(&x)?;
        let x = (x + residual)?;

        let residual = x.clone();
        let x = self.final_layer_norm.forward(&x)?;
        let x = self.fc1.forward(&x)?;
        let x = x.gelu()?.to_dtype(x.dtype())?;
        let x = self.fc2.forward(&x)?;
        x + residual
    }
}

pub struct AudioEncoder {
    conv2d1: Conv2d,
    conv2d2: Conv2d,
    conv2d3: Conv2d,
    conv_out: Linear,
    positional_embedding: Tensor,
    layers: Vec<AudioEncoderLayer>,
    ln_post: LayerNorm,
    proj1: Linear,
    proj2: Linear,
    n_window: usize,
    n_window_infer: usize,
    conv_chunksize: usize,
}

impl AudioEncoder {
    fn load(vb: VarBuilder, cfg: &AudioConfig) -> Result<Self> {
        let ds = cfg.downsample_hidden_size;
        let conv_cfg_s2 = Conv2dConfig { stride: 2, padding: 1, ..Default::default() };

        let conv2d1 = candle_nn::conv2d(1, ds, 3, conv_cfg_s2, vb.pp("conv2d1"))?;
        let conv2d2 = candle_nn::conv2d(ds, ds, 3, conv_cfg_s2, vb.pp("conv2d2"))?;
        let conv2d3 = candle_nn::conv2d(ds, ds, 3, conv_cfg_s2, vb.pp("conv2d3"))?;

        // After 3x stride-2 convs on mel_bins: each conv halves spatially (with padding=1)
        // I-10: Compute actual output dim from CNN spatial reduction instead of hardcoding /8
        let freq_after_conv = |dim: usize| -> usize {
            // Each Conv2d with stride=2, padding=1, kernel=3: out = floor((in + 2*1 - 3)/2 + 1)
            let d1 = (dim + 2 - 3) / 2 + 1;
            let d2 = (d1 + 2 - 3) / 2 + 1;
            let d3 = (d2 + 2 - 3) / 2 + 1;
            d3
        };
        let freq_out = freq_after_conv(cfg.num_mel_bins);
        let conv_out_dim = freq_out * ds;
        let conv_out = candle_nn::linear_no_bias(conv_out_dim, cfg.d_model, vb.pp("conv_out"))?;

        // Sinusoidal positional embeddings (computed, not learned)
        let positional_embedding = sinusoidal_position_embedding(
            cfg.max_source_positions,
            cfg.d_model,
            vb.device(),
        )?;

        let mut layers = Vec::with_capacity(cfg.encoder_layers);
        for i in 0..cfg.encoder_layers {
            layers.push(AudioEncoderLayer::load(
                vb.pp(format!("layers.{}", i)),
                cfg.d_model,
                cfg.encoder_attention_heads,
                cfg.encoder_ffn_dim,
            )?);
        }

        let ln_cfg = LayerNormConfig { eps: 1e-5, ..Default::default() };
        let ln_post = candle_nn::layer_norm(cfg.d_model, ln_cfg, vb.pp("ln_post"))?;
        let proj1 = candle_nn::linear(cfg.d_model, cfg.d_model, vb.pp("proj1"))?;
        let proj2 = candle_nn::linear(cfg.d_model, cfg.output_dim, vb.pp("proj2"))?;

        Ok(Self {
            conv2d1,
            conv2d2,
            conv2d3,
            conv_out,
            positional_embedding,
            layers,
            ln_post,
            proj1,
            proj2,
            n_window: cfg.n_window,
            n_window_infer: cfg.n_window_infer,
            conv_chunksize: cfg.conv_chunksize,
        })
    }

    /// Run the audio encoder on a mel spectrogram of shape `(1, n_mels, n_frames)`.
    /// Returns audio features of shape `(1, n_audio_tokens, output_dim)`.
    ///
    /// Implements windowed processing matching the Python reference:
    /// 1. Split mel into chunks of `n_window * 2` frames
    /// 2. Process chunks through conv layers as a batch
    /// 3. Extract valid tokens per chunk, add positional embeddings
    /// 4. Run transformer layers with windowed attention
    /// 5. Project to output dimension
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let (_b, n_mels, n_frames) = mel.dims3()?;
        let device = mel.device();
        let dtype = mel.dtype();
        let chunk_size = self.n_window * 2; // 100 frames per chunk

        // Step 1: Split mel into chunks of chunk_size frames
        let n_chunks = (n_frames + chunk_size - 1) / chunk_size;
        let mut chunk_lengths = vec![chunk_size; n_chunks];
        let last_len = n_frames % chunk_size;
        if last_len > 0 {
            chunk_lengths[n_chunks - 1] = last_len;
        }

        // Step 2: Pad each chunk to chunk_size and stack into a batch
        let mel_2d = mel.squeeze(0)?; // (n_mels, n_frames)
        let mut chunk_tensors = Vec::with_capacity(n_chunks);
        let mut offset = 0;
        for &clen in &chunk_lengths {
            let chunk = mel_2d.narrow(1, offset, clen)?; // (n_mels, clen)
            let padded = if clen < chunk_size {
                let pad = Tensor::zeros((n_mels, chunk_size - clen), dtype, device)?;
                Tensor::cat(&[&chunk, &pad], 1)?
            } else {
                chunk
            };
            chunk_tensors.push(padded.unsqueeze(0)?.unsqueeze(0)?); // (1, 1, n_mels, chunk_size)
            offset += clen;
        }
        let batch = Tensor::cat(&chunk_tensors, 0)?; // (n_chunks, 1, n_mels, chunk_size)

        // Step 3: Process through conv layers (in sub-batches of conv_chunksize)
        let mut conv_outputs = Vec::new();
        for start in (0..n_chunks).step_by(self.conv_chunksize) {
            let end = (start + self.conv_chunksize).min(n_chunks);
            let sub = batch.narrow(0, start, end - start)?;
            let x = self.conv2d1.forward(&sub)?.gelu()?.to_dtype(dtype)?;
            let x = self.conv2d2.forward(&x)?.gelu()?.to_dtype(dtype)?;
            let x = self.conv2d3.forward(&x)?.gelu()?.to_dtype(dtype)?;
            conv_outputs.push(x);
        }
        let x = Tensor::cat(&conv_outputs, 0)?; // (n_chunks, ds, freq_out, time_out)

        // Step 4: Reshape: (n_chunks, ds, freq, t) -> (n_chunks, t, ds*freq) -> conv_out
        let (b, c, freq, time) = x.dims4()?;
        let x = x.permute((0, 3, 1, 2))?.contiguous()?.reshape((b, time, c * freq))?;
        let x = self.conv_out.forward(&x)?; // (n_chunks, time, d_model)

        // Step 5: Add sinusoidal positional embeddings (per-chunk, truncated to time)
        let pos_emb = self.positional_embedding.i(..time)?.to_dtype(dtype)?;
        let x = x.broadcast_add(&pos_emb)?;

        // Step 6: Extract valid tokens per chunk using after-CNN lengths
        let after_cnn_lengths: Vec<usize> = chunk_lengths
            .iter()
            .map(|&cl| get_feat_extract_output_lengths(cl))
            .collect();

        let mut all_tokens = Vec::new();
        for (chunk_idx, &valid_len) in after_cnn_lengths.iter().enumerate() {
            if valid_len > 0 {
                let chunk_tokens = x.i(chunk_idx)?.narrow(0, 0, valid_len)?;
                all_tokens.push(chunk_tokens);
            }
        }
        let hidden_states = Tensor::cat(&all_tokens, 0)?; // (total_tokens, d_model)
        let total_tokens = hidden_states.dims()[0];

        // Step 7: Compute attention window boundaries
        // Each window spans n_window_infer / chunk_size chunks = 8 chunks = 104 tokens
        let aftercnn_per_full_chunk = get_feat_extract_output_lengths(chunk_size); // 13
        let chunks_per_window = self.n_window_infer / chunk_size; // 800/100 = 8
        let window_size = aftercnn_per_full_chunk * chunks_per_window; // 13*8 = 104

        let mut window_boundaries = vec![0usize];
        let mut pos = 0;
        while pos < total_tokens {
            let this_window = window_size.min(total_tokens - pos);
            pos += this_window;
            window_boundaries.push(pos);
        }

        // Step 8: Run transformer layers with windowed attention
        // Process each window independently through all layers
        let mut window_outputs = Vec::new();
        for w in 0..window_boundaries.len() - 1 {
            let start = window_boundaries[w];
            let wlen = window_boundaries[w + 1] - start;
            let window = hidden_states.narrow(0, start, wlen)?.unsqueeze(0)?; // (1, wlen, d_model)

            let mut x = window;
            for layer in &self.layers {
                x = layer.forward(&x)?;
            }
            window_outputs.push(x.squeeze(0)?); // (wlen, d_model)
        }
        let hidden = Tensor::cat(&window_outputs, 0)?; // (total_tokens, d_model)

        // Step 9: Post-processing: ln_post -> proj1 -> gelu -> proj2
        let hidden = hidden.unsqueeze(0)?; // (1, total_tokens, d_model)
        let hidden = self.ln_post.forward(&hidden)?;
        let hidden = self.proj1.forward(&hidden)?.gelu()?.to_dtype(dtype)?;
        self.proj2.forward(&hidden) // (1, total_tokens, output_dim)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MRoPE (Multi-dimensional Rotary Position Embedding)
// ═══════════════════════════════════════════════════════════════════════════

pub struct MRoPEEmbedding {
    head_dim: usize,
    /// Precomputed cos/sin cache indexed by position.
    /// Shape: (max_cached_pos, head_dim).
    /// Computed at model load time using the interleaved MRoPE frequency
    /// assignment — since ASR uses identical positions across all 3 MRoPE
    /// dimensions, the dim_assignment doesn't affect the result and we can
    /// use a single unified table.
    cos_cache: Tensor,
    sin_cache: Tensor,
}

impl MRoPEEmbedding {
    /// I-3: Maximum positions to precompute (audio prefix + max generated tokens).
    /// 5-min audio ≈ 3950 tokens + 2048 generated = ~6000; use 8192 for headroom.
    const MAX_POSITIONS: usize = 8192;

    fn new(head_dim: usize, rope_theta: f64, _mrope_section: Vec<usize>, device: &Device, dtype: DType) -> Result<Self> {
        let half_dim = head_dim / 2;
        let inv_freq: Vec<f64> = (0..half_dim)
            .map(|i| 1.0 / rope_theta.powf(2.0 * i as f64 / head_dim as f64))
            .collect();

        // Precompute cos/sin for all positions 0..MAX_POSITIONS on CPU once,
        // then transfer to GPU. This replaces the per-step CPU computation.
        //
        // NOTE: In ASR, all 3 MRoPE dimensions (temporal, height, width) use
        // identical position sequences (0, 1, 2, ..., seq_len-1), so the
        // interleaved dimension assignment doesn't change the output. Each
        // frequency index i simply uses cos(pos * inv_freq[i]).
        let max_pos = Self::MAX_POSITIONS;
        let mut cos_data = vec![0.0f32; max_pos * head_dim];
        let mut sin_data = vec![0.0f32; max_pos * head_dim];

        for p in 0..max_pos {
            let base = p * head_dim;
            for i in 0..half_dim {
                let angle = p as f64 * inv_freq[i];
                let c = angle.cos() as f32;
                let s = angle.sin() as f32;
                // Duplicated across both halves of head_dim
                cos_data[base + i] = c;
                cos_data[base + half_dim + i] = c;
                sin_data[base + i] = s;
                sin_data[base + half_dim + i] = s;
            }
        }

        let cos_cache = Tensor::from_vec(cos_data, (max_pos, head_dim), device)?.to_dtype(dtype)?;
        let sin_cache = Tensor::from_vec(sin_data, (max_pos, head_dim), device)?.to_dtype(dtype)?;

        Ok(Self { head_dim, cos_cache, sin_cache })
    }

    /// Compute cos/sin for position_ids of shape `(3, batch, seq_len)`.
    /// Returns `(cos, sin)` each of shape `(batch, seq_len, head_dim)`.
    ///
    /// Uses GPU-resident precomputed tables — zero CPU round-trips during decode.
    /// Since ASR uses identical positions across all 3 MRoPE dimensions,
    /// we only need to gather from dimension 0.
    fn forward(&self, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_three, batch, seq_len) = position_ids.dims3()?;

        // All 3 dims have same positions in ASR — just use dim 0
        let positions = position_ids.i(0)?; // (batch, seq_len)
        let pos_flat = positions.reshape((batch * seq_len,))?;

        // GPU index_select: gather rows from precomputed table
        let cos = self.cos_cache.index_select(&pos_flat, 0)?; // (batch*seq_len, head_dim)
        let sin = self.sin_cache.index_select(&pos_flat, 0)?;

        let cos = cos.reshape((batch, seq_len, self.head_dim))?;
        let sin = sin.reshape((batch, seq_len, self.head_dim))?;

        Ok((cos, sin))
    }
}

/// Apply rotary embeddings to q or k tensor.
fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let last_dim = *x.dims().last().unwrap();
    let half = last_dim / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;

    // Rotate: [x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin]
    let cos = cos.unsqueeze(1)?; // (b, 1, seq, dim)
    let sin = sin.unsqueeze(1)?;
    let cos_half = cos.narrow(D::Minus1, 0, half)?;
    let sin_half = sin.narrow(D::Minus1, 0, half)?;

    let r1 = (x1.broadcast_mul(&cos_half)? - x2.broadcast_mul(&sin_half)?)?;
    let r2 = (x2.broadcast_mul(&cos_half)? + x1.broadcast_mul(&sin_half)?)?;
    Tensor::cat(&[r1, r2], D::Minus1)
}

// ═══════════════════════════════════════════════════════════════════════════
// RMS Norm
// ═══════════════════════════════════════════════════════════════════════════

struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn load(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let x = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        x.to_dtype(dtype)?.broadcast_mul(&self.weight)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Text Decoder (Qwen3 with MRoPE)
// ═══════════════════════════════════════════════════════════════════════════

struct DecoderAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    /// Pre-allocated KV cache: (k_cache, v_cache) each of shape
    /// (batch, num_kv_heads, cache_capacity, head_dim). Only the first
    /// `cache_len` entries along dim 2 are valid.
    kv_cache: Option<(Tensor, Tensor)>,
    cache_len: usize,
    cache_capacity: usize,
}

impl DecoderAttention {
    fn load(vb: VarBuilder, cfg: &ThinkerConfig) -> Result<Self> {
        let h = cfg.hidden_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        let q_proj = candle_nn::linear_no_bias(h, q_dim, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear_no_bias(h, kv_dim, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear_no_bias(h, kv_dim, vb.pp("v_proj"))?;
        let o_proj = candle_nn::linear_no_bias(h, h, vb.pp("o_proj"))?;
        let q_norm = RmsNorm::load(cfg.head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::load(cfg.head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        Ok(Self {
            q_proj, k_proj, v_proj, o_proj, q_norm, k_norm,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            kv_cache: None,
            cache_len: 0,
            cache_capacity: 0,
        })
    }

    /// Pre-allocate KV cache buffers for the given maximum sequence length.
    /// This avoids repeated Tensor::cat (which creates old+new+result = 3× peak VRAM).
    fn allocate_cache(&mut self, max_seq_len: usize, device: &Device, dtype: DType) -> Result<()> {
        let k = Tensor::zeros((1, self.num_kv_heads, max_seq_len, self.head_dim), dtype, device)?;
        let v = Tensor::zeros((1, self.num_kv_heads, max_seq_len, self.head_dim), dtype, device)?;
        self.kv_cache = Some((k, v));
        self.cache_len = 0;
        self.cache_capacity = max_seq_len;
        Ok(())
    }

    fn forward(&mut self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // BUG 12: capture dtype before reshape so we can restore it after F32 softmax.
        let q_dtype = q.dtype();
        let q = q.reshape((b, seq, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape((b, seq, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape((b, seq, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;

        // QK-norm
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // Apply rotary embeddings
        let q = apply_rotary_emb(&q, cos, sin)?;
        let k = apply_rotary_emb(&k, cos, sin)?;

        // KV cache: write new K/V into pre-allocated buffers via slice_set.
        // This avoids Tensor::cat which creates old+new+result = 3× peak VRAM.
        // Falls back to Tensor::cat if cache wasn't pre-allocated (shouldn't happen).
        if self.cache_capacity > 0 {
            // Pre-allocated path: write into existing buffer
            let (k_buf, v_buf) = self.kv_cache.as_ref().unwrap();
            k_buf.slice_set(&k, 2, self.cache_len)?;
            v_buf.slice_set(&v, 2, self.cache_len)?;
            self.cache_len += seq;
        } else {
            // Fallback: no pre-allocation, use cat (legacy path)
            let (k_full, v_full) = match &self.kv_cache {
                Some((prev_k, prev_v)) => {
                    let k_full = Tensor::cat(&[prev_k, &k], 2)?;
                    let v_full = Tensor::cat(&[prev_v, &v], 2)?;
                    (k_full, v_full)
                }
                None => (k, v),
            };
            self.cache_len = k_full.dims()[2];
            self.kv_cache = Some((k_full, v_full));
        }
        let (k_cache, v_cache) = self.kv_cache.as_ref().unwrap();
        // Narrow to only the valid portion of the pre-allocated buffer
        let k_full = k_cache.narrow(2, 0, self.cache_len)?;
        let v_full = v_cache.narrow(2, 0, self.cache_len)?;

        // GQA (Grouped Query Attention): Q has more heads than KV.
        // For seq=1 decode (hot path): reshape Q into KV groups instead of
        // expanding KV — avoids expensive expand+contiguous on the entire cache.
        // For seq>1 prefill (once): expand KV to match Q (simpler, correct).
        let n_rep = self.num_heads / self.num_kv_heads;
        let scale = (self.head_dim as f64).sqrt();
        let kv_len = k_full.dims()[2];

        let out = if seq == 1 && n_rep > 1 {
            // Decode hot-path: reshape Q into KV head groups.
            // q: (b, num_heads, 1, hd) → (b, num_kv_heads, n_rep, hd)
            let q = q.reshape((b, self.num_kv_heads, n_rep, self.head_dim))?;
            let k_t = k_full.transpose(D::Minus2, D::Minus1)?.contiguous()?; // (b, kv_h, hd, kv_len)
            drop(k_full);
            // (b, kv_h, n_rep, hd) @ (b, kv_h, hd, kv_len) → (b, kv_h, n_rep, kv_len)
            let attn = (q.matmul(&k_t)? / scale)?;
            drop(q);
            drop(k_t);
            // BUG 12: upcast to F32 for numerically stable softmax, then restore q_dtype.
            let attn = candle_nn::ops::softmax(&attn.to_dtype(DType::F32)?, D::Minus1)?.to_dtype(q_dtype)?;
            // (b, kv_h, n_rep, kv_len) @ (b, kv_h, kv_len, hd) → (b, kv_h, n_rep, hd)
            let out = attn.matmul(&v_full)?;
            drop(attn);
            drop(v_full);
            // (b, kv_h, n_rep, hd) → (b, num_heads, 1, hd) → (b, 1, hidden)
            out.reshape((b, self.num_heads, 1, self.head_dim))?
                .transpose(1, 2)?.contiguous()?
                .reshape((b, 1, self.num_heads * self.head_dim))?
        } else {
            // Prefill or n_rep==1: expand KV heads to match Q (happens once).
            let k_expanded = if n_rep > 1 {
                let (b, h, s, d) = k_full.dims4()?;
                k_full.unsqueeze(2)?.expand((b, h, n_rep, s, d))?.reshape((b, h * n_rep, s, d))?.contiguous()?
            } else {
                k_full.clone()
            };
            drop(k_full);
            let v_expanded = if n_rep > 1 {
                let (b, h, s, d) = v_full.dims4()?;
                v_full.unsqueeze(2)?.expand((b, h, n_rep, s, d))?.reshape((b, h * n_rep, s, d))?.contiguous()?
            } else {
                v_full.clone()
            };
            drop(v_full);

            let attn = (q.contiguous()?.matmul(&k_expanded.transpose(D::Minus2, D::Minus1)?.contiguous()?)? / scale)?;
            drop(q);
            drop(k_expanded);

            // Causal mask for prefill (seq > 1)
            // Build at target dtype on CPU, then transfer — halves GPU transfer for F16.
            let attn = if seq > 1 {
                let offset = kv_len - seq;
                // Fill the causal (zero) region row-by-row instead of per-element branch.
                let mut mask_data = vec![f32::NEG_INFINITY; seq * kv_len];
                for i in 0..seq {
                    let allowed = i + offset + 1; // number of allowed positions in this row
                    let row_start = i * kv_len;
                    for j in 0..allowed.min(kv_len) {
                        mask_data[row_start + j] = 0.0;
                    }
                }
                let mask = Tensor::from_vec(mask_data, (1, 1, seq, kv_len), &Device::Cpu)?
                    .to_dtype(attn.dtype())?
                    .to_device(x.device())?;
                attn.broadcast_add(&mask)?
            } else {
                attn
            };

            // BUG 12: upcast to F32 for numerically stable softmax, then restore q_dtype.
            let attn = candle_nn::ops::softmax(&attn.to_dtype(DType::F32)?, D::Minus1)?.to_dtype(q_dtype)?;
            let out = attn.matmul(&v_expanded)?;
            drop(attn);
            drop(v_expanded);
            out.transpose(1, 2)?.contiguous()?.reshape((b, seq, self.num_heads * self.head_dim))?
        };

        self.o_proj.forward(&out)
    }

    fn clear_cache(&mut self) {
        self.kv_cache = None;
        self.cache_len = 0;
        self.cache_capacity = 0;
    }
}

struct DecoderMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl DecoderMLP {
    fn load(vb: VarBuilder, hidden: usize, intermediate: usize) -> Result<Self> {
        let gate_proj = candle_nn::linear_no_bias(hidden, intermediate, vb.pp("gate_proj"))?;
        let up_proj = candle_nn::linear_no_bias(hidden, intermediate, vb.pp("up_proj"))?;
        let down_proj = candle_nn::linear_no_bias(intermediate, hidden, vb.pp("down_proj"))?;
        Ok(Self { gate_proj, up_proj, down_proj })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.silu()?.to_dtype(x.dtype())?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

struct DecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: DecoderAttention,
    post_attention_layernorm: RmsNorm,
    mlp: DecoderMLP,
}

impl DecoderLayer {
    fn load(vb: VarBuilder, cfg: &ThinkerConfig) -> Result<Self> {
        let input_layernorm = RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let self_attn = DecoderAttention::load(vb.pp("self_attn"), cfg)?;
        let post_attention_layernorm = RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?;
        let mlp = DecoderMLP::load(vb.pp("mlp"), cfg.hidden_size, cfg.intermediate_size)?;
        Ok(Self { input_layernorm, self_attn, post_attention_layernorm, mlp })
    }

    fn forward(&mut self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward(&x, cos, sin)?;
        let x = (x + residual)?;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        x + residual
    }

    fn allocate_cache(&mut self, max_seq_len: usize, device: &Device, dtype: DType) -> Result<()> {
        self.self_attn.allocate_cache(max_seq_len, device, dtype)
    }

    fn clear_cache(&mut self) {
        self.self_attn.clear_cache();
    }
}

pub struct TextDecoder {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    rotary_emb: MRoPEEmbedding,
    #[allow(dead_code)]
    device: Device,
    dtype: DType,
}

impl TextDecoder {
    fn load(vb_model: VarBuilder, vb_lm_head: VarBuilder, cfg: &ThinkerConfig, device: &Device, dtype: DType) -> Result<Self> {
        let embed_tokens = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_model.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::load(vb_model.pp(format!("layers.{}", i)), cfg)?);
        }

        let norm = RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb_model.pp("norm"))?;
        let lm_head = candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb_lm_head)?;
        let rotary_emb = MRoPEEmbedding::new(cfg.head_dim, cfg.rope_theta, cfg.mrope_section(), device, dtype)?;

        Ok(Self {
            embed_tokens, layers, norm, lm_head, rotary_emb,
            device: device.clone(), dtype,
        })
    }

    /// Forward pass with token IDs.
    fn forward_ids(&mut self, input_ids: &Tensor, position_ids: &Tensor) -> Result<Tensor> {
        let embeds = self.embed_tokens.forward(input_ids)?;
        self.forward_embeds(&embeds, position_ids)
    }

    /// Forward pass with pre-computed embeddings (for mixed audio+text input).
    fn forward_embeds(&mut self, embeds: &Tensor, position_ids: &Tensor) -> Result<Tensor> {
        let (cos, sin) = self.rotary_emb.forward(position_ids)?;
        let cos = cos.to_dtype(self.dtype)?;
        let sin = sin.to_dtype(self.dtype)?;

        let mut x = embeds.clone();
        for layer in &mut self.layers {
            x = layer.forward(&x, &cos, &sin)?;
        }

        let x = self.norm.forward(&x)?;
        self.lm_head.forward(&x)
    }

    /// Pre-allocate KV cache buffers across all layers for the given max sequence length.
    fn allocate_caches(&mut self, max_seq_len: usize, device: &Device, dtype: DType) -> Result<()> {
        for layer in &mut self.layers {
            layer.allocate_cache(max_seq_len, device, dtype)?;
        }
        Ok(())
    }

    fn clear_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_cache();
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Cache guard — ensures KV cache is cleared on all exit paths from transcribe
// ═══════════════════════════════════════════════════════════════════════════

struct CacheGuard<'a>(&'a mut TextDecoder);

impl Drop for CacheGuard<'_> {
    fn drop(&mut self) {
        self.0.clear_cache();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Top-level model
// ═══════════════════════════════════════════════════════════════════════════

pub struct Qwen3ASRModel {
    audio_encoder: AudioEncoder,
    text_decoder: TextDecoder,
    #[allow(dead_code)]
    config: ModelConfig,
    device: Device,
    dtype: DType,
    /// I-6: Configurable max_new_tokens (was hardcoded to 2048)
    pub max_new_tokens: usize,
}

impl Qwen3ASRModel {
    /// Load from a directory containing safetensors + config.json.
    pub fn load(model_dir: &Path, device: &Device, dtype: DType) -> Result<Self> {
        // Parse config
        let config_path = model_dir.join("config.json");
        let config_text = std::fs::read_to_string(&config_path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to read config.json: {}", e)))?;
        let config: ModelConfig = serde_json::from_str(&config_text)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to parse config.json: {}", e)))?;

        // Load safetensors
        let safetensor_files = find_safetensors(model_dir)?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&safetensor_files, dtype, device)?
        };

        let vb_audio = vb.pp("thinker.audio_tower");
        let vb_model = vb.pp("thinker.model");
        let vb_lm_head = vb.pp("thinker.lm_head");

        let audio_encoder = AudioEncoder::load(vb_audio, &config.thinker_config.audio_config)?;
        let text_decoder = TextDecoder::load(vb_model, vb_lm_head, &config.thinker_config.text_config, device, dtype)?;

        Ok(Self {
            audio_encoder,
            text_decoder,
            config,
            device: device.clone(),
            dtype,
            max_new_tokens: 2048, // default, overridden by QwenEngine
        })
    }

    /// Maximum audio duration in seconds before we reject (to avoid OOM).
    /// ~5 minutes is a safe limit for 8GB VRAM with F32 inference.
    const MAX_AUDIO_SECONDS: usize = 300;

    /// Transcribe audio samples to text.
    pub fn transcribe(
        &mut self,
        samples: &[f32],
        language: Option<&str>,
        tokenizer: &Qwen3ASRTokenizer,
    ) -> Result<String> {
        // Guard against excessively long audio that would OOM
        let duration_secs = samples.len() / candle_audio::SAMPLE_RATE;
        if duration_secs > Self::MAX_AUDIO_SECONDS {
            return Err(candle_core::Error::Msg(format!(
                "Audio too long ({} seconds, max {}). Please use shorter recordings.",
                duration_secs, Self::MAX_AUDIO_SECONDS
            )));
        }

        // 1-2. Compute mel and run audio encoder in a scoped block so all
        // intermediate tensors (mel, conv outputs, attention matrices) are
        // dropped before the decode phase begins — frees ~200-400MB VRAM.
        let (audio_features, n_audio_tokens) = {
            let mel = candle_audio::pcm_to_mel(samples, &self.device)?.to_dtype(self.dtype)?;
            let n_mel_frames = mel.dims()[2];
            let features = self.audio_encoder.forward(&mel)?;
            let n_tokens = get_feat_extract_output_lengths(n_mel_frames);
            let encoder_output_len = features.dims()[1];
            tracing::debug!(
                "Audio encoder: {} tokens (expected {}), mel_frames={}",
                encoder_output_len, n_tokens, n_mel_frames
            );

            let features = if encoder_output_len != n_tokens {
                tracing::warn!(
                    "Audio encoder output {} tokens, expected {}; trimming",
                    encoder_output_len, n_tokens
                );
                features.narrow(1, 0, n_tokens.min(encoder_output_len))?
            } else {
                features
            };
            // BUG 11: use the actual post-trim length, not the pre-trim estimate.
            let actual_len = features.dims()[1];
            (features, actual_len)
        }; // mel and encoder intermediates dropped here

        // Sync device to reclaim mel/encoder intermediate VRAM before decode phase.
        self.device.synchronize()?;

        // 3. Build prompt
        let (input_ids, audio_positions) = tokenizer.encode_asr_prompt(n_audio_tokens, language)?;

        // 4. Embed text tokens
        let input_ids_tensor = Tensor::from_vec(
            input_ids.iter().map(|&id| id as i64).collect::<Vec<_>>(),
            (1, input_ids.len()),
            &self.device,
        )?;
        let mut embeds = self.text_decoder.embed_tokens.forward(&input_ids_tensor)?;

        // 5. Replace audio positions with audio features (batched)
        // Audio positions are contiguous — splice in the audio features as a
        // single block instead of N sequential narrow+cat operations.
        // I-4: When encoder produces fewer tokens than predicted, skip ALL pad
        // positions (not just the replaced ones) to prevent <audio_pad> embeddings
        // from leaking into the model input.
        if !audio_positions.is_empty() {
            let n_replace = audio_positions.len().min(audio_features.dims()[1]);
            let first_pos = audio_positions[0];
            let audio_block = audio_features.narrow(1, 0, n_replace)?; // (1, n_replace, hidden)

            let mut parts: Vec<Tensor> = Vec::new();
            if first_pos > 0 {
                parts.push(embeds.narrow(1, 0, first_pos)?);
            }
            parts.push(audio_block);
            // Skip past ALL audio pad positions, not just the ones we replaced
            let after_start = first_pos + audio_positions.len();
            let total_len = embeds.dims()[1];
            if after_start < total_len {
                parts.push(embeds.narrow(1, after_start, total_len - after_start)?);
            }
            embeds = Tensor::cat(&parts, 1)?;
            drop(audio_features);
        }

        // 6. Build position IDs for MRoPE (3, batch, seq_len)
        let seq_len = embeds.dims()[1];
        let position_ids = self.build_position_ids(seq_len, &audio_positions)?;

        // 7. First forward pass with embeddings.
        // Clone device reference into a local so we can use it while the guard
        // holds &mut self.text_decoder (two disjoint borrows on different fields).
        let device = self.device.clone();

        // Sync device before KV cache allocation to ensure splicing intermediates are freed.
        device.synchronize()?;

        // The guard ensures clear_cache() is called on all exit paths (BUG 1).
        self.text_decoder.clear_cache();

        // Pre-allocate KV cache buffers for the entire sequence (prefill + max generated tokens).
        // This avoids Tensor::cat per decode step which creates 3× peak VRAM.
        let cache_capacity = seq_len + self.max_new_tokens;
        self.text_decoder.allocate_caches(cache_capacity, &device, self.dtype)?;

        let _cache_guard = CacheGuard(&mut self.text_decoder);
        let logits = _cache_guard.0.forward_embeds(&embeds, &position_ids)?;
        drop(embeds);

        // Extract last-token logits immediately and drop the full prefill tensor.
        // Shape: logits is (1, seq_len, vocab_size) → extract (vocab_size,)
        let mut last_logits = logits.i((0, logits.dims()[1] - 1))?;
        drop(logits);

        // 8. Autoregressive generation
        // I-6: Use configurable max_new_tokens instead of hardcoded value
        let max_new_tokens = self.max_new_tokens;
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut cur_pos = seq_len;

        for step in 0..max_new_tokens {
            // last_logits is (vocab_size,) — argmax gives the next token
            let next_token = last_logits
                .argmax(D::Minus1)?
                .to_scalar::<u32>()?;

            if EOS_TOKEN_IDS.contains(&next_token) {
                break;
            }

            generated_tokens.push(next_token);

            // Prepare next step; use local `device` to avoid reborrowing `self`
            // while _cache_guard holds &mut self.text_decoder.
            let next_ids = Tensor::from_vec(vec![next_token as i64], (1, 1), &device)?;
            let p = cur_pos as i64;
            let pos_ids = Tensor::from_vec(vec![p, p, p], (3, 1, 1), &device)?;
            let full_logits = _cache_guard.0.forward_ids(&next_ids, &pos_ids)?;
            // full_logits is (1, 1, vocab_size) → extract (vocab_size,)
            last_logits = full_logits.i((0, full_logits.dims()[1] - 1))?;
            drop(full_logits);
            cur_pos += 1;

            // Periodic sync to reclaim transient decode VRAM
            if step > 0 && step % 100 == 0 {
                device.synchronize()?;
            }
        }

        // 9. Decode tokens to text
        let text = tokenizer.decode(&generated_tokens)?;

        // 10. Clean up the output (strip language tags etc.)
        // _cache_guard drops here, calling clear_cache() automatically (BUG 1).
        Ok(clean_transcription(&text))
    }

    /// Build 3-dimensional position IDs for MRoPE.
    /// All 3 dims have the same sequential positions (0, 1, 2, ..., seq_len-1),
    /// matching the Python reference which uses `attention_mask.cumsum(-1) - 1`.
    ///
    /// I-8: `_audio_positions` is retained as a parameter for future models that
    /// need 2D audio position encoding. For Qwen3-ASR, the Python reference uses
    /// uniform sequential positions for all tokens (including audio), so the
    /// parameter is intentionally unused.
    fn build_position_ids(&self, seq_len: usize, _audio_positions: &[usize]) -> Result<Tensor> {
        let positions: Vec<i64> = (0..seq_len as i64).collect();
        let pos = Tensor::from_vec(positions, (1, seq_len), &self.device)?;
        Tensor::stack(&[&pos, &pos, &pos], 0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Compute sinusoidal positional embeddings (Qwen2AudioSinusoidalPositionalEmbedding).
/// Returns tensor of shape `(max_positions, d_model)`.
fn sinusoidal_position_embedding(
    max_positions: usize,
    d_model: usize,
    device: &Device,
) -> Result<Tensor> {
    let half_dim = d_model / 2;
    let log_10000 = 10000.0_f64.ln();
    // BUG 38: guard against division by zero when half_dim <= 1.
    let emb_scale: Vec<f64> = (0..half_dim)
        .map(|i| (-log_10000 * i as f64 / (half_dim.max(2) - 1) as f64).exp())
        .collect();

    let mut data = vec![0.0f32; max_positions * d_model];
    for pos in 0..max_positions {
        for i in 0..half_dim {
            let angle = pos as f64 * emb_scale[i];
            data[pos * d_model + i] = angle.sin() as f32;
            data[pos * d_model + half_dim + i] = angle.cos() as f32;
        }
    }

    Tensor::from_vec(data, (max_positions, d_model), device)
}

fn find_safetensors(model_dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(model_dir)
        .map_err(|e| candle_core::Error::Msg(format!("Cannot read model dir: {}", e)))?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    if files.is_empty() {
        return Err(candle_core::Error::Msg(
            "No safetensors files found in model directory".to_string(),
        ));
    }

    files.sort();
    Ok(files)
}

fn clean_transcription(text: &str) -> String {
    let text = text.trim();
    // Extract text after <asr_text> tag if present
    let result = if let Some(pos) = text.find("<asr_text>") {
        &text[pos + "<asr_text>".len()..]
    } else {
        text
    };
    // Remove closing tag and other artifacts
    let mut result = result.to_string();
    for tag in &["</asr_text>", "<|en|>", "<|zh|>", "<|ja|>", "<|ko|>", "<|yue|>", "<|nospeech|>"] {
        result = result.replace(tag, "");
    }
    result.trim().to_string()
}

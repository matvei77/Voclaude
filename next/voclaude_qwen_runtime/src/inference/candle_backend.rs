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

        // After 3x stride-2 convs on (mel_bins=128): 128 -> 64 -> 32 -> 16
        // Flattened: 16 * ds = 16 * 480 = 7680
        let conv_out_dim = (cfg.num_mel_bins / 8) * ds;
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
    inv_freq: Tensor,
    mrope_section: Vec<usize>,
}

impl MRoPEEmbedding {
    fn new(head_dim: usize, rope_theta: f64, mrope_section: Vec<usize>, device: &Device, dtype: DType) -> Result<Self> {
        let half_dim = head_dim / 2;
        let inv_freq: Vec<f64> = (0..half_dim)
            .map(|i| 1.0 / rope_theta.powf(2.0 * i as f64 / head_dim as f64))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (half_dim,), device)?.to_dtype(dtype)?;
        Ok(Self { inv_freq, mrope_section })
    }

    /// Compute cos/sin for position_ids of shape `(3, batch, seq_len)`.
    /// Returns `(cos, sin)` each of shape `(batch, seq_len, head_dim)`.
    ///
    /// Uses interleaved MRoPE: frequencies are assigned in a [THWTHW...TT] pattern
    /// instead of chunked [TTT...HHH...WWW], matching `mrope_interleaved: true`.
    fn forward(&self, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let half_dim = self.inv_freq.dims()[0];
        let inv_freq_vec = self.inv_freq.to_dtype(DType::F64)?.to_vec1::<f64>()?;

        // Extract position_ids for each dimension: (batch, seq_len) as Vec<Vec<i64>>
        let pos_vecs: Vec<Vec<Vec<i64>>> = (0..3)
            .map(|d| position_ids.i(d).and_then(|t| t.to_vec2::<i64>()))
            .collect::<Result<_>>()?;

        let batch = pos_vecs[0].len();
        let seq_len = pos_vecs[0][0].len();
        let head_dim = half_dim * 2;

        // Build interleaved dimension assignment for each frequency index.
        // mrope_section = [24, 20, 20]:
        //   H (dim 1): indices where i%3==1 and i < section[1]*3=60 → 20 indices
        //   W (dim 2): indices where i%3==2 and i < section[2]*3=60 → 20 indices
        //   T (dim 0): all remaining indices → 24 indices
        let sec_h = self.mrope_section[1]; // 20
        let sec_w = self.mrope_section[2]; // 20
        let len_h = sec_h * 3; // 60
        let len_w = sec_w * 3; // 60

        let mut dim_assignment = vec![0usize; half_dim]; // default: temporal
        for i in 0..half_dim {
            if i < len_h && i % 3 == 1 {
                dim_assignment[i] = 1; // height
            } else if i < len_w && i % 3 == 2 {
                dim_assignment[i] = 2; // width
            }
            // else: temporal (already 0)
        }

        // Compute interleaved cos/sin on CPU, then create tensors
        let total = batch * seq_len * head_dim;
        let mut cos_data = vec![0.0f64; total];
        let mut sin_data = vec![0.0f64; total];

        for b in 0..batch {
            for s in 0..seq_len {
                let base = (b * seq_len + s) * head_dim;
                for i in 0..half_dim {
                    let dim_idx = dim_assignment[i];
                    let pos = pos_vecs[dim_idx][b][s] as f64;
                    let angle = pos * inv_freq_vec[i];
                    let c = angle.cos();
                    let sn = angle.sin();
                    // First half and second half (duplicated)
                    cos_data[base + i] = c;
                    cos_data[base + half_dim + i] = c;
                    sin_data[base + i] = sn;
                    sin_data[base + half_dim + i] = sn;
                }
            }
        }

        let device = position_ids.device();
        let cos = Tensor::from_vec(cos_data, (batch, seq_len, head_dim), device)?;
        let sin = Tensor::from_vec(sin_data, (batch, seq_len, head_dim), device)?;

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
    kv_cache: Option<(Tensor, Tensor)>,
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
        })
    }

    fn forward(&mut self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((b, seq, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape((b, seq, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape((b, seq, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;

        // QK-norm
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // Apply rotary embeddings
        let q = apply_rotary_emb(&q, cos, sin)?;
        let k = apply_rotary_emb(&k, cos, sin)?;

        // KV cache
        let (k, v) = match &self.kv_cache {
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k, &k], 2)?;
                let v = Tensor::cat(&[prev_v, &v], 2)?;
                (k, v)
            }
            None => (k, v),
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        // GQA: repeat KV heads to match Q heads
        let n_rep = self.num_heads / self.num_kv_heads;
        let k = if n_rep > 1 {
            let (b, h, s, d) = k.dims4()?;
            k.unsqueeze(2)?.expand((b, h, n_rep, s, d))?.reshape((b, h * n_rep, s, d))?.contiguous()?
        } else {
            k
        };
        let v = if n_rep > 1 {
            let (b, h, s, d) = v.dims4()?;
            v.unsqueeze(2)?.expand((b, h, n_rep, s, d))?.reshape((b, h * n_rep, s, d))?.contiguous()?
        } else {
            v
        };

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let attn = (q.contiguous()?.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)? / scale)?;

        // Causal mask for decoder
        let kv_len = k.dims()[2];
        let attn = if seq > 1 {
            // Build lower-triangular causal mask manually
            let offset = (kv_len - seq) as i64;
            let mut mask_data = vec![f32::NEG_INFINITY; seq * kv_len];
            for i in 0..seq {
                for j in 0..kv_len {
                    if j as i64 <= i as i64 + offset {
                        mask_data[i * kv_len + j] = 0.0;
                    }
                }
            }
            let mask = Tensor::from_vec(mask_data, (1, 1, seq, kv_len), x.device())?
                .to_dtype(attn.dtype())?;
            let attn = attn.broadcast_add(&mask)?;
            candle_nn::ops::softmax(&attn, D::Minus1)?.to_dtype(q.dtype())?
        } else {
            candle_nn::ops::softmax(&attn, D::Minus1)?.to_dtype(q.dtype())?
        };
        let out = attn.matmul(&v)?;
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, seq, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&out)
    }

    fn clear_cache(&mut self) {
        self.kv_cache = None;
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

    fn clear_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_cache();
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Top-level model
// ═══════════════════════════════════════════════════════════════════════════

pub struct Qwen3ASRModel {
    audio_encoder: AudioEncoder,
    text_decoder: TextDecoder,
    config: ModelConfig,
    device: Device,
    dtype: DType,
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

        // 1. Compute mel spectrogram
        let mel = candle_audio::pcm_to_mel(samples, &self.device)?.to_dtype(self.dtype)?;
        let n_mel_frames = mel.dims()[2];

        // 2. Run audio encoder
        let audio_features = self.audio_encoder.forward(&mel)?;
        let n_audio_tokens = get_feat_extract_output_lengths(n_mel_frames);
        let encoder_output_len = audio_features.dims()[1];
        tracing::debug!(
            "Audio encoder: {} tokens (expected {}), mel_frames={}",
            encoder_output_len, n_audio_tokens, n_mel_frames
        );

        // Trim or pad audio features to match expected token count
        let audio_features = if encoder_output_len != n_audio_tokens {
            tracing::warn!(
                "Audio encoder output {} tokens, expected {}; trimming",
                encoder_output_len, n_audio_tokens
            );
            audio_features.narrow(1, 0, n_audio_tokens.min(encoder_output_len))?
        } else {
            audio_features
        };

        // 3. Build prompt
        let (input_ids, audio_positions) = tokenizer.encode_asr_prompt(n_audio_tokens, language)?;

        // 4. Embed text tokens
        let input_ids_tensor = Tensor::from_vec(
            input_ids.iter().map(|&id| id as i64).collect::<Vec<_>>(),
            (1, input_ids.len()),
            &self.device,
        )?;
        let mut embeds = self.text_decoder.embed_tokens.forward(&input_ids_tensor)?;

        // 5. Replace audio positions with audio features
        let hidden_size = self.config.thinker_config.text_config.hidden_size;
        for (feat_idx, &pos) in audio_positions.iter().enumerate() {
            if feat_idx < audio_features.dims()[1] {
                let feat = audio_features.i((0, feat_idx))?.reshape((1, 1, hidden_size))?;
                // Replace embedding at position `pos`
                let before = if pos > 0 { Some(embeds.narrow(1, 0, pos)?) } else { None };
                let after_start = pos + 1;
                let after_len = embeds.dims()[1] - after_start;
                let after = if after_len > 0 { Some(embeds.narrow(1, after_start, after_len)?) } else { None };

                let mut parts: Vec<Tensor> = Vec::new();
                if let Some(b) = before { parts.push(b); }
                parts.push(feat);
                if let Some(a) = after { parts.push(a); }
                embeds = Tensor::cat(&parts, 1)?;
            }
        }

        // 6. Build position IDs for MRoPE (3, batch, seq_len)
        let seq_len = embeds.dims()[1];
        let position_ids = self.build_position_ids(seq_len, &audio_positions)?;

        // 7. First forward pass with embeddings
        self.text_decoder.clear_cache();
        let logits = self.text_decoder.forward_embeds(&embeds, &position_ids)?;

        // 8. Autoregressive generation
        let max_new_tokens = 2048usize;
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut next_logits = logits;
        let mut cur_pos = seq_len;

        for _ in 0..max_new_tokens {
            // Get logits for last position
            let last_logits = next_logits.i((0, next_logits.dims()[1] - 1))?;
            let next_token = last_logits.argmax(D::Minus1)?.to_scalar::<u32>()?;

            if EOS_TOKEN_IDS.contains(&next_token) {
                break;
            }

            generated_tokens.push(next_token);

            // Prepare next step
            let next_ids = Tensor::from_vec(vec![next_token as i64], (1, 1), &self.device)?;
            let pos_ids = self.build_position_ids_single(cur_pos)?;
            next_logits = self.text_decoder.forward_ids(&next_ids, &pos_ids)?;
            cur_pos += 1;
        }

        // 9. Decode tokens to text
        let text = tokenizer.decode(&generated_tokens)?;

        // 10. Clean up the output (strip language tags etc.)
        Ok(clean_transcription(&text))
    }

    /// Build 3-dimensional position IDs for MRoPE.
    /// All 3 dims have the same sequential positions (0, 1, 2, ..., seq_len-1),
    /// matching the Python reference which uses `attention_mask.cumsum(-1) - 1`.
    fn build_position_ids(&self, seq_len: usize, _audio_positions: &[usize]) -> Result<Tensor> {
        let positions: Vec<i64> = (0..seq_len as i64).collect();
        let pos = Tensor::from_vec(positions, (1, seq_len), &self.device)?;
        Tensor::stack(&[&pos, &pos, &pos], 0)
    }

    /// Build position IDs for a single new token during generation.
    fn build_position_ids_single(&self, pos: usize) -> Result<Tensor> {
        let p = pos as i64;
        let t = Tensor::from_vec(vec![p], (1, 1), &self.device)?;
        let h = Tensor::from_vec(vec![p], (1, 1), &self.device)?;
        let w = Tensor::from_vec(vec![p], (1, 1), &self.device)?;
        Tensor::stack(&[t, h, w], 0)
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
    let emb_scale: Vec<f64> = (0..half_dim)
        .map(|i| (-log_10000 * i as f64 / (half_dim - 1) as f64).exp())
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

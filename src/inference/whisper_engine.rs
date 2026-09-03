//! Whisper (large-v3-turbo) bench engine on top of candle-transformers.
//!
//! Bench-only: used by `voclaude --bench --engine whisper --model-dir <dir>`
//! to compare against Qwen3-ASR on the same recordings. Windows of at most
//! 30 s are cut at pauses with the live segmenter, the language is detected
//! per window, and tokens are decoded greedily.

use crate::audio::segmenter::{segment_all, SegmenterConfig};
use crate::inference::candle_audio;
use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as w, model::Whisper, Config};
use std::path::Path;
use std::time::Instant;
use tracing::{debug, info};

const WINDOW_SAMPLES: usize = w::N_SAMPLES; // 30 s
const MAX_NEW_TOKENS: usize = 224; // max_target_positions / 2

pub struct WhisperEngine {
    model: Whisper,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
    dtype: DType,
    sot: u32,
    eot: u32,
    transcribe: u32,
    no_timestamps: u32,
    no_speech: u32,
    /// (token id, language code) for every `<|xx|>` token.
    lang_tokens: Vec<(u32, String)>,
    /// First timestamp token id; everything from here up is masked out.
    first_timestamp: u32,
    pub n_generated: usize,
    pub decode_ms: f64,
    pub encode_ms: f64,
}

impl WhisperEngine {
    pub fn load(model_dir: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let cfg_text = std::fs::read_to_string(model_dir.join("config.json"))
            .map_err(|e| candle_core::Error::Msg(format!("config.json: {}", e)))?;
        let raw: serde_json::Value = serde_json::from_str(&cfg_text)
            .map_err(|e| candle_core::Error::Msg(format!("config.json parse: {}", e)))?;
        let get = |k: &str| raw.get(k).and_then(|v| v.as_u64()).map(|v| v as usize);
        let config = Config {
            num_mel_bins: get("num_mel_bins").unwrap_or(128),
            max_source_positions: get("max_source_positions").unwrap_or(1500),
            d_model: get("d_model").unwrap_or(1280),
            encoder_attention_heads: get("encoder_attention_heads").unwrap_or(20),
            encoder_layers: get("encoder_layers").unwrap_or(32),
            vocab_size: get("vocab_size").unwrap_or(51866),
            max_target_positions: get("max_target_positions").unwrap_or(448),
            decoder_attention_heads: get("decoder_attention_heads").unwrap_or(20),
            decoder_layers: get("decoder_layers").unwrap_or(4),
            suppress_tokens: raw
                .get("suppress_tokens")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_u64().map(|x| x as u32)).collect())
                .unwrap_or_default(),
        };
        if config.num_mel_bins != candle_audio::N_MELS {
            return Err(candle_core::Error::Msg(format!(
                "whisper model wants {} mel bins, front-end produces {}",
                config.num_mel_bins,
                candle_audio::N_MELS
            )));
        }
        let weights = model_dir.join("model.safetensors");
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], dtype, device)? };
        let model = Whisper::load(&vb, config)?;

        let tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| candle_core::Error::Msg(format!("tokenizer: {}", e)))?;
        let tok = |s: &str| -> Result<u32> {
            tokenizer
                .token_to_id(s)
                .ok_or_else(|| candle_core::Error::Msg(format!("missing token {}", s)))
        };
        let sot = tok(w::SOT_TOKEN)?;
        let eot = tok(w::EOT_TOKEN)?;
        let transcribe = tok(w::TRANSCRIBE_TOKEN)?;
        let no_timestamps = tok(w::NO_TIMESTAMPS_TOKEN)?;
        let no_speech = tok("<|nospeech|>").or_else(|_| tok("<|nocaptions|>"))?;
        let first_timestamp = tok("<|0.00|>").unwrap_or(no_timestamps + 1);

        let mut lang_tokens = Vec::new();
        for (name, id) in tokenizer.get_added_vocabulary().get_vocab() {
            if let Some(code) = name.strip_prefix("<|").and_then(|s| s.strip_suffix("|>")) {
                if (2..=3).contains(&code.len()) && code.chars().all(|c| c.is_ascii_lowercase()) {
                    lang_tokens.push((*id, code.to_string()));
                }
            }
        }
        lang_tokens.sort();
        info!(
            "Whisper loaded: {} language tokens, dtype {:?}",
            lang_tokens.len(),
            dtype
        );

        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
            dtype,
            sot,
            eot,
            transcribe,
            no_timestamps,
            no_speech,
            lang_tokens,
            first_timestamp,
            n_generated: 0,
            decode_ms: 0.0,
            encode_ms: 0.0,
        })
    }

    pub fn reset_stats(&mut self) {
        self.n_generated = 0;
        self.decode_ms = 0.0;
        self.encode_ms = 0.0;
    }

    /// Transcribe a recording of any length. Returns the joined text and the
    /// language detected per window.
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<(String, Vec<String>)> {
        let seg_cfg = SegmenterConfig { min_secs: 20.0, max_secs: 30.0, pause_secs: 0.5 };
        let mut texts = Vec::new();
        let mut langs = Vec::new();
        for b in segment_all(samples, seg_cfg) {
            let (text, lang) = self.transcribe_window(&samples[b.start..b.end])?;
            langs.push(lang);
            let t = text.trim();
            if !t.is_empty() {
                texts.push(t.to_string());
            }
        }
        Ok((texts.join(" "), langs))
    }

    /// One ≤30 s window: pad to 30 s, encode, detect language, greedy decode.
    pub fn transcribe_window(&mut self, samples: &[f32]) -> Result<(String, String)> {
        let mut padded = samples.to_vec();
        padded.truncate(WINDOW_SAMPLES);
        padded.resize(WINDOW_SAMPLES, 0.0);

        let t_enc = Instant::now();
        let mel = candle_audio::pcm_to_mel(&padded, &self.device)?.to_dtype(self.dtype)?; // (1, 128, 3000)
        let n_frames = mel.dims()[2];
        let mel = if n_frames > w::N_FRAMES {
            mel.narrow(2, 0, w::N_FRAMES)?
        } else if n_frames < w::N_FRAMES {
            let pad = Tensor::zeros((1, candle_audio::N_MELS, w::N_FRAMES - n_frames), self.dtype, &self.device)?;
            Tensor::cat(&[&mel, &pad], 2)?
        } else {
            mel
        };
        let audio_features = self.model.encoder.forward(&mel, true)?;
        self.device.synchronize()?;
        self.encode_ms += t_enc.elapsed().as_secs_f64() * 1000.0;

        let t_dec = Instant::now();
        // Language detection: one decoder step on [sot], argmax over language tokens.
        let lang_id = {
            let tokens = Tensor::new(&[self.sot], &self.device)?.unsqueeze(0)?;
            let ys = self.model.decoder.forward(&tokens, &audio_features, true)?;
            let logits = self.model.decoder.final_linear(&ys.i((..1, 0..1))?)?.i(0)?.i(0)?;
            let logits = logits.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let mut best = (f32::NEG_INFINITY, self.lang_tokens[0].0);
            for (id, _) in &self.lang_tokens {
                let v = logits[*id as usize];
                if v > best.0 {
                    best = (v, *id);
                }
            }
            best.1
        };
        let lang_code = self
            .lang_tokens
            .iter()
            .find(|(id, _)| *id == lang_id)
            .map(|(_, c)| c.clone())
            .unwrap_or_default();

        let mut tokens: Vec<u32> = vec![self.sot, lang_id, self.transcribe, self.no_timestamps];
        let prompt_len = tokens.len();
        let mut no_speech = false;
        for i in 0..MAX_NEW_TOKENS {
            let tokens_t = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let ys = self.model.decoder.forward(&tokens_t, &audio_features, i == 0)?;
            let (_, seq_len, _) = ys.dims3()?;
            let logits = self
                .model
                .decoder
                .final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?;
            let mut logits = logits.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            if i == 0 {
                // No-speech probability check (whisper's threshold on the first step).
                let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
                let sum: f32 = exp.iter().sum();
                let p_no_speech = exp[self.no_speech as usize] / sum;
                if p_no_speech > w::NO_SPEECH_THRESHOLD as f32 {
                    no_speech = true;
                    break;
                }
                // begin_suppress_tokens: blank and eot at the first position
                logits[220] = f32::NEG_INFINITY;
                logits[self.eot as usize] = f32::NEG_INFINITY;
            }
            // Never emit control/timestamp tokens as text.
            for id in self.first_timestamp as usize..logits.len() {
                logits[id] = f32::NEG_INFINITY;
            }
            logits[self.no_speech as usize] = f32::NEG_INFINITY;
            logits[self.sot as usize] = f32::NEG_INFINITY;
            logits[self.transcribe as usize] = f32::NEG_INFINITY;
            logits[self.no_timestamps as usize] = f32::NEG_INFINITY;
            let mut best = (f32::NEG_INFINITY, self.eot);
            for (id, v) in logits.iter().enumerate() {
                if *v > best.0 {
                    best = (*v, id as u32);
                }
            }
            let next = best.1;
            if next == self.eot || tokens.len() >= self.model.config.max_target_positions - 1 {
                break;
            }
            tokens.push(next);
        }
        self.device.synchronize()?;
        self.decode_ms += t_dec.elapsed().as_secs_f64() * 1000.0;
        let generated = &tokens[prompt_len..];
        self.n_generated += generated.len();
        let text = if no_speech {
            String::new()
        } else {
            self.tokenizer
                .decode(generated, true)
                .map_err(|e| candle_core::Error::Msg(format!("decode: {}", e)))?
        };
        debug!("whisper window: lang={} tokens={} text_len={}", lang_code, generated.len(), text.len());
        let _ = D::Minus1;
        Ok((text, lang_code))
    }
}

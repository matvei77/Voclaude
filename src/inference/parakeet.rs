//! NVIDIA Parakeet-TDT-0.6B-V2 inference engine.
//!
//! Uses sherpa-onnx ONNX export with separate encoder + decoder + joiner.
//! Implements Token-and-Duration Transducer (TDT) greedy decoding.

use super::mel::MelSpectrogram;
use super::model_manager::ModelManager;

use ort::{
    execution_providers::CUDAExecutionProvider,
    session::Session,
    value::Tensor,
};
use std::collections::HashMap;
use std::fs;
use tracing::{debug, info};

/// TDT model constants - retrieved from model metadata
const PRED_RNN_LAYERS: usize = 2;  // Model has 2 LSTM layers in prediction network
const PRED_HIDDEN: usize = 640;

/// Parakeet TDT inference engine with CUDA acceleration
pub struct ParakeetEngine {
    encoder: Option<Session>,
    decoder: Option<Session>,
    joiner: Option<Session>,
    tokens: Option<HashMap<i64, String>>,
    vocab_size: usize,
    blank_id: i64,
    mel: MelSpectrogram,
    is_loaded: bool,
}

impl ParakeetEngine {
    /// Create a new engine (lazy loading - doesn't load model yet)
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            encoder: None,
            decoder: None,
            joiner: None,
            tokens: None,
            vocab_size: 0,
            blank_id: 0,
            mel: MelSpectrogram::new(),
            is_loaded: false,
        })
    }

    /// Check if model is loaded
    pub fn is_loaded(&self) -> bool {
        self.is_loaded
    }

    /// Unload model to free GPU memory
    pub fn unload(&mut self) {
        if self.is_loaded {
            info!("Unloading Parakeet model");
            self.encoder = None;
            self.decoder = None;
            self.joiner = None;
            self.tokens = None;
            self.is_loaded = false;
        }
    }

    /// Load tokens from tokens.txt
    /// Format: "token index" per line
    fn load_tokens(tokens_path: &std::path::Path) -> Result<(HashMap<i64, String>, usize), Box<dyn std::error::Error>> {
        let content = fs::read_to_string(tokens_path)?;
        let mut tokens = HashMap::new();
        let mut max_id: i64 = 0;

        for line in content.lines() {
            // Format: "token index" - split from the right
            if let Some(last_space) = line.rfind(' ') {
                let token = &line[..last_space];
                if let Ok(idx) = line[last_space + 1..].parse::<i64>() {
                    tokens.insert(idx, token.to_string());
                    if idx > max_id {
                        max_id = idx;
                    }
                }
            }
        }

        let vocab_size = (max_id + 1) as usize;
        info!("Loaded {} tokens, vocab_size={}", tokens.len(), vocab_size);
        Ok((tokens, vocab_size))
    }

    /// Ensure model is loaded (lazy loading)
    fn ensure_loaded(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.is_loaded {
            return Ok(());
        }

        info!("Loading Parakeet TDT model with CUDA...");

        // Ensure all model files are downloaded
        let encoder_path = ModelManager::ensure_encoder()?;
        let decoder_path = ModelManager::ensure_decoder()?;
        let joiner_path = ModelManager::ensure_joiner()?;
        let tokens_path = ModelManager::ensure_tokens()?;

        // Initialize ONNX Runtime
        ort::init()
            .with_name("voclaude")
            .commit()?;

        // Create encoder session with CUDA
        info!("Loading encoder...");
        let encoder = Session::builder()?
            .with_execution_providers([
                CUDAExecutionProvider::default().build(),
            ])?
            .commit_from_file(&encoder_path)?;
        info!("Encoder loaded");

        // Create decoder session with CUDA
        info!("Loading decoder...");
        let decoder = Session::builder()?
            .with_execution_providers([
                CUDAExecutionProvider::default().build(),
            ])?
            .commit_from_file(&decoder_path)?;
        info!("Decoder loaded");

        // Create joiner session with CUDA
        info!("Loading joiner...");
        let joiner = Session::builder()?
            .with_execution_providers([
                CUDAExecutionProvider::default().build(),
            ])?
            .commit_from_file(&joiner_path)?;
        info!("Joiner loaded");

        // Load tokens
        let (tokens, vocab_size) = Self::load_tokens(&tokens_path)?;
        let blank_id = (vocab_size - 1) as i64;  // Blank is last token

        self.encoder = Some(encoder);
        self.decoder = Some(decoder);
        self.joiner = Some(joiner);
        self.tokens = Some(tokens);
        self.vocab_size = vocab_size;
        self.blank_id = blank_id;
        self.is_loaded = true;

        info!("Parakeet TDT model loaded! vocab_size={}, blank_id={}", vocab_size, blank_id);
        Ok(())
    }

    /// Decode token IDs to text using vocabulary
    fn decode_tokens(&self, token_ids: &[i64]) -> String {
        let tokens = match &self.tokens {
            Some(t) => t,
            None => return String::new(),
        };

        let mut text = String::new();
        for &id in token_ids {
            // Skip blank and invalid tokens
            if id == self.blank_id || id < 0 || id as usize >= self.vocab_size {
                continue;
            }

            if let Some(token) = tokens.get(&id) {
                // Handle SentencePiece-style tokens (▁ prefix means space)
                if token.starts_with('▁') {
                    if !text.is_empty() {
                        text.push(' ');
                    }
                    text.push_str(&token[3..]); // Skip the ▁ character (3 bytes in UTF-8)
                } else if token == "<space>" {
                    text.push(' ');
                } else if !token.starts_with('<') {
                    // Skip special tokens like <blk>, <unk>
                    text.push_str(token);
                }
            }
        }

        text.trim().to_string()
    }

    /// Transcribe audio samples using TDT decoding
    /// Input: f32 samples at 16kHz mono
    /// Output: transcribed text
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<String, Box<dyn std::error::Error>> {
        self.ensure_loaded()?;

        // Run encoder and TDT decoding
        let output_tokens = self.run_inference(samples)?;

        // Decode tokens to text
        let text = self.decode_tokens(&output_tokens);
        debug!("Transcription: {}", text);

        Ok(text)
    }

    /// Helper to run decoder and extract outputs
    fn run_decoder(
        decoder: &mut Session,
        context: i64,
        state_h: &[f32],
        state_c: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
        let context_tensor = Tensor::from_array(([1_usize, 1_usize], vec![context as i32].into_boxed_slice()))?;
        let context_len_tensor = Tensor::from_array(([1_usize], vec![1_i32].into_boxed_slice()))?;
        let state_h_tensor = Tensor::from_array(([PRED_RNN_LAYERS, 1, PRED_HIDDEN], state_h.to_vec().into_boxed_slice()))?;
        let state_c_tensor = Tensor::from_array(([PRED_RNN_LAYERS, 1, PRED_HIDDEN], state_c.to_vec().into_boxed_slice()))?;

        // Get actual input names from model (clone to owned strings to avoid borrow conflict)
        let dec_in0 = decoder.inputs.get(0).map(|i| i.name.clone()).unwrap_or_else(|| "y".to_string());
        let dec_in1 = decoder.inputs.get(1).map(|i| i.name.clone()).unwrap_or_else(|| "y_lens".to_string());
        let dec_in2 = decoder.inputs.get(2).map(|i| i.name.clone()).unwrap_or_else(|| "state0".to_string());
        let dec_in3 = decoder.inputs.get(3).map(|i| i.name.clone()).unwrap_or_else(|| "state1".to_string());

        let outputs = decoder.run(ort::inputs![
            dec_in0.as_str() => context_tensor,
            dec_in1.as_str() => context_len_tensor,
            dec_in2.as_str() => state_h_tensor,
            dec_in3.as_str() => state_c_tensor,
        ])?;

        // Extract decoder output
        let dec_out_name = outputs.iter().next().map(|(n, _)| n.to_string());
        let dec_out = outputs.get("decoder_out")
            .or_else(|| dec_out_name.as_ref().and_then(|n| outputs.get(n)))
            .ok_or("Missing decoder output")?;
        let (_, dec_data) = dec_out.try_extract_tensor::<f32>()?;
        let decoder_out: Vec<f32> = dec_data.iter().cloned().collect();

        // Extract new states
        let mut new_h = state_h.to_vec();
        let mut new_c = state_c.to_vec();

        if let Some(h) = outputs.get("state0_out").or_else(|| outputs.get("out_state0")) {
            if let Ok((_, data)) = h.try_extract_tensor::<f32>() {
                for (i, &v) in data.iter().enumerate().take(PRED_RNN_LAYERS * PRED_HIDDEN) {
                    new_h[i] = v;
                }
            }
        }
        if let Some(c) = outputs.get("state1_out").or_else(|| outputs.get("out_state1")) {
            if let Ok((_, data)) = c.try_extract_tensor::<f32>() {
                for (i, &v) in data.iter().enumerate().take(PRED_RNN_LAYERS * PRED_HIDDEN) {
                    new_c[i] = v;
                }
            }
        }

        Ok((decoder_out, new_h, new_c))
    }

    /// Helper to run joiner and get logits
    fn run_joiner(
        joiner: &mut Session,
        enc_frame: Vec<f32>,
        dec_frame: Vec<f32>,
        enc_dim: usize,
        dec_dim: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let enc_tensor = Tensor::from_array(([1, enc_dim, 1], enc_frame.into_boxed_slice()))?;
        let dec_tensor = Tensor::from_array(([1, dec_dim, 1], dec_frame.into_boxed_slice()))?;

        // Get actual input names from model (clone to owned strings to avoid borrow conflict)
        let join_in0 = joiner.inputs.get(0).map(|i| i.name.clone()).unwrap_or_else(|| "encoder_out".to_string());
        let join_in1 = joiner.inputs.get(1).map(|i| i.name.clone()).unwrap_or_else(|| "decoder_out".to_string());

        let outputs = joiner.run(ort::inputs![
            join_in0.as_str() => enc_tensor,
            join_in1.as_str() => dec_tensor,
        ])?;

        let logit_name = outputs.iter().next().map(|(n, _)| n.to_string());
        let logits = outputs.get("logit")
            .or_else(|| logit_name.as_ref().and_then(|n| outputs.get(n)))
            .ok_or("Missing joiner output")?;
        let (_, logit_data) = logits.try_extract_tensor::<f32>()?;

        Ok(logit_data.iter().cloned().collect())
    }

    /// Run encoder and TDT greedy decoding
    fn run_inference(&mut self, samples: &[f32]) -> Result<Vec<i64>, Box<dyn std::error::Error>> {
        debug!("Transcribing {} samples ({:.2}s)", samples.len(), samples.len() as f32 / 16000.0);

        // Compute mel spectrogram: [n_mels, n_frames]
        let mel_spec = self.mel.compute(samples);
        let (n_mels, n_frames) = mel_spec.dim();
        debug!("Mel spectrogram: {} mels x {} frames", n_mels, n_frames);

        // Prepare encoder input: [batch=1, n_mels, time]
        let mel_vec: Vec<f32> = mel_spec.iter().cloned().collect();
        let mel_tensor = Tensor::from_array(([1, n_mels, n_frames], mel_vec.into_boxed_slice()))?;
        let length_tensor = Tensor::from_array(([1_usize], vec![n_frames as i64].into_boxed_slice()))?;

        // Run encoder - get input names dynamically
        debug!("Running encoder...");
        let encoder = self.encoder.as_mut().ok_or("Encoder not loaded")?;

        // Get actual input names from model (clone to owned strings to avoid borrow conflict)
        let enc_in0 = encoder.inputs.get(0).map(|i| i.name.clone()).unwrap_or_else(|| "x".to_string());
        let enc_in1 = encoder.inputs.get(1).map(|i| i.name.clone()).unwrap_or_else(|| "x_lens".to_string());
        debug!("Encoder input names: '{}', '{}'", enc_in0, enc_in1);

        let encoder_outputs = encoder.run(ort::inputs![
            enc_in0.as_str() => mel_tensor,
            enc_in1.as_str() => length_tensor,
        ])?;

        // Get encoder output - shape: [batch, dim, time]
        let (enc_data, enc_dim, enc_time) = {
            let enc_out_name = encoder_outputs.iter().next().map(|(n, _)| n.to_string());
            let enc_out = encoder_outputs.get("encoder_out")
                .or_else(|| enc_out_name.as_ref().and_then(|n| encoder_outputs.get(n)))
                .ok_or("Missing encoder output")?;
            let (enc_shape, enc_data_view) = enc_out.try_extract_tensor::<f32>()?;
            debug!("Encoder output shape: {:?}", enc_shape);

            // Parse encoder dimensions: [batch, dim, time]
            let enc_time = if enc_shape.len() == 3 {
                enc_shape[2] as usize
            } else {
                return Err(format!("Unexpected encoder shape: {:?}", enc_shape).into());
            };
            let enc_dim = enc_shape[1] as usize;

            // Copy encoder data so we can drop the session outputs
            let enc_data: Vec<f32> = enc_data_view.iter().cloned().collect();
            (enc_data, enc_dim, enc_time)
        };
        debug!("Encoder: {} frames x {} dim", enc_time, enc_dim);

        // TDT Greedy Decoding
        debug!("Starting TDT greedy decoding...");

        // Initialize decoder state: zeros for h and c
        let mut state_h = vec![0.0f32; PRED_RNN_LAYERS * PRED_HIDDEN];
        let mut state_c = vec![0.0f32; PRED_RNN_LAYERS * PRED_HIDDEN];

        // Initialize with blank context
        let mut context: i64 = self.blank_id;
        let mut output_tokens: Vec<i64> = Vec::new();

        // Run decoder once to get initial decoder_out (with blank context)
        let decoder = self.decoder.as_mut().ok_or("Decoder not loaded")?;
        let (initial_dec_out, state_h_next, state_c_next) = Self::run_decoder(decoder, context, &state_h, &state_c)?;

        let dec_dim = initial_dec_out.len();
        let mut decoder_out = initial_dec_out;
        let mut state_h_next = state_h_next;
        let mut state_c_next = state_c_next;

        debug!("Decoder output dim: {}", dec_dim);

        // Main decoding loop
        let mut t: usize = 0;
        while t < enc_time {
            // Extract encoder frame at time t
            let enc_frame: Vec<f32> = (0..enc_dim).map(|d| {
                // Data layout: [batch, dim, time] - batch=0, so index is d * enc_time + t
                let idx = d * enc_time + t;
                if idx < enc_data.len() { enc_data[idx] } else { 0.0 }
            }).collect();

            // Run joiner
            let joiner = self.joiner.as_mut().ok_or("Joiner not loaded")?;
            let logits = Self::run_joiner(joiner, enc_frame, decoder_out.clone(), enc_dim, dec_dim)?;

            // Split logits: first vocab_size are token logits, rest are duration logits
            let token_logits = &logits[..self.vocab_size.min(logits.len())];
            let duration_logits = if logits.len() > self.vocab_size {
                &logits[self.vocab_size..]
            } else {
                &[] as &[f32]
            };

            // Find best token
            let mut best_token: i64 = self.blank_id;
            let mut best_score = f32::NEG_INFINITY;
            for (i, &score) in token_logits.iter().enumerate() {
                if score > best_score {
                    best_score = score;
                    best_token = i as i64;
                }
            }

            // Find best duration (skip value)
            let skip: usize = if !duration_logits.is_empty() {
                let mut best_dur_idx: usize = 0;
                let mut best_dur_score = f32::NEG_INFINITY;
                for (i, &score) in duration_logits.iter().enumerate() {
                    if score > best_dur_score {
                        best_dur_score = score;
                        best_dur_idx = i;
                    }
                }
                best_dur_idx
            } else {
                1 // Default skip of 1 if no duration logits
            };

            if best_token != self.blank_id {
                // Non-blank token: emit token and update decoder state
                output_tokens.push(best_token);
                context = best_token;

                // Commit state update
                state_h = state_h_next;
                state_c = state_c_next;

                // Run decoder with new context to get new decoder_out
                let decoder = self.decoder.as_mut().ok_or("Decoder not loaded")?;
                let (new_dec_out, new_h, new_c) = Self::run_decoder(decoder, context, &state_h, &state_c)?;
                decoder_out = new_dec_out;
                state_h_next = new_h;
                state_c_next = new_c;
            }

            // Advance time by skip value (minimum 1 to ensure progress)
            t += skip.max(1);
        }

        debug!("Decoded {} tokens", output_tokens.len());
        Ok(output_tokens)
    }
}

impl Default for ParakeetEngine {
    fn default() -> Self {
        Self::new().expect("Failed to create ParakeetEngine")
    }
}

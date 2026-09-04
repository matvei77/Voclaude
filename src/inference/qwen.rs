//! Qwen speech-to-text inference engine (Candle native backend).
//!
//! Loads Qwen3-ASR-1.7B safetensors directly via candle-core and runs
//! inference entirely in Rust. No Python dependency.

use crate::audio::TARGET_SAMPLE_RATE;
use crate::config::Config;
use crate::inference::candle_backend::{Quantization, Qwen3ASRModel, TranscribeStats};
use crate::inference::candle_tokenizer::{PromptStyle, Qwen3ASRTokenizer};
use crate::inference::{AsrEngine, InferenceProgress, InferenceStage};
use candle_core::{DType, Device};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing::{debug, info, warn};

const DEFAULT_MODEL_SIZE_MB: u64 = 3300;

const SAMPLE_RATE: usize = TARGET_SAMPLE_RATE as usize;

/// Maximum chunk size in samples (5 minutes at 16kHz).
/// At 13 audio tokens/second, 5 min = 3900 tokens. With ~30 prompt tokens
/// and 2048 max_new_tokens, total positions = ~5978, well within MRoPE's 16384 limit.
const MAX_CHUNK_SAMPLES: usize = 5 * 60 * SAMPLE_RATE;
const MIN_CHUNK_SAMPLES: usize = 10 * SAMPLE_RATE;

/// Number of transcriptions before proactively resetting the CUDA context
/// to prevent driver-level memory fragmentation from accumulating to OOM.
const DEFRAG_INTERVAL: u32 = 250;

pub struct QwenEngine {
    active_gpu: bool,
    use_gpu: bool,
    language: Option<String>,
    model_id: String,
    model_path: Option<String>,
    max_new_tokens: u32,
    adaptive_max_new_tokens: bool,
    max_chunk_samples: usize,
    require_gpu: bool,
    model: Option<Qwen3ASRModel>,
    tokenizer: Option<Qwen3ASRTokenizer>,
    device: Device,
    dtype: DType,
    /// Tracks consecutive transcriptions since last model load/reload.
    transcription_count: u32,
    /// Chat template to use (official unless `legacy_prompt` is configured).
    prompt_style: PromptStyle,
    /// Per-phase timing accumulated since the last `take_stats()`.
    stats: TranscribeStats,
    /// Decoder weight format.
    quantization: Quantization,
}

impl QwenEngine {
    pub fn new(use_gpu: bool) -> Result<Self, Box<dyn std::error::Error>> {
        let (device, dtype) = if use_gpu {
            match try_gpu_device() {
                Ok(dev) => (dev, DType::F16),
                Err(e) => {
                    warn!("GPU not available: {}; falling back to CPU", e);
                    (Device::Cpu, DType::F32)
                }
            }
        } else {
            (Device::Cpu, DType::F32)
        };

        let active_gpu = !device.is_cpu();

        Ok(Self {
            active_gpu,
            use_gpu,
            language: None,
            model_id: "Qwen/Qwen3-ASR-1.7B".to_string(),
            model_path: None,
            max_new_tokens: 2048,
            adaptive_max_new_tokens: true,
            max_chunk_samples: MAX_CHUNK_SAMPLES,
            require_gpu: true,
            model: None,
            tokenizer: None,
            device,
            dtype,
            transcription_count: 0,
            prompt_style: PromptStyle::Official,
            stats: TranscribeStats::default(),
            quantization: Quantization::None,
        })
    }

    pub fn new_with_config(config: &Config) -> Result<Self, Box<dyn std::error::Error>> {
        let mut engine = Self::new(config.use_gpu)?;
        engine.language = normalize_language(config.language.as_deref());
        engine.model_id = config.model.clone();
        engine.model_path = config.model_path.clone();
        engine.max_new_tokens = config.max_new_tokens;
        engine.adaptive_max_new_tokens = config.adaptive_max_new_tokens;
        engine.max_chunk_samples = ((config.max_chunk_seconds as usize) * SAMPLE_RATE)
            .clamp(MIN_CHUNK_SAMPLES, MAX_CHUNK_SAMPLES);
        engine.require_gpu = config.require_gpu;
        engine.prompt_style = if config.legacy_prompt {
            PromptStyle::Legacy
        } else {
            PromptStyle::Official
        };
        engine.quantization = Quantization::parse(&config.quantization)
            .ok_or_else(|| format!("Unknown quantization: {}", config.quantization))?;
        Ok(engine)
    }

    /// Per-phase timing accumulated since the previous call; resets the counter.
    pub fn take_stats(&mut self) -> TranscribeStats {
        std::mem::take(&mut self.stats)
    }

    pub fn prompt_style(&self) -> PromptStyle {
        self.prompt_style
    }

    pub fn is_loaded(&self) -> bool {
        self.model.is_some()
    }

    #[allow(dead_code)]
    pub fn set_prompt_style(&mut self, style: PromptStyle) {
        self.prompt_style = style;
        if let Some(model) = self.model.as_mut() {
            model.prompt_style = style;
        }
    }

    /// Transcribe one segment of a live recording.
    ///
    /// `context` is the tail of the previous segment's transcript; it is fed to
    /// the model's system slot so language and vocabulary carry over. Loads the
    /// model if needed and retries once after a CUDA out-of-memory error.
    pub fn transcribe_segment(
        &mut self,
        samples: &[f32],
        context: Option<&str>,
    ) -> Result<(String, TranscribeStats), Box<dyn std::error::Error>> {
        self.prepare(None)?;
        match self.run_segment(samples, context) {
            Ok(out) => {
                self.transcription_count += 1;
                Ok(out)
            }
            Err(e) => {
                let msg = format!("{}", e);
                if Self::is_cuda_oom(&msg) && self.use_gpu {
                    warn!("CUDA OOM on segment, reloading model and retrying once");
                    self.unload();
                    self.prepare(None)?;
                    let out = self.run_segment(samples, context)?;
                    self.transcription_count = 1;
                    Ok(out)
                } else {
                    Err(e)
                }
            }
        }
    }

    fn run_segment(
        &mut self,
        samples: &[f32],
        context: Option<&str>,
    ) -> Result<(String, TranscribeStats), Box<dyn std::error::Error>> {
        if samples.is_empty() {
            return Ok((String::new(), TranscribeStats::default()));
        }
        let model = self.model.as_mut().ok_or("Model not loaded")?;
        let tokenizer = self.tokenizer.as_ref().ok_or("Tokenizer not loaded")?;
        model.max_new_tokens = adaptive_token_limit(
            samples.len(),
            self.max_new_tokens as usize,
            self.adaptive_max_new_tokens,
        );
        let (text, stats) = model
            .transcribe_with_stats(samples, context, tokenizer)
            .map_err(|e| format!("{}", e))?;
        self.stats.add(&stats);
        debug!(
            "Segment: {:.1}s audio, mel {:.0}ms, encode {:.0}ms, prefill {:.0}ms, decode {:.0}ms ({} tok, {:.1} tok/s)",
            stats.audio_secs, stats.mel_ms, stats.encode_ms, stats.prefill_ms, stats.decode_ms,
            stats.n_generated, stats.decode_tokens_per_sec()
        );
        Ok((text, stats))
    }

    #[allow(dead_code)]
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<String, Box<dyn std::error::Error>> {
        self.transcribe_with_progress(samples, None)
    }

    pub fn transcribe_with_progress(
        &mut self,
        samples: &[f32],
        mut progress: Option<&mut dyn FnMut(InferenceProgress)>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        // Proactive CUDA defrag: periodically unload/reload to prevent
        // driver-level memory fragmentation from accumulating to OOM.
        if self.active_gpu && self.transcription_count > 0 && self.transcription_count % DEFRAG_INTERVAL == 0 {
            info!(
                "Proactive CUDA defrag after {} transcriptions",
                self.transcription_count
            );
            if let Some(cb) = progress.as_deref_mut() {
                cb(InferenceProgress {
                    stage: InferenceStage::LoadingModel,
                    message: "Optimizing GPU memory...".to_string(),
                });
            }
            self.unload();
        }

        self.prepare(progress.as_deref_mut().map(|p| p as &mut dyn FnMut(InferenceProgress)))?;

        if let Some(cb) = progress.as_deref_mut() {
            cb(InferenceProgress {
                stage: InferenceStage::Transcribing,
                message: "Transcribing...".to_string(),
            });
        }

        let started = Instant::now();
        let result = self.run_transcription(samples);

        match result {
            Ok(text) => {
                self.transcription_count += 1;
                let elapsed = started.elapsed();
                info!(
                    "Transcription #{} complete: infer={:.2}s",
                    self.transcription_count,
                    elapsed.as_secs_f64()
                );
                Ok(text)
            }
            Err(e) => {
                let err_msg = format!("{}", e);
                if Self::is_cuda_oom(&err_msg) && self.use_gpu {
                    warn!(
                        "CUDA OOM on transcription #{}, recovering...",
                        self.transcription_count + 1
                    );
                    if let Some(cb) = progress.as_deref_mut() {
                        cb(InferenceProgress {
                            stage: InferenceStage::LoadingModel,
                            message: "Recovering from GPU memory pressure...".to_string(),
                        });
                    }

                    // Drop CUDA context entirely and reload with fresh memory state
                    self.unload();
                    self.prepare(progress.as_deref_mut().map(|p| p as &mut dyn FnMut(InferenceProgress)))?;

                    if let Some(cb) = progress.as_deref_mut() {
                        cb(InferenceProgress {
                            stage: InferenceStage::Transcribing,
                            message: "Retrying transcription...".to_string(),
                        });
                    }

                    let retry_result = self.run_transcription(samples);
                    match retry_result {
                        Ok(text) => {
                            self.transcription_count = 1; // reset after successful recovery
                            let elapsed = started.elapsed();
                            info!(
                                "Transcription recovered after OOM: infer={:.2}s",
                                elapsed.as_secs_f64()
                            );
                            Ok(text)
                        }
                        Err(retry_err) => {
                            Err(format!("Transcription failed: {}", retry_err).into())
                        }
                    }
                } else {
                    Err(format!("Transcription failed: {}", e).into())
                }
            }
        }
    }

    /// Run the actual model transcription. Extracted to allow OOM retry.
    /// Automatically chunks long audio (>5 min) to stay within MRoPE position limits.
    fn run_transcription(&mut self, samples: &[f32]) -> Result<String, Box<dyn std::error::Error>> {
        let model = self.model.as_mut().ok_or("Model not loaded")?;
        let tokenizer = self.tokenizer.as_ref().ok_or("Tokenizer not loaded")?;

        let chunk_limit = self.max_chunk_samples;
        if samples.len() <= chunk_limit {
            model.max_new_tokens = adaptive_token_limit(
                samples.len(),
                self.max_new_tokens as usize,
                self.adaptive_max_new_tokens,
            );
            let (text, stats) = model
                .transcribe_with_stats(samples, None, tokenizer)
                .map_err(|e| format!("{}", e))?;
            self.stats.add(&stats);
            return Ok(text);
        }

        // Chunk long audio at silence boundaries
        let chunks = split_audio_at_silence(samples, chunk_limit);
        info!(
            "Audio too long ({:.1}s), split into {} chunks (limit={}s)",
            samples.len() as f64 / SAMPLE_RATE as f64,
            chunks.len(),
            chunk_limit / SAMPLE_RATE
        );

        let mut texts = Vec::new();
        for (i, chunk) in chunks.iter().enumerate() {
            model.max_new_tokens = adaptive_token_limit(
                chunk.len(),
                self.max_new_tokens as usize,
                self.adaptive_max_new_tokens,
            );
            info!(
                "Transcribing chunk {}/{} ({:.1}s, max_new_tokens={})",
                i + 1,
                chunks.len(),
                chunk.len() as f64 / SAMPLE_RATE as f64,
                model.max_new_tokens
            );
            let (text, stats) = model
                .transcribe_with_stats(chunk, None, tokenizer)
                .map_err(|e| format!("Chunk {}/{} failed: {}", i + 1, chunks.len(), e))?;
            self.stats.add(&stats);
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                texts.push(trimmed.to_string());
            }
        }

        Ok(texts.join(" "))
    }

    /// Check if an error message indicates CUDA out-of-memory.
    fn is_cuda_oom(err_msg: &str) -> bool {
        let lower = err_msg.to_lowercase();
        lower.contains("out of memory") || lower.contains("out_of_memory")
    }

    fn resolve_model_dir(&self) -> Result<PathBuf, Box<dyn std::error::Error>> {
        // 1. Explicit config path
        if let Some(path) = &self.model_path {
            let p = PathBuf::from(path);
            // S-2: Canonicalize and validate the path
            if !p.exists() || !p.is_dir() {
                return Err(format!("Configured model_path not found: {}", path).into());
            }
            let canonical = p.canonicalize()
                .map_err(|e| format!("Cannot canonicalize model_path: {}", e))?;
            return Ok(canonical);
        }

        // 2. Try HuggingFace cache
        let hf_cache = resolve_hf_cache(&self.model_id);
        if let Some(path) = hf_cache {
            return Ok(path);
        }

        // 3. Try downloading via hf-hub
        info!("Attempting to download model {} from HuggingFace", self.model_id);
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| format!("Failed to create HF API: {}", e))?;
        let repo = api.model(self.model_id.clone());

        // Qwen3-ASR publishes the tokenizer as separate files rather than a
        // combined tokenizer.json. candle_tokenizer reconstructs the fast
        // tokenizer from these components at load time.
        for filename in &[
            "config.json",
            "vocab.json",
            "merges.txt",
            "tokenizer_config.json",
        ] {
            repo.get(filename)
                .map_err(|e| format!("Failed to download {}: {}", filename, e))?;
        }

        // Download safetensors (try model.safetensors.index.json first for sharded)
        let index_result = repo.get("model.safetensors.index.json");
        if let Ok(index_path) = index_result {
            let index_text = std::fs::read_to_string(&index_path)?;
            let index: serde_json::Value = serde_json::from_str(&index_text)?;
            if let Some(weight_map) = index.get("weight_map").and_then(|m| m.as_object()) {
                let files: std::collections::HashSet<String> = weight_map
                    .values()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
                for file in &files {
                    // S-3: Reject filenames with path traversal components
                    if file.contains("..") || file.contains('/') || file.contains('\\') {
                        return Err(format!("Suspicious filename in weight_map: {}", file).into());
                    }
                    repo.get(file)
                        .map_err(|e| format!("Failed to download {}: {}", file, e))?;
                }
            }
        } else {
            // Single safetensors file
            repo.get("model.safetensors")
                .map_err(|e| format!("Failed to download model.safetensors: {}", e))?;
        }

        // Resolve again from cache
        resolve_hf_cache(&self.model_id)
            .ok_or_else(|| format!("Model {} not found after download", self.model_id).into())
    }
}

impl AsrEngine for QwenEngine {
    fn prepare(
        &mut self,
        mut progress: Option<&mut dyn FnMut(InferenceProgress)>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.model.is_some() {
            return Ok(());
        }

        // Re-initialize CUDA device if it was released by unload() and GPU is requested.
        if self.use_gpu && self.device.is_cpu() {
            match try_gpu_device() {
                Ok(device) => {
                    self.device = device;
                    self.dtype = DType::F16;
                    self.active_gpu = true;
                }
                Err(e) => {
                    warn!("GPU not available: {}; using CPU", e);
                    self.device = Device::Cpu;
                    self.dtype = DType::F32;
                    self.active_gpu = false;
                }
            }
        }

        if self.use_gpu && self.require_gpu && !self.active_gpu {
            return Err("A GPU is required but none is available".into());
        }

        // CPU only: the 1.7B model decodes slower than realtime (3.5 tok/s on a
        // desktop CPU); the 0.6B model keeps up with speech (10 tok/s). Switch
        // unless the user pointed at an explicit local model directory.
        if !self.active_gpu && self.model_id.contains("1.7B") && self.model_path.is_none() {
            warn!("No GPU: using Qwen/Qwen3-ASR-0.6B instead of {} for speed", self.model_id);
            if let Some(cb) = progress.as_deref_mut() {
                cb(InferenceProgress {
                    stage: InferenceStage::LoadingModel,
                    message: "No GPU found: using the smaller 0.6B model on the CPU".to_string(),
                });
            }
            self.model_id = crate::config::MODEL_QWEN3_ASR_0_6B.to_string();
        }

        if let Some(cb) = progress.as_deref_mut() {
            cb(InferenceProgress {
                stage: InferenceStage::LoadingModel,
                message: "Loading model...".to_string(),
            });
        }

        if self.model_path.is_none() && resolve_hf_cache(&self.model_id).is_none() {
            if let Some(cb) = progress.as_deref_mut() {
                cb(InferenceProgress {
                    stage: InferenceStage::LoadingModel,
                    message: format!(
                        "Downloading {} ({} MB, first run only)...",
                        self.model_id.rsplit('/').next().unwrap_or(&self.model_id),
                        self.model_size_mb()
                    ),
                });
            }
        }
        let model_dir = self.resolve_model_dir()?;
        let started = Instant::now();

        info!("Loading model from {:?} on {:?}", model_dir, self.device);

        if let Some(cb) = progress.as_deref_mut() {
            cb(InferenceProgress {
                stage: InferenceStage::LoadingModel,
                message: "Loading model weights (this may take a moment)...".to_string(),
            });
        }

        let tokenizer_dir = resolve_tokenizer_dir(&model_dir, &self.model_id)?;
        let tokenizer = Qwen3ASRTokenizer::load(&tokenizer_dir)
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;
        // Q8_0 QMatMul runs on every Candle backend (CUDA, Metal, and CPU via
        // AVX2/NEON dot products), so the requested quantization applies everywhere.
        let quant = self.quantization;
        let cache_path = quant_cache_path(&self.model_id, quant);
        let mut model = match Qwen3ASRModel::load(&model_dir, &self.device, self.dtype, quant, cache_path.as_deref()) {
            Ok(m) => m,
            Err(e) if !self.device.is_cpu() && Self::is_cuda_oom(&e.to_string()) && self.model_id.contains("1.7B") => {
                // Small GPU: the 1.7B model does not fit. Drop to the 0.6B model.
                warn!("GPU out of memory loading {}; switching to Qwen/Qwen3-ASR-0.6B", self.model_id);
                if let Some(cb) = progress.as_deref_mut() {
                    cb(InferenceProgress {
                        stage: InferenceStage::LoadingModel,
                        message: "GPU memory too small for the 1.7B model; using the 0.6B model".to_string(),
                    });
                }
                self.unload();
                self.model_id = crate::config::MODEL_QWEN3_ASR_0_6B.to_string();
                self.model_path = None;
                return self.prepare(progress);
            }
            Err(e) => return Err(format!("Failed to load model: {}", e).into()),
        };
        // I-6: Wire max_new_tokens config through to the model
        model.max_new_tokens = self.max_new_tokens as usize;
        model.prompt_style = self.prompt_style;

        let elapsed = started.elapsed();
        info!("Model loaded in {:.2}s ({:?} weights)", elapsed.as_secs_f64(), model.quantization);

        self.tokenizer = Some(tokenizer);
        self.model = Some(model);
        Ok(())
    }

    fn transcribe_file_with_progress(
        &mut self,
        path: &Path,
        progress: Option<&mut dyn FnMut(InferenceProgress)>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let samples = load_audio_file(path)?;
        self.transcribe_with_progress(&samples, progress)
    }

    fn unload(&mut self) {
        if self.model.is_some() {
            info!("Unloading model from memory (after {} transcriptions)", self.transcription_count);
            self.model = None;
            self.tokenizer = None;
            // Release CUDA context and its memory pool by dropping the Device handle.
            // The device will be recreated on next prepare() call.
            self.device = Device::Cpu;
            self.active_gpu = false;
            self.transcription_count = 0;
            info!("GPU context released");
        }
    }

    fn active_gpu(&self) -> bool {
        self.active_gpu
    }

    fn model_label(&self) -> String {
        match self.quantization {
            Quantization::None => format!("candle ({})", self.model_id),
            q => format!("candle ({}, {:?})", self.model_id, q),
        }
    }

    fn model_size_mb(&self) -> u64 {
        if self.model_id.contains("0.6B") {
            1400
        } else if self.model_id.contains("1.7B") {
            3300
        } else {
            DEFAULT_MODEL_SIZE_MB
        }
    }
}

impl Default for QwenEngine {
    fn default() -> Self {
        Self::new(true).expect("Failed to create Qwen engine")
    }
}

impl Drop for QwenEngine {
    fn drop(&mut self) {
        self.unload();
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn normalize_language(lang: Option<&str>) -> Option<String> {
    let trimmed = lang.map(str::trim).filter(|value| !value.is_empty());
    match trimmed {
        Some("auto") => None,
        Some(value) => Some(value.to_string()),
        None => None,
    }
}

/// Load a `.f32` (raw little-endian 16 kHz mono) or `.wav` file as samples.
pub fn load_audio_file(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if path
        .extension()
        .map(|v| v.to_string_lossy().eq_ignore_ascii_case("f32"))
        .unwrap_or(false)
    {
        load_f32_file(path)
    } else {
        load_wav_file(path)
    }
}

fn load_f32_file(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err("Invalid f32 audio file length".into());
    }
    let mut samples = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        samples.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(samples)
}

fn load_wav_file(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // Simple WAV reader using raw parsing (no hound dependency)
    let data = fs::read(path)?;
    if data.len() < 44 {
        return Err("WAV file too small".into());
    }

    // Verify RIFF header
    if &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err("Not a valid WAV file".into());
    }

    // Parse fmt chunk to get actual format info
    let mut num_channels: u16 = 1;
    let mut bits_per_sample: u16 = 16;
    let mut audio_format: u16 = 1; // PCM

    // Find data chunk
    let mut pos = 12;
    // I-11: Use <= to include final chunk when data chunk is at end
    while pos + 8 <= data.len() {
        let chunk_id = &data[pos..pos + 4];
        // S-5: Guard against integer overflow on crafted chunk_size
        let raw_size = u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]);
        let chunk_size = (raw_size as usize).min(data.len().saturating_sub(pos + 8));

        if chunk_id == b"fmt " && pos + 26 <= data.len() {
            audio_format = u16::from_le_bytes([data[pos + 8], data[pos + 9]]);
            num_channels = u16::from_le_bytes([data[pos + 10], data[pos + 11]]);
            bits_per_sample = u16::from_le_bytes([data[pos + 22], data[pos + 23]]);
            debug!("WAV format: {}, channels: {}, bits: {}", audio_format, num_channels, bits_per_sample);
        }

        if chunk_id == b"data" {
            let audio_data = &data[pos + 8..pos + 8 + chunk_size];
            // I-7: Handle different sample formats and channel counts
            let raw_samples: Vec<f32> = match (audio_format, bits_per_sample) {
                (1, 16) => {
                    // 16-bit signed PCM
                    audio_data.chunks_exact(2)
                        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                        .collect()
                }
                (3, 32) | (1, 32) => {
                    // 32-bit float or 32-bit PCM
                    if audio_format == 3 {
                        // IEEE float
                        audio_data.chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect()
                    } else {
                        // 32-bit signed PCM
                        audio_data.chunks_exact(4)
                            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32 / 2147483648.0)
                            .collect()
                    }
                }
                (1, 24) => {
                    // 24-bit signed PCM
                    audio_data.chunks_exact(3)
                        .map(|c| {
                            let val = (c[0] as i32) | ((c[1] as i32) << 8) | ((c[2] as i32) << 16);
                            let val = if val & 0x800000 != 0 { val | !0xFFFFFF } else { val };
                            val as f32 / 8388608.0
                        })
                        .collect()
                }
                _ => {
                    return Err(format!(
                        "Unsupported WAV format: format={}, bits={}",
                        audio_format, bits_per_sample
                    ).into());
                }
            };

            // Convert to mono if multi-channel
            let channels = num_channels as usize;
            let samples = if channels > 1 {
                raw_samples.chunks(channels)
                    .map(|frame| {
                        let sum: f32 = frame.iter().sum();
                        sum / channels as f32
                    })
                    .collect()
            } else {
                raw_samples
            };

            return Ok(samples);
        }

        pos += 8 + chunk_size;
        // Pad to even
        if chunk_size % 2 != 0 {
            pos += 1;
        }
    }

    Err("No data chunk found in WAV file".into())
}

/// Create the GPU device: CUDA when built with the `cuda` feature, Metal with
/// the `metal` feature. CUDA is only attempted if the driver and cuBLAS
/// libraries can be loaded: with runtime loading, cudarc panics (instead of
/// erroring) when a library is missing, which would kill the worker process on
/// machines without an NVIDIA driver.
fn try_gpu_device() -> Result<Device, String> {
    #[cfg(feature = "cuda")]
    {
        use candle_core::cuda_backend::cudarc;
        // SAFETY: these only attempt to load shared libraries.
        let driver = unsafe { cudarc::driver::sys::is_culib_present() };
        if !driver {
            return Err("NVIDIA driver (nvcuda.dll) not found".to_string());
        }
        let cublas = unsafe { cudarc::cublas::sys::is_culib_present() };
        let cublaslt = unsafe { cudarc::cublaslt::sys::is_culib_present() };
        if !cublas || !cublaslt {
            return Err("cuBLAS runtime DLLs not found next to voclaude.exe".to_string());
        }
        Device::new_cuda(0).map_err(|e| e.to_string())
    }
    #[cfg(not(feature = "cuda"))]
    {
        // Metal when compiled with the `metal` feature; otherwise candle
        // reports that no GPU backend was built in.
        Device::new_metal(0).map_err(|e| format!("no GPU backend in this build: {}", e))
    }
}

/// Where the quantized projections of `model_id` are cached
/// (`%LOCALAPPDATA%\voclaude\Voclaude\cache\weights\<model>-<quant>.gguf`).
fn quant_cache_path(model_id: &str, quant: Quantization) -> Option<PathBuf> {
    if quant == Quantization::None {
        return None;
    }
    let dirs = crate::config::project_dirs()?;
    let name = model_id
        .rsplit('/')
        .next()
        .unwrap_or(model_id)
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '.' || c == '-' { c } else { '_' })
        .collect::<String>();
    Some(
        dirs.cache_dir()
            .join("weights")
            .join(format!("{}-{}.gguf", name, format!("{:?}", quant).to_ascii_lowercase())),
    )
}

const TOKENIZER_COMPONENTS: &[&str] = &["vocab.json", "merges.txt", "tokenizer_config.json"];

fn has_tokenizer_files(model_dir: &Path) -> bool {
    model_dir.join("tokenizer.json").is_file()
        || TOKENIZER_COMPONENTS
            .iter()
            .all(|filename| model_dir.join(filename).is_file())
}

/// Return a directory containing either a combined tokenizer.json or the
/// components published by Qwen3-ASR. Fill a partial Hugging Face cache when
/// possible instead of borrowing a tokenizer from an unrelated snapshot.
fn resolve_tokenizer_dir(model_dir: &Path, model_id: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if has_tokenizer_files(model_dir) {
        return Ok(model_dir.to_path_buf());
    }

    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| format!("Failed to create HF API for tokenizer files: {}", e))?;
    let repo = api.model(model_id.to_string());
    let mut downloaded_dir = None;
    for filename in TOKENIZER_COMPONENTS {
        if !model_dir.join(filename).is_file() {
            let path = repo
                .get(filename)
                .map_err(|e| format!("Failed to download {} for {}: {}", filename, model_id, e))?;
            downloaded_dir = path.parent().map(Path::to_path_buf);
        }
    }

    if has_tokenizer_files(model_dir) {
        return Ok(model_dir.to_path_buf());
    }
    if let Some(dir) = downloaded_dir.filter(|dir| has_tokenizer_files(dir)) {
        return Ok(dir);
    }

    Err(format!(
        "Tokenizer files not found for {} (looked in {})",
        model_id,
        model_dir.display()
    )
    .into())
}

/// Try to find the model in the HuggingFace cache directory.
fn resolve_hf_cache(model_id: &str) -> Option<PathBuf> {
    let cache_dir = dirs_for_hf_cache()?;
    let model_dir_name = format!("models--{}", model_id.replace('/', "--"));
    let model_cache = cache_dir.join(&model_dir_name).join("snapshots");

    if !model_cache.exists() {
        return None;
    }

    // I-9: Prefer refs/main symlink over mtime-based selection
    // The refs/main file contains the commit hash of the default snapshot
    let refs_main = cache_dir.join(&model_dir_name).join("refs").join("main");
    if let Ok(commit_hash) = std::fs::read_to_string(&refs_main) {
        let commit_hash = commit_hash.trim();
        if !commit_hash.is_empty() {
            let snapshot_dir = model_cache.join(commit_hash);
            if snapshot_dir.is_dir() && snapshot_dir.join("config.json").exists() {
                return Some(snapshot_dir);
            }
        }
    }

    // Fallback: find any valid snapshot (sorted by name for determinism)
    let mut snapshots: Vec<PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&model_cache) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() && path.join("config.json").exists() {
                snapshots.push(path);
            }
        }
    }
    snapshots.sort();
    snapshots.into_iter().last()
}

fn dirs_for_hf_cache() -> Option<PathBuf> {
    // I-12: Check HUGGINGFACE_HUB_CACHE first (enterprise deployments)
    if let Ok(cache) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        return Some(PathBuf::from(cache));
    }

    // Check HF_HOME next, then default
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        return Some(PathBuf::from(hf_home).join("hub"));
    }

    // Default: ~/.cache/huggingface/hub
    #[cfg(target_os = "windows")]
    {
        std::env::var("USERPROFILE")
            .ok()
            .map(|home| PathBuf::from(home).join(".cache").join("huggingface").join("hub"))
    }
    #[cfg(not(target_os = "windows"))]
    {
        std::env::var("HOME")
            .ok()
            .map(|home| PathBuf::from(home).join(".cache").join("huggingface").join("hub"))
    }
}

/// Split long audio into chunks at silence boundaries.
///
/// Finds low-energy (near-silence) points near each nominal 5-minute boundary
/// to avoid cutting mid-word. Returns slices into the original sample buffer.
fn adaptive_token_limit(sample_count: usize, configured_max: usize, adaptive: bool) -> usize {
    if !adaptive {
        return configured_max.max(1);
    }

    // Typical dictation is far below this, but the margin avoids truncating
    // fast speech and languages that tokenize more densely than English.
    let seconds = sample_count as f64 / SAMPLE_RATE as f64;
    let estimated = (seconds * 16.0).ceil() as usize + 128;
    configured_max.min(estimated.max(96)).max(1)
}

fn split_audio_at_silence(samples: &[f32], max_chunk_samples: usize) -> Vec<&[f32]> {
    let max_chunk_samples = max_chunk_samples.clamp(MIN_CHUNK_SAMPLES, MAX_CHUNK_SAMPLES);
    if samples.len() <= max_chunk_samples {
        return vec![samples];
    }

    /// Half-second analysis frame for RMS energy calculation.
    const ANALYSIS_FRAME: usize = SAMPLE_RATE / 2; // 8000 samples = 0.5s
    /// Search window: ±5 seconds around the nominal cut point.
    const SEARCH_HALF_WINDOW: usize = 5 * SAMPLE_RATE; // 80000 samples
    /// Step size for sliding the analysis frame (0.125 seconds).
    const STEP: usize = SAMPLE_RATE / 8; // 2000 samples

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < samples.len() {
        let remaining = samples.len() - start;
        if remaining <= max_chunk_samples {
            chunks.push(&samples[start..]);
            break;
        }

        let nominal_end = start + max_chunk_samples;
        let search_start = nominal_end.saturating_sub(SEARCH_HALF_WINDOW);
        let search_end = (nominal_end + SEARCH_HALF_WINDOW).min(samples.len());

        // Find the half-second window with minimum RMS energy
        let mut best_pos = nominal_end;
        let mut best_energy = f32::MAX;

        let mut pos = search_start;
        while pos + ANALYSIS_FRAME <= search_end {
            let window = &samples[pos..pos + ANALYSIS_FRAME];
            let energy: f32 = window.iter().map(|s| s * s).sum::<f32>() / ANALYSIS_FRAME as f32;
            if energy < best_energy {
                best_energy = energy;
                best_pos = pos + ANALYSIS_FRAME / 2; // cut at center of quiet window
            }
            pos += STEP;
        }

        // Clamp: never let a chunk exceed MAX_CHUNK_SAMPLES from start,
        // even if silence was found in the forward half of the search window.
        best_pos = best_pos.min(start + max_chunk_samples);

        // If the remaining tail after this cut is too short to produce a
        // meaningful transcription (< 1 second), absorb it into this chunk.
        const MIN_TAIL_SAMPLES: usize = SAMPLE_RATE; // 1 second
        let tail = samples.len() - best_pos;
        if tail < MIN_TAIL_SAMPLES {
            chunks.push(&samples[start..]);
            break;
        }

        chunks.push(&samples[start..best_pos]);
        start = best_pos;
    }

    chunks
}

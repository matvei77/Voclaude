//! Qwen speech-to-text inference engine (Candle native backend).
//!
//! Loads Qwen3-ASR-1.7B safetensors directly via candle-core and runs
//! inference entirely in Rust. No Python dependency.

use crate::config::Config;
use crate::inference::candle_backend::Qwen3ASRModel;
use crate::inference::candle_tokenizer::Qwen3ASRTokenizer;
use crate::inference::{InferenceProgress, InferenceStage};
use candle_core::{DType, Device};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing::{debug, info, warn};

const DEFAULT_MODEL_SIZE_MB: u64 = 3300;

pub struct QwenEngine {
    active_gpu: bool,
    use_gpu: bool,
    language: Option<String>,
    model_id: String,
    model_path: Option<String>,
    max_new_tokens: u32,
    require_gpu: bool,
    model: Option<Qwen3ASRModel>,
    tokenizer: Option<Qwen3ASRTokenizer>,
    device: Device,
    dtype: DType,
}

impl QwenEngine {
    pub fn new(use_gpu: bool) -> Result<Self, Box<dyn std::error::Error>> {
        let (device, dtype) = if use_gpu {
            match Device::new_cuda(0) {
                Ok(dev) => (dev, DType::F16),
                Err(e) => {
                    warn!("CUDA not available: {}, falling back to CPU", e);
                    (Device::Cpu, DType::F32)
                }
            }
        } else {
            (Device::Cpu, DType::F32)
        };

        let active_gpu = matches!(&device, Device::Cuda(_));

        Ok(Self {
            active_gpu,
            use_gpu,
            language: None,
            model_id: "Qwen/Qwen3-ASR-1.7B".to_string(),
            model_path: None,
            max_new_tokens: 2048,
            require_gpu: true,
            model: None,
            tokenizer: None,
            device,
            dtype,
        })
    }

    pub fn new_with_config(config: &Config) -> Result<Self, Box<dyn std::error::Error>> {
        let mut engine = Self::new(config.use_gpu)?;
        engine.language = normalize_language(config.language.as_deref());
        engine.model_id = config.qwen_model.clone();
        engine.model_path = config.qwen_model_path.clone();
        engine.max_new_tokens = config.qwen_max_new_tokens;
        engine.require_gpu = config.qwen_require_gpu;
        Ok(engine)
    }

    pub fn active_gpu(&self) -> bool {
        self.active_gpu
    }

    pub fn model_label(&self) -> String {
        format!("qwen-candle ({})", self.model_id)
    }

    pub fn model_size_mb(&self) -> u64 {
        if self.model_id.contains("0.6B") {
            1400
        } else if self.model_id.contains("1.7B") {
            3300
        } else {
            DEFAULT_MODEL_SIZE_MB
        }
    }

    pub fn unload(&mut self) {
        if self.model.is_some() {
            info!("Unloading Qwen model from memory");
            self.model = None;
            self.tokenizer = None;
        }
    }

    pub fn prepare(
        &mut self,
        mut progress: Option<&mut dyn FnMut(InferenceProgress)>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.model.is_some() {
            return Ok(());
        }

        if self.use_gpu && self.require_gpu && !self.active_gpu {
            return Err("CUDA is required but not available".into());
        }

        if let Some(cb) = progress.as_deref_mut() {
            cb(InferenceProgress {
                stage: InferenceStage::LoadingModel,
                message: "Loading Qwen model...".to_string(),
                percent: None,
            });
        }

        let model_dir = self.resolve_model_dir()?;
        let started = Instant::now();

        info!("Loading Qwen model from {:?} on {:?}", model_dir, self.device);

        if let Some(cb) = progress.as_deref_mut() {
            cb(InferenceProgress {
                stage: InferenceStage::LoadingModel,
                message: "Loading model weights (this may take a moment)...".to_string(),
                percent: None,
            });
        }

        let tokenizer = Qwen3ASRTokenizer::load(&model_dir)
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;
        let model = Qwen3ASRModel::load(&model_dir, &self.device, self.dtype)
            .map_err(|e| format!("Failed to load model: {}", e))?;

        let elapsed = started.elapsed();
        info!("Qwen model loaded in {:.2}s", elapsed.as_secs_f64());

        self.tokenizer = Some(tokenizer);
        self.model = Some(model);
        Ok(())
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
        self.prepare(progress.as_deref_mut().map(|p| p as &mut dyn FnMut(InferenceProgress)))?;

        if let Some(cb) = progress.as_deref_mut() {
            cb(InferenceProgress {
                stage: InferenceStage::Transcribing,
                message: "Transcribing with Qwen...".to_string(),
                percent: None,
            });
        }

        let started = Instant::now();
        let model = self.model.as_mut().ok_or("Model not loaded")?;
        let tokenizer = self.tokenizer.as_ref().ok_or("Tokenizer not loaded")?;

        let text = model
            .transcribe(samples, self.language.as_deref(), tokenizer)
            .map_err(|e| format!("Transcription failed: {}", e))?;

        let elapsed = started.elapsed();
        info!("Qwen transcription complete: infer={:.2}s", elapsed.as_secs_f64());

        Ok(text)
    }

    pub fn transcribe_file_with_progress(
        &mut self,
        path: &Path,
        progress: Option<&mut dyn FnMut(InferenceProgress)>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let samples = if path
            .extension()
            .map(|v| v.to_string_lossy().eq_ignore_ascii_case("f32"))
            .unwrap_or(false)
        {
            load_f32_file(path)?
        } else {
            // For WAV files, use hound to read them
            load_wav_file(path)?
        };

        self.transcribe_with_progress(&samples, progress)
    }

    fn resolve_model_dir(&self) -> Result<PathBuf, Box<dyn std::error::Error>> {
        // 1. Explicit config path
        if let Some(path) = &self.model_path {
            let p = PathBuf::from(path);
            if p.exists() && p.is_dir() {
                return Ok(p);
            }
            return Err(format!("Configured qwen_model_path not found: {}", path).into());
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

        // Download essential files
        for filename in &["config.json", "tokenizer.json"] {
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

    // Find data chunk
    let mut pos = 12;
    while pos + 8 < data.len() {
        let chunk_id = &data[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]) as usize;

        if chunk_id == b"fmt " {
            let format = u16::from_le_bytes([data[pos + 8], data[pos + 9]]);
            let bits_per_sample = if pos + 22 < data.len() {
                u16::from_le_bytes([data[pos + 22], data[pos + 23]])
            } else {
                16
            };
            debug!("WAV format: {}, bits: {}", format, bits_per_sample);
        }

        if chunk_id == b"data" {
            let audio_data = &data[pos + 8..pos + 8 + chunk_size.min(data.len() - pos - 8)];
            // Assume 16-bit PCM
            let mut samples = Vec::with_capacity(audio_data.len() / 2);
            for chunk in audio_data.chunks_exact(2) {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                samples.push(sample as f32 / 32768.0);
            }
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

/// Try to find the model in the HuggingFace cache directory.
fn resolve_hf_cache(model_id: &str) -> Option<PathBuf> {
    let cache_dir = dirs_for_hf_cache()?;
    let model_dir_name = format!("models--{}", model_id.replace('/', "--"));
    let model_cache = cache_dir.join(&model_dir_name).join("snapshots");

    if !model_cache.exists() {
        return None;
    }

    // Find the latest snapshot
    let mut latest: Option<PathBuf> = None;
    if let Ok(entries) = std::fs::read_dir(&model_cache) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                // Check if this snapshot has the required files
                if path.join("config.json").exists() {
                    latest = Some(path);
                }
            }
        }
    }

    latest
}

fn dirs_for_hf_cache() -> Option<PathBuf> {
    // Check HF_HOME first, then default
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

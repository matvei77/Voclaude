//! Whisper speech-to-text inference engine.
//!
//! Uses whisper-rs (whisper.cpp bindings) for fast, accurate transcription.

use crate::config::Config;
use crate::inference::{InferenceProgress, InferenceStage};
use std::fs;
use std::path::PathBuf;
use tracing::{debug, info, warn};
use whisper_rs::{WhisperContext, WhisperContextParameters, FullParams, SamplingStrategy};

/// Whisper model size
#[derive(Debug, Clone, Copy)]
pub enum WhisperModel {
    Medium,  // ~1.5GB, good balance of speed/quality
}

impl WhisperModel {
    fn filename(&self) -> &'static str {
        match self {
            WhisperModel::Medium => "ggml-medium.bin",
        }
    }

    fn url(&self) -> &'static str {
        match self {
            WhisperModel::Medium => "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
        }
    }

    fn size_mb(&self) -> u64 {
        match self {
            WhisperModel::Medium => 1533,
        }
    }
}

/// Whisper inference engine
pub struct WhisperEngine {
    context: Option<WhisperContext>,
    model: WhisperModel,
    is_loaded: bool,
    use_gpu: bool,
}

impl WhisperEngine {
    /// Create a new Whisper engine (lazy loading)
    pub fn new(use_gpu: bool) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            context: None,
            model: WhisperModel::Medium,
            is_loaded: false,
            use_gpu,
        })
    }

    /// Create a new Whisper engine with config
    pub fn new_with_config(config: &Config) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new(config.use_gpu)
    }

    /// Check if model is loaded
    pub fn is_loaded(&self) -> bool {
        self.is_loaded
    }

    /// Unload model to free memory
    pub fn unload(&mut self) {
        if self.is_loaded {
            info!("Unloading Whisper model");
            self.context = None;
            self.is_loaded = false;
        }
    }

    /// Get models directory
    fn models_dir() -> Option<PathBuf> {
        Config::models_dir()
    }

    /// Ensure model is downloaded
    fn ensure_model(
        &self,
        progress: &mut Option<&mut dyn FnMut(InferenceProgress)>,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let models_dir = Self::models_dir().ok_or("Could not determine models directory")?;
        fs::create_dir_all(&models_dir)?;

        let model_path = models_dir.join(self.model.filename());

        if model_path.exists() {
            info!("Model already downloaded: {}", model_path.display());
            return Ok(model_path);
        }

        let download_message = format!(
            "Downloading Whisper {} model (~{}MB)...",
            format!("{:?}", self.model).to_lowercase(),
            self.model.size_mb()
        );
        info!("{}", download_message);
        if let Some(cb) = progress.as_deref_mut() {
            cb(InferenceProgress {
                stage: InferenceStage::DownloadingModel,
                message: download_message,
                percent: None,
            });
        }

        Self::download_file(self.model.url(), &model_path, progress)?;

        info!("Model downloaded: {}", model_path.display());
        Ok(model_path)
    }

    /// Download a file with progress
    fn download_file(
        url: &str,
        dest: &PathBuf,
        progress: &mut Option<&mut dyn FnMut(InferenceProgress)>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use std::io::{Read, Write};

        info!("Downloading: {}", url);

        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(7200))
            .build()?;

        let mut response = client.get(url).send()?;

        if !response.status().is_success() {
            return Err(format!("Download failed: HTTP {}", response.status()).into());
        }

        let total_size = response.content_length();
        if let Some(size) = total_size {
            info!("Download size: {:.1} MB", size as f64 / 1024.0 / 1024.0);
        }

        let temp_path = dest.with_extension("tmp");
        let mut file = fs::File::create(&temp_path)?;

        let mut downloaded: u64 = 0;
        let mut last_progress = 0;
        let mut buffer = [0u8; 131072];

        loop {
            let bytes_read = response.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            file.write_all(&buffer[..bytes_read])?;
            downloaded += bytes_read as u64;

            if let Some(total) = total_size {
                let progress_pct = (downloaded * 100 / total) as u32;
                if progress_pct >= last_progress + 5 {
                    info!("Download progress: {}% ({:.1} MB / {:.1} MB)",
                          progress_pct,
                          downloaded as f64 / 1024.0 / 1024.0,
                          total as f64 / 1024.0 / 1024.0);
                    last_progress = progress_pct;
                    if let Some(cb) = progress.as_deref_mut() {
                        cb(InferenceProgress {
                            stage: InferenceStage::DownloadingModel,
                            message: format!(
                                "Downloading model... {}% ({:.1}/{:.1} MB)",
                                progress_pct,
                                downloaded as f64 / 1024.0 / 1024.0,
                                total as f64 / 1024.0 / 1024.0
                            ),
                            percent: Some(progress_pct as u8),
                        });
                    }
                }
            }
        }

        file.flush()?;
        drop(file);
        fs::rename(&temp_path, dest)?;

        info!("Download complete: {}", dest.display());
        Ok(())
    }

    /// Ensure model is loaded
    fn ensure_loaded(
        &mut self,
        progress: &mut Option<&mut dyn FnMut(InferenceProgress)>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.is_loaded {
            return Ok(());
        }

        info!("Loading Whisper model...");

        let model_path = self.ensure_model(progress)?;
        if let Some(cb) = progress.as_deref_mut() {
            cb(InferenceProgress {
                stage: InferenceStage::LoadingModel,
                message: "Loading Whisper model...".to_string(),
                percent: None,
            });
        }

        let mut prefer_gpu = self.use_gpu;
        if prefer_gpu && !cfg!(feature = "cuda") {
            info!("GPU requested but CUDA feature is disabled; falling back to CPU");
            prefer_gpu = false;
        }

        let mut params = WhisperContextParameters::default();
        params.use_gpu(prefer_gpu);

        let ctx = match WhisperContext::new_with_params(
            model_path.to_str().ok_or("Invalid model path")?,
            params,
        ) {
            Ok(ctx) => ctx,
            Err(e) if prefer_gpu => {
                warn!("GPU init failed ({}); retrying on CPU", e);
                let mut cpu_params = WhisperContextParameters::default();
                cpu_params.use_gpu(false);
                WhisperContext::new_with_params(
                    model_path.to_str().ok_or("Invalid model path")?,
                    cpu_params,
                ).map_err(|e| format!("Failed to load Whisper model on CPU: {}", e))?
            }
            Err(e) => return Err(format!("Failed to load Whisper model: {}", e).into()),
        };

        self.context = Some(ctx);
        self.is_loaded = true;

        info!("Whisper model loaded!");
        Ok(())
    }

    /// Transcribe audio samples
    /// Input: f32 samples at 16kHz mono
    /// Output: transcribed text
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<String, Box<dyn std::error::Error>> {
        self.transcribe_with_progress(samples, None)
    }

    pub fn transcribe_with_progress(
        &mut self,
        samples: &[f32],
        mut progress: Option<&mut dyn FnMut(InferenceProgress)>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.ensure_loaded(&mut progress)?;

        let ctx = self.context.as_mut().ok_or("Whisper context not loaded")?;

        debug!("Transcribing {} samples ({:.2}s)",
               samples.len(),
               samples.len() as f32 / 16000.0);

        // Create state for this transcription
        let mut state = ctx.create_state()
            .map_err(|e| format!("Failed to create Whisper state: {}", e))?;

        // Configure transcription parameters
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

        // Set language to English for faster processing
        params.set_language(Some("en"));

        // Disable translation (we want transcription)
        params.set_translate(false);

        // Single segment mode for voice input
        params.set_single_segment(false);

        // Print progress
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        // Suppress non-speech tokens
        params.set_suppress_blank(true);
        params.set_suppress_nst(true);

        // Run inference
        if let Some(cb) = progress.as_deref_mut() {
            cb(InferenceProgress {
                stage: InferenceStage::Transcribing,
                message: "Transcribing audio...".to_string(),
                percent: None,
            });
        }
        let start = std::time::Instant::now();
        state.full(params, samples)
            .map_err(|e| format!("Whisper inference failed: {}", e))?;
        let elapsed = start.elapsed();

        debug!("Inference took {:.2}s", elapsed.as_secs_f32());

        // Collect all segments
        let num_segments = state.full_n_segments()
            .map_err(|e| format!("Failed to get segments: {}", e))?;

        let mut text = String::new();
        for i in 0..num_segments {
            if let Ok(segment) = state.full_get_segment_text(i) {
                text.push_str(&segment);
            }
        }

        let text = text.trim().to_string();
        debug!("Transcription: {}", text);

        Ok(text)
    }
}

impl Default for WhisperEngine {
    fn default() -> Self {
        Self::new(true).expect("Failed to create WhisperEngine")
    }
}

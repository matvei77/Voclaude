//! Speech-to-text inference backend.

pub mod candle_audio;
pub mod candle_backend;
pub mod candle_tokenizer;
mod parakeet;
mod qwen;
pub mod whisper_engine;

pub use parakeet::ParakeetEngine;
pub use qwen::{load_audio_file, QwenEngine};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceStage {
    LoadingModel,
    Transcribing,
}

#[derive(Debug, Clone)]
pub struct InferenceProgress {
    pub stage: InferenceStage,
    pub message: String,
}

/// Backend-agnostic speech-to-text engine interface.
pub trait AsrEngine {
    /// Ensure the model is loaded and ready. No-op if already prepared.
    fn prepare(
        &mut self,
        progress: Option<&mut dyn FnMut(InferenceProgress)>,
    ) -> Result<(), Box<dyn std::error::Error>>;

    /// Transcribe an audio file on disk, reporting progress.
    fn transcribe_file_with_progress(
        &mut self,
        path: &std::path::Path,
        progress: Option<&mut dyn FnMut(InferenceProgress)>,
    ) -> Result<String, Box<dyn std::error::Error>>;

    /// Unload model weights from memory.
    fn unload(&mut self);

    /// Whether the engine is currently using a GPU.
    fn active_gpu(&self) -> bool;

    /// Human-readable label for the loaded model.
    fn model_label(&self) -> String;

    /// Approximate model size in megabytes.
    fn model_size_mb(&self) -> u64;
}

/// The engine selected by `config.engine`.
pub enum Engine {
    Parakeet(ParakeetEngine),
    Qwen(QwenEngine),
}

impl Engine {
    pub fn from_config(config: &crate::config::Config) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(if config.uses_parakeet() {
            Engine::Parakeet(ParakeetEngine::new_with_config(config)?)
        } else {
            Engine::Qwen(QwenEngine::new_with_config(config)?)
        })
    }

    pub fn is_loaded(&self) -> bool {
        match self {
            Engine::Parakeet(e) => e.is_loaded(),
            Engine::Qwen(e) => e.is_loaded(),
        }
    }

    /// Transcribe one live segment. `context` (the previous segment's text) is
    /// only used by Qwen; Parakeet has no prompt.
    pub fn transcribe_segment(
        &mut self,
        samples: &[f32],
        context: Option<&str>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        match self {
            Engine::Parakeet(e) => e.transcribe_segment(samples),
            Engine::Qwen(e) => e.transcribe_segment(samples, context).map(|(text, _stats)| text),
        }
    }

    fn inner(&mut self) -> &mut dyn AsrEngine {
        match self {
            Engine::Parakeet(e) => e,
            Engine::Qwen(e) => e,
        }
    }

    fn inner_ref(&self) -> &dyn AsrEngine {
        match self {
            Engine::Parakeet(e) => e,
            Engine::Qwen(e) => e,
        }
    }
}

impl AsrEngine for Engine {
    fn prepare(
        &mut self,
        progress: Option<&mut dyn FnMut(InferenceProgress)>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.inner().prepare(progress)
    }

    fn transcribe_file_with_progress(
        &mut self,
        path: &std::path::Path,
        progress: Option<&mut dyn FnMut(InferenceProgress)>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.inner().transcribe_file_with_progress(path, progress)
    }

    fn unload(&mut self) {
        self.inner().unload()
    }

    fn active_gpu(&self) -> bool {
        self.inner_ref().active_gpu()
    }

    fn model_label(&self) -> String {
        self.inner_ref().model_label()
    }

    fn model_size_mb(&self) -> u64 {
        self.inner_ref().model_size_mb()
    }
}

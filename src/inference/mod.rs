//! Speech-to-text inference backend.

pub mod candle_audio;
pub mod candle_backend;
pub mod candle_tokenizer;
mod qwen;

pub use qwen::QwenEngine;

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

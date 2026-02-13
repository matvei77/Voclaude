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
    pub percent: Option<u8>,
}

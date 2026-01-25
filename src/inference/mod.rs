//! Whisper speech-to-text inference.

mod whisper;

pub use whisper::WhisperEngine;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceStage {
    DownloadingModel,
    LoadingModel,
    Transcribing,
}

#[derive(Debug, Clone)]
pub struct InferenceProgress {
    pub stage: InferenceStage,
    pub message: String,
    pub percent: Option<u8>,
}

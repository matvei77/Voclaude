use std::path::Path;

use anyhow::Result;
use tracing::info;

use crate::audio::prepare_audio_input;
use crate::backend::{InferenceBackend, TranscriptionResult};
use crate::config::AppConfig;

pub fn run_transcription(
    backend: &dyn InferenceBackend,
    cfg: &AppConfig,
    input: &Path,
) -> Result<TranscriptionResult> {
    let prepared = prepare_audio_input(input)?;

    info!(
        "transcribing input={} backend={} chunk={:.1}s overlap={:.1}s",
        prepared.path.display(),
        backend.name(),
        cfg.chunking.seconds,
        cfg.chunking.overlap_seconds
    );

    backend.transcribe_path(
        &prepared.path,
        cfg.chunking.seconds,
        cfg.chunking.overlap_seconds,
    )
}

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use hound::{SampleFormat, WavSpec, WavWriter};
use tempfile::TempDir;

pub struct PreparedAudio {
    pub path: PathBuf,
    _tempdir: Option<TempDir>,
}

pub fn prepare_audio_input(input: &Path) -> Result<PreparedAudio> {
    if !input.exists() {
        bail!("input file not found: {}", input.display());
    }

    let ext = input
        .extension()
        .map(|v| v.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();

    if ext != "f32" {
        return Ok(PreparedAudio {
            path: input.to_path_buf(),
            _tempdir: None,
        });
    }

    let bytes = fs::read(input)
        .with_context(|| format!("failed to read {}", input.display()))?;
    if bytes.len() % 4 != 0 {
        bail!("invalid .f32 file: byte length is not divisible by 4");
    }

    let tempdir = tempfile::tempdir().context("failed to create temp dir")?;
    let wav_path = tempdir.path().join("input_16k_mono.wav");

    let spec = WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let mut writer = WavWriter::create(&wav_path, spec)
        .with_context(|| format!("failed to create {}", wav_path.display()))?;

    for chunk in bytes.chunks_exact(4) {
        let sample = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        writer
            .write_sample(sample)
            .with_context(|| "failed writing converted .f32 sample")?;
    }

    writer.finalize().context("failed to finalize wav writer")?;

    Ok(PreparedAudio {
        path: wav_path,
        _tempdir: Some(tempdir),
    })
}

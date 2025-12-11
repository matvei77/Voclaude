//! Model downloading and management for Parakeet TDT ONNX.
//!
//! Uses sherpa-onnx's well-tested ONNX export with encoder + decoder + joiner format.

use crate::config::Config;

use bzip2::read::BzDecoder;
use std::fs::{self, File};
use std::io::{Read, Write, BufReader};
use std::path::PathBuf;
use tar::Archive;
use tracing::info;

/// sherpa-onnx Parakeet TDT 0.6B v2 int8 quantized (encoder + decoder + joiner format)
const MODEL_TAR_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2";
const MODEL_DIR_NAME: &str = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8";

/// Model files (inside the extracted directory) - int8 quantized
const ENCODER_FILE: &str = "encoder.int8.onnx";
const DECODER_FILE: &str = "decoder.int8.onnx";
const JOINER_FILE: &str = "joiner.int8.onnx";
const TOKENS_FILE: &str = "tokens.txt";

pub struct ModelManager;

impl ModelManager {
    /// Get the models directory
    pub fn models_dir() -> Option<PathBuf> {
        Config::models_dir()
    }

    /// Ensure all models are downloaded and extracted
    fn ensure_models() -> Result<PathBuf, Box<dyn std::error::Error>> {
        let models_dir = Self::models_dir().ok_or("Could not determine models directory")?;
        fs::create_dir_all(&models_dir)?;

        let model_dir = models_dir.join(MODEL_DIR_NAME);
        let encoder_path = model_dir.join(ENCODER_FILE);

        // Check if already extracted
        if encoder_path.exists() {
            info!("Model already downloaded: {}", model_dir.display());
            return Ok(model_dir);
        }

        // Download and extract tar.bz2
        info!("Downloading Parakeet TDT 0.6B v2 model (~2.4GB)...");
        let tar_path = models_dir.join("model.tar.bz2");
        Self::download_file(MODEL_TAR_URL, &tar_path)?;

        info!("Extracting model files...");
        Self::extract_tar_bz2(&tar_path, &models_dir)?;

        // Clean up tar file
        fs::remove_file(&tar_path)?;

        info!("All model files ready: {}", model_dir.display());
        Ok(model_dir)
    }

    /// Download a file with progress
    fn download_file(url: &str, dest: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        info!("Downloading: {}", url);

        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(7200)) // 2 hours for large file
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
                let progress = (downloaded * 100 / total) as u32;
                if progress >= last_progress + 5 {
                    info!("Download progress: {}% ({:.1} MB / {:.1} MB)",
                          progress,
                          downloaded as f64 / 1024.0 / 1024.0,
                          total as f64 / 1024.0 / 1024.0);
                    last_progress = progress;
                }
            }
        }

        file.flush()?;
        drop(file);
        fs::rename(&temp_path, dest)?;

        info!("Download complete: {}", dest.display());
        Ok(())
    }

    /// Extract tar.bz2 archive
    fn extract_tar_bz2(tar_path: &PathBuf, dest_dir: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::open(tar_path)?;
        let reader = BufReader::new(file);
        let decompressor = BzDecoder::new(reader);
        let mut archive = Archive::new(decompressor);

        archive.unpack(dest_dir)?;

        info!("Extraction complete");
        Ok(())
    }

    /// Get path to the encoder ONNX model
    pub fn ensure_encoder() -> Result<PathBuf, Box<dyn std::error::Error>> {
        let model_dir = Self::ensure_models()?;
        let path = model_dir.join(ENCODER_FILE);
        if !path.exists() {
            return Err(format!("Encoder not found: {}", path.display()).into());
        }
        info!("Encoder: {}", path.display());
        Ok(path)
    }

    /// Get path to the decoder ONNX model
    pub fn ensure_decoder() -> Result<PathBuf, Box<dyn std::error::Error>> {
        let model_dir = Self::ensure_models()?;
        let path = model_dir.join(DECODER_FILE);
        if !path.exists() {
            return Err(format!("Decoder not found: {}", path.display()).into());
        }
        info!("Decoder: {}", path.display());
        Ok(path)
    }

    /// Get path to the joiner ONNX model
    pub fn ensure_joiner() -> Result<PathBuf, Box<dyn std::error::Error>> {
        let model_dir = Self::ensure_models()?;
        let path = model_dir.join(JOINER_FILE);
        if !path.exists() {
            return Err(format!("Joiner not found: {}", path.display()).into());
        }
        info!("Joiner: {}", path.display());
        Ok(path)
    }

    /// Get path to the tokens file
    pub fn ensure_tokens() -> Result<PathBuf, Box<dyn std::error::Error>> {
        let model_dir = Self::ensure_models()?;
        let path = model_dir.join(TOKENS_FILE);
        if !path.exists() {
            return Err(format!("Tokens not found: {}", path.display()).into());
        }
        info!("Tokens: {}", path.display());
        Ok(path)
    }
}

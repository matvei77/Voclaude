//! Voclaude - Voice input anywhere
//!
//! Local-first, GPU-accelerated speech-to-text that runs in your system tray.

mod app;
mod audio;
mod config;
mod hotkey;
mod inference;
mod tray;

use app::App;
use config::Config;
use inference::WhisperEngine;
use tracing::{info, error, Level};
use tracing_subscriber::FmtSubscriber;

fn main() {
    // Initialize logging
    let _subscriber = FmtSubscriber::builder()
        .with_max_level(Level::DEBUG)
        .with_target(false)
        .compact()
        .init();

    // Check for --test argument
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "--test" {
        let wav_path = if args.len() > 2 {
            &args[2]
        } else {
            eprintln!("Usage: voclaude --test <path/to/audio.wav>");
            eprintln!("No WAV file path provided");
            std::process::exit(1);
        };

        if let Err(e) = run_test(wav_path) {
            error!("Test failed: {}", e);
            std::process::exit(1);
        }
        return;
    }

    info!("Voclaude starting...");

    // Load config
    let config = match Config::load() {
        Ok(c) => {
            info!("Config loaded successfully");
            info!("  hotkey: {}", c.hotkey);
            info!("  idle_unload_seconds: {}", c.idle_unload_seconds);
            c
        }
        Err(e) => {
            error!("Failed to load config: {}", e);
            Config::default()
        }
    };

    // Run the app
    if let Err(e) = App::run(config) {
        error!("Application error: {}", e);
        std::process::exit(1);
    }
}

/// Test mode: load WAV file and transcribe it
fn run_test(wav_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    info!("=== TEST MODE ===");
    info!("Loading WAV file: {}", wav_path);

    // Read WAV file
    let mut reader = hound::WavReader::open(wav_path)?;
    let spec = reader.spec();
    info!("WAV format: {} Hz, {} channels, {} bits",
          spec.sample_rate, spec.channels, spec.bits_per_sample);

    // Convert samples to f32 mono at 16kHz
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => {
            reader.samples::<f32>()
                .filter_map(|s| s.ok())
                .collect()
        }
    };

    // Convert to mono if stereo
    let mono_samples: Vec<f32> = if spec.channels == 2 {
        samples.chunks(2)
            .map(|chunk| (chunk[0] + chunk.get(1).unwrap_or(&0.0)) / 2.0)
            .collect()
    } else {
        samples
    };

    // Resample to 16kHz if needed
    let samples_16k = if spec.sample_rate != 16000 {
        info!("Resampling from {} Hz to 16000 Hz...", spec.sample_rate);
        resample(&mono_samples, spec.sample_rate, 16000)
    } else {
        mono_samples
    };

    // No limit - process full file

    info!("Audio: {:.2}s ({} samples at 16kHz)",
          samples_16k.len() as f32 / 16000.0, samples_16k.len());

    // Create engine and transcribe
    info!("Creating Whisper engine...");
    let mut engine = WhisperEngine::new()?;

    info!("Transcribing...");
    let start = std::time::Instant::now();
    let text = engine.transcribe(&samples_16k)?;
    let elapsed = start.elapsed();

    info!("=== RESULT ===");
    info!("Time: {:.2}s", elapsed.as_secs_f32());
    info!("Text: {}", text);
    println!("\n>>> TRANSCRIPTION:\n{}\n", text);

    Ok(())
}

/// Simple linear resampling
fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    let ratio = from_rate as f64 / to_rate as f64;
    let new_len = (samples.len() as f64 / ratio) as usize;

    (0..new_len)
        .map(|i| {
            let src_idx = i as f64 * ratio;
            let idx = src_idx as usize;
            let frac = src_idx - idx as f64;

            let s0 = samples.get(idx).copied().unwrap_or(0.0);
            let s1 = samples.get(idx + 1).copied().unwrap_or(s0);

            s0 * (1.0 - frac as f32) + s1 * frac as f32
        })
        .collect()
}

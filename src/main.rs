//! Voclaude - Voice input anywhere
//!
//! Local-first, GPU-accelerated speech-to-text that runs in your system tray.

mod app;
mod audio;
mod config;
mod history;
mod hotkey;
mod inference;
mod tray;
mod ui;

use app::App;
use audio::resample_linear;
use config::Config;
use inference::WhisperEngine;
use tracing::{info, error, Level};

fn main() {
    let log_buffer = ui::LogBuffer::new(400);
    let log_writer = log_buffer.make_writer();

    tracing_subscriber::fmt()
        .with_max_level(Level::DEBUG)
        .with_target(false)
        .compact()
        .with_writer(log_writer)
        .init();

    // Check for --test argument
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "--test" {
        let wav_path = if args.len() > 2 {
            &args[2]
        } else {
            r"C:\Users\matvei\Documents\Sound Recordings\pedro_email.wav"
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
            info!("  history_hotkey: {}", c.history_hotkey);
            info!("  idle_unload_seconds: {}", c.idle_unload_seconds);
            info!("  history_max_entries: {}", c.history_max_entries);
            info!("  use_gpu: {}", c.use_gpu);
            c
        }
        Err(e) => {
            error!("Failed to load config: {}", e);
            Config::default()
        }
    };

    // Run the app
    if let Err(e) = App::run(config, log_buffer) {
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
        resample_linear(&mono_samples, spec.sample_rate, 16000)
    } else {
        mono_samples
    };

    // No limit - process full file

    info!("Audio: {:.2}s ({} samples at 16kHz)",
          samples_16k.len() as f32 / 16000.0, samples_16k.len());

    // Create engine and transcribe
    info!("Creating Whisper engine...");
    let test_config = Config::load().unwrap_or_default();
    let mut engine = WhisperEngine::new_with_config(&test_config)?;

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

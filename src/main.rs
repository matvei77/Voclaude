#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]
//! Voclaude - Voice input anywhere
//!
//! Local-first, GPU-accelerated speech-to-text that runs in your system tray.

mod app;
mod audio;
mod config;
mod history;
mod hotkey;
mod inference;
mod session;
mod tray;
mod ui;

use app::App;
use config::Config;
use inference::QwenEngine;
use tracing::{info, error, Level};

fn main() {
    // Check for --test argument early (before tracing init)
    let args: Vec<String> = std::env::args().collect();
    let test_mode = args.len() > 1 && args[1] == "--test";

    if test_mode {
        // Attach to parent console so test output is visible on Windows GUI subsystem
        #[cfg(target_os = "windows")]
        unsafe {
            use windows_sys::Win32::System::Console::{AttachConsole, ATTACH_PARENT_PROCESS, AllocConsole};
            if AttachConsole(ATTACH_PARENT_PROCESS) == 0 {
                AllocConsole();
            }
        }

        // Test mode: log to a file for reliable output capture
        let log_path = std::env::current_dir().unwrap_or_default().join("test_output.log");
        let log_file = std::fs::File::create(&log_path).expect("Failed to create log file");
        tracing_subscriber::fmt()
            .with_max_level(Level::DEBUG)
            .with_target(false)
            .compact()
            .with_writer(std::sync::Mutex::new(log_file))
            .init();

        if args.len() < 3 {
            eprintln!("Usage: voclaude-qwen-runtime --test <path-to-audio.wav>");
            std::process::exit(1);
        }
        let wav_path = &args[2];

        if let Err(e) = run_test(wav_path) {
            error!("Test failed: {}", e);
            show_fatal_error_dialog(&format!("Test failed: {}", e));
            std::process::exit(1);
        }
        return;
    }

    // Normal mode: log to UI buffer
    let log_buffer = ui::LogBuffer::new(400);
    let log_writer = log_buffer.make_writer();

    tracing_subscriber::fmt()
        .with_max_level(Level::DEBUG)
        .with_target(false)
        .compact()
        .with_writer(log_writer)
        .init();

    info!("Voclaude Qwen runtime starting...");

    // Load config
    let config = match Config::load() {
        Ok(c) => {
            info!("Config loaded successfully");
            if let Some(path) = Config::config_path() {
                info!("  config_path: {}", path.display());
            }
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
        show_fatal_error_dialog(&format!("Application error: {}", e));
        std::process::exit(1);
    }
}

/// Test mode: load audio file and transcribe it
fn run_test(audio_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    info!("=== TEST MODE ===");
    info!("Loading audio file: {}", audio_path);

    let path = std::path::Path::new(audio_path);

    // Create engine and transcribe using file-based path
    info!("Creating Qwen engine...");
    let mut test_config = Config::load().unwrap_or_default();
    test_config.qwen_require_gpu = false; // test mode always allows CPU fallback
    let mut engine = QwenEngine::new_with_config(&test_config)?;

    info!("Transcribing...");
    let start = std::time::Instant::now();
    let text = engine.transcribe_file_with_progress(path, None)?;
    let elapsed = start.elapsed();

    info!("=== RESULT ===");
    info!("Time: {:.2}s", elapsed.as_secs_f32());
    info!("Text: {}", text);
    // Write result to file for reliable capture (Windows GUI subsystem may not have console)
    let result_path = std::path::Path::new(audio_path).with_extension("result.txt");
    let _ = std::fs::write(&result_path, format!("Time: {:.2}s\n{}", elapsed.as_secs_f32(), text));
    println!("\n>>> TRANSCRIPTION:\n{}\n", text);

    Ok(())
}

#[cfg(target_os = "windows")]
fn show_fatal_error_dialog(message: &str) {
    use std::ffi::OsStr;
    use std::iter::once;
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::UI::WindowsAndMessaging::{MessageBoxW, MB_ICONERROR, MB_OK};

    let title: Vec<u16> = OsStr::new("Voclaude Qwen Runtime Error")
        .encode_wide()
        .chain(once(0))
        .collect();
    let body: Vec<u16> = OsStr::new(message)
        .encode_wide()
        .chain(once(0))
        .collect();

    unsafe {
        MessageBoxW(
            std::ptr::null_mut(),
            body.as_ptr(),
            title.as_ptr(),
            MB_OK | MB_ICONERROR,
        );
    }
}

#[cfg(not(target_os = "windows"))]
fn show_fatal_error_dialog(_message: &str) {}

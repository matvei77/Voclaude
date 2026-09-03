#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]
//! Voclaude - Voice input anywhere
//!
//! Local-first, GPU-accelerated speech-to-text that runs in your system tray.

mod app;
mod audio;
mod bench;
mod config;
mod history;
mod hotkey;
mod inference;
mod session;
mod tray;
mod ui;
mod worker;

use app::App;
use config::Config;
use inference::{AsrEngine, QwenEngine};
use tracing::{info, error, Level};

/// Writer that sends output to both the UI log buffer and a file appender.
struct CombinedWriter {
    ui: ui::LogBufferMakeWriter,
    file: tracing_appender::non_blocking::NonBlocking,
}

impl std::io::Write for CombinedWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        use tracing_subscriber::fmt::MakeWriter;
        // Write to file first (non-blocking)
        let _ = std::io::Write::write(&mut self.file, buf);
        // Write to UI buffer (includes stdout)
        let mut ui_writer = self.ui.make_writer();
        ui_writer.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        // O-6: Propagate file flush errors instead of discarding
        std::io::Write::flush(&mut self.file)?;
        Ok(())
    }
}

fn main() {
    // Defer cuBLAS/cuDNN kernel loading until first use (~200-400 MB saved).
    // Must be set before any CUDA initialization occurs.
    // SAFETY: Called at the very start of main() before any threads are spawned.
    if std::env::var_os("CUDA_MODULE_LOADING").is_none() {
        unsafe { std::env::set_var("CUDA_MODULE_LOADING", "LAZY") };
    }

    let args: Vec<String> = std::env::args().collect();

    // --version: print version and exit
    if args.iter().any(|a| a == "--version") {
        println!("voclaude {} ({})", env!("CARGO_PKG_VERSION"), option_env!("VOCLAUDE_GIT_HASH").unwrap_or("unknown"));
        return;
    }

    // Inference child process (spawned by the app itself).
    if args.len() > 1 && args[1] == "--worker" {
        tracing_subscriber::fmt()
            .with_max_level(Level::INFO)
            .with_target(false)
            .with_ansi(false)
            .without_time()
            .compact()
            .with_writer(std::io::stderr)
            .init();
        if let Err(e) = worker::run_child() {
            error!("worker failed: {}", e);
            std::process::exit(1);
        }
        return;
    }

    if args.iter().any(|a| a == "--list-models") {
        attach_console_for_cli();
        print_model_list();
        return;
    }

    if args.len() > 1 && args[1] == "--bench" {
        attach_console_for_cli();
        tracing_subscriber::fmt()
            .with_max_level(Level::INFO)
            .with_target(false)
            .compact()
            .with_writer(std::io::stderr)
            .init();
        let mut bench_config = Config::load().unwrap_or_default();
        bench_config.require_gpu = false;
        if let Err(err) = apply_cli_overrides(&mut bench_config, &args) {
            eprintln!("Invalid CLI option: {}", err);
            std::process::exit(1);
        }
        if let Err(e) = bench::run(&bench_config, &args[2..]) {
            eprintln!("Bench failed: {}", e);
            std::process::exit(1);
        }
        return;
    }

    let test_mode = args.len() > 1 && args[1] == "--test";
    let validate_mode = args.iter().any(|a| a == "--validate");

    if test_mode {
        // Attach to parent console so test output is visible on Windows GUI subsystem
        attach_console_for_cli();

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
            eprintln!("Usage: voclaude --test <path-to-audio.wav>");
            std::process::exit(1);
        }
        let wav_path = &args[2];

        if let Err(e) = run_test(wav_path, &args) {
            error!("Test failed: {}", e);
            show_fatal_error_dialog(&format!("Test failed: {}", e));
            std::process::exit(1);
        }
        return;
    }

    // Normal mode: log to UI buffer + file
    let log_buffer = ui::LogBuffer::new(400);
    let log_writer = log_buffer.make_writer();

    // File logging: %LOCALAPPDATA%/Voclaude/logs/ with daily rotation, keep 5 files
    let _file_guard = if let Some(dirs) = config::project_dirs() {
        let log_dir = dirs.data_dir().join("logs");
        let _ = std::fs::create_dir_all(&log_dir);
        // Clean up old log files (keep last 5)
        if let Ok(entries) = std::fs::read_dir(&log_dir) {
            let mut log_files: Vec<_> = entries
                .filter_map(|e| e.ok())
                .filter(|e| e.file_name().to_string_lossy().starts_with("voclaude.log"))
                .collect();
            log_files.sort_by_key(|e| std::cmp::Reverse(e.metadata().and_then(|m| m.modified()).unwrap_or(std::time::SystemTime::UNIX_EPOCH)));
            for old_file in log_files.into_iter().skip(5) {
                let _ = std::fs::remove_file(old_file.path());
            }
        }
        let file_appender = tracing_appender::rolling::daily(&log_dir, "voclaude.log");
        let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

        // Dual writer: UI buffer + file
        let combined = CombinedWriter { ui: log_writer, file: non_blocking };
        tracing_subscriber::fmt()
            .with_max_level(Level::DEBUG)
            .with_target(false)
            .compact()
            .with_writer(std::sync::Mutex::new(combined))
            .init();
        Some(guard)
    } else {
        tracing_subscriber::fmt()
            .with_max_level(Level::DEBUG)
            .with_target(false)
            .compact()
            .with_writer(log_writer)
            .init();
        None
    };

    Config::migrate_from_legacy();
    info!("Voclaude {} starting...", env!("CARGO_PKG_VERSION"));

    // Load config
    let mut config = match Config::load() {
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

    // CLI override: --model-dir <path> sets model_path for IT pre-staging
    // O-5: Validate --model-dir has a value and the path exists
    if let Err(err) = apply_cli_overrides(&mut config, &args) {
        error!("Invalid CLI option: {}", err);
        show_fatal_error_dialog(&err);
        std::process::exit(1);
    }

    // --validate: check GPU, model, audio device, config — then exit
    if validate_mode {
        attach_console_for_cli();
        let validate_result = (|| -> Result<(), Box<dyn std::error::Error>> {
            println!("Voclaude {} — Validation Mode", env!("CARGO_PKG_VERSION"));
            println!("Config: OK");
            println!("  model: {}", config.model);
            println!("  model_tier: {} ({})", config.model_tier, config.model_tier_summary());
            println!("  use_gpu: {}", config.use_gpu);
            println!("  model_path: {}", config.model_path.as_deref().unwrap_or("(auto)"));
            println!("  max_new_tokens: {}", config.max_new_tokens);
            println!("  adaptive_max_new_tokens: {}", config.adaptive_max_new_tokens);
            println!("  max_chunk_seconds: {}", config.max_chunk_seconds);

            let mut engine = QwenEngine::new_with_config(&config)?;
            println!("Engine init: OK (gpu={})", engine.active_gpu());
            engine.prepare(None)?;
            println!("Model load: OK");
            drop(engine);

            // O-4: Test audio device availability (previously missing)
            {
                use cpal::traits::HostTrait;
                let host = cpal::default_host();
                match host.default_input_device() {
                    Some(device) => {
                        use cpal::traits::DeviceTrait;
                        let name = device.name().unwrap_or_else(|_| "(unknown)".to_string());
                        println!("Audio device: OK ({})", name);
                    }
                    None => {
                        println!("Audio device: WARN — no default input device detected");
                    }
                }
            }
            Ok(())
        })();
        match validate_result {
            Ok(()) => { println!("Validation passed."); return; }
            Err(e) => { println!("Validation FAILED: {}", e); std::process::exit(1); }
        }
    }

    // Run the app
    if let Err(e) = App::run(config, log_buffer) {
        error!("Application error: {}", e);
        show_fatal_error_dialog(&format!("Application error: {}", e));
        std::process::exit(1);
    }
}

fn apply_cli_overrides(config: &mut Config, args: &[String]) -> Result<(), String> {
    if args.iter().any(|a| a == "--cpu") {
        config.use_gpu = false;
        config.require_gpu = false;
    }
    if args.iter().any(|a| a == "--gpu") {
        config.use_gpu = true;
    }

    if let Some(tier) = arg_value(args, "--model-tier") {
        config.set_model_tier(tier);
    } else if config.auto_select_model {
        config.model = config.model_for_current_tier();
    }

    if let Some(model) = arg_value(args, "--model") {
        if model.trim().is_empty() {
            return Err("--model cannot be empty".to_string());
        }
        config.model = model.to_string();
    }

    if let Some(dir) = arg_value(args, "--model-dir") {
        let path = std::path::Path::new(dir);
        if !path.exists() || !path.is_dir() {
            return Err(format!("--model-dir path not found: {}", dir));
        }
        config.model_path = Some(dir.to_string());
    }

    if let Some(value) = arg_value(args, "--max-new-tokens") {
        config.max_new_tokens = value
            .parse::<u32>()
            .map_err(|_| "--max-new-tokens must be an integer".to_string())?;
    }

    if let Some(value) = arg_value(args, "--chunk-seconds") {
        config.max_chunk_seconds = value
            .parse::<u32>()
            .map_err(|_| "--chunk-seconds must be an integer".to_string())?;
    }

    if args.iter().any(|a| a == "--no-adaptive-tokens") {
        config.adaptive_max_new_tokens = false;
    }

    if args.iter().any(|a| a == "--legacy-prompt") {
        config.legacy_prompt = true;
    }

    if let Some(q) = arg_value(args, "--quant") {
        config.quantization = q.to_string();
    }

    config.validate().map_err(|err| err.to_string())
}

fn arg_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|arg| arg == flag)
        .and_then(|idx| args.get(idx + 1))
        .filter(|value| !value.starts_with('-'))
        .map(String::as_str)
}

fn print_model_list() {
    let mut fast = Config::default();
    fast.set_model_tier("fast");
    let mut best = Config::default();
    best.set_model_tier("best");

    println!("Available Voclaude model tiers:");
    println!("  fast / small / medium");
    println!("    {}", fast.model);
    println!("    {}", fast.model_tier_summary());
    println!("  best / large / accurate");
    println!("    {}", best.model);
    println!("    {}", best.model_tier_summary());
    println!();
    println!("CLI examples:");
    println!("  voclaude.exe --test audio.f32 --cpu --model-tier fast");
    println!("  voclaude.exe --test audio.f32 --gpu --model-tier best");
    println!("  voclaude.exe --validate --model-tier medium");
}

/// Test mode: load audio file and transcribe it
fn run_test(audio_path: &str, args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    info!("=== TEST MODE ===");
    info!("Loading audio file: {}", audio_path);

    let path = std::path::Path::new(audio_path);

    // Create engine and transcribe using file-based path
    info!("Creating inference engine...");
    let mut test_config = Config::load().unwrap_or_default();
    test_config.require_gpu = false; // test mode always allows CPU fallback
    apply_cli_overrides(&mut test_config, args)
        .map_err(|err| format!("Invalid CLI option: {}", err))?;
    info!("Test model: {}", test_config.model);
    info!("Test use_gpu: {}", test_config.use_gpu);
    info!("Test max_chunk_seconds: {}", test_config.max_chunk_seconds);
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
fn attach_console_for_cli() {
    unsafe {
        use windows_sys::Win32::System::Console::{
            AllocConsole, AttachConsole, ATTACH_PARENT_PROCESS,
        };
        if AttachConsole(ATTACH_PARENT_PROCESS) == 0 {
            AllocConsole();
        }
    }
}

#[cfg(not(target_os = "windows"))]
fn attach_console_for_cli() {}

#[cfg(target_os = "windows")]
fn show_fatal_error_dialog(message: &str) {
    use std::ffi::OsStr;
    use std::iter::once;
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::UI::WindowsAndMessaging::{MessageBoxW, MB_ICONERROR, MB_OK};

    let title: Vec<u16> = OsStr::new("Voclaude Error")
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

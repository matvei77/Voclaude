mod audio;
mod backend;
mod config;
mod pipeline;

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use tracing::info;

use crate::backend::build_backend;
use crate::config::AppConfig;
use crate::pipeline::run_transcription;

#[derive(Debug, Parser)]
#[command(name = "voclaude-frontier")]
#[command(about = "Parallel next-gen local STT runtime (Qwen-first, GPU-required).")]
struct Cli {
    #[arg(long, default_value = "config/default.toml")]
    config: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Healthcheck,
    Transcribe {
        #[arg(long)]
        input: PathBuf,

        #[arg(long)]
        output: Option<PathBuf>,
    },
}

fn setup_tracing() {
    let filter = std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string());
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .compact()
        .init();
}

fn resolve_config_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    let cwd = std::env::current_dir().context("failed to get current directory")?;
    Ok(cwd.join(path))
}

fn main() -> Result<()> {
    setup_tracing();
    let cli = Cli::parse();

    let config_path = resolve_config_path(&cli.config)?;
    let config = AppConfig::load(&config_path)?;
    let backend = build_backend(&config.backend)?;

    match cli.command {
        Command::Healthcheck => {
            backend.health_check()?;
            println!("OK: backend={} healthy", backend.name());
        }
        Command::Transcribe { input, output } => {
            let result = run_transcription(&*backend, &config, &input)?;
            println!("{}", result.text);

            if config.runtime.write_json || output.is_some() {
                let out_path = output.unwrap_or_else(|| {
                    let file_name = input
                        .file_stem()
                        .map(|v| v.to_string_lossy().to_string())
                        .unwrap_or_else(|| "transcript".to_string());
                    PathBuf::from(format!("{}_{}.json", file_name, backend.name()))
                });
                let json = serde_json::to_string_pretty(&result)?;
                fs::write(&out_path, json)
                    .with_context(|| format!("failed to write {}", out_path.display()))?;
                info!("Wrote {}", out_path.display());
            }
        }
    }

    Ok(())
}

//! Parakeet speech-to-text through transcribe.cpp (ggml).
//!
//! Runs NVIDIA Parakeet GGUF models (default: Parakeet Ultra, 25 European
//! languages) on the GPU through Vulkan, or on the CPU with ggml's per-ISA
//! kernels. Parakeet is a transducer: there is no prompt or context, and it
//! picks the language of each segment by itself, so short pause-cut segments
//! are what let a recording switch between Russian and English.
//!
//! transcribe.dll and the ggml DLLs are delay-loaded (see build.rs); they are
//! mapped on the first call here, in the `--worker` child only.

use crate::audio::segmenter::{segment_all, SegmenterConfig};
use crate::config::Config;
use crate::inference::{load_audio_file, AsrEngine, InferenceProgress, InferenceStage};
use std::ffi::{c_char, c_void, CStr};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Instant;
use tracing::{debug, error, info, warn};
use transcribe_cpp::{Device, DeviceType, Model, ModelOptions, RunOptions, Session, SessionOptions};
use transcribe_cpp_sys as sys;

/// Shorter input is a click or a breath; the encoder rejects it anyway.
const MIN_SAMPLES: usize = 16_000 / 10;

/// Silent inputs (seconds) run once at load on the GPU: one per Vulkan
/// pipeline size class that real segments (up to 45 s) reach.
const WARMUP_SECS: [usize; 4] = [1, 5, 12, 30];

pub struct ParakeetEngine {
    model_spec: String,
    use_gpu: bool,
    require_gpu: bool,
    cpu_threads: u32,
    segmenter: SegmenterConfig,
    loaded: Option<Loaded>,
}

struct Loaded {
    // Field order is drop order: the session goes before its model.
    session: Session,
    _model: Model,
    on_gpu: bool,
    label: String,
    size_mb: u64,
}

impl ParakeetEngine {
    pub fn new_with_config(config: &Config) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            model_spec: config.parakeet_model.trim().to_string(),
            use_gpu: config.use_gpu,
            require_gpu: config.require_gpu,
            cpu_threads: config.cpu_threads,
            segmenter: config.segmenter_config(),
            loaded: None,
        })
    }

    pub fn is_loaded(&self) -> bool {
        self.loaded.is_some()
    }

    /// Transcribe one segment of a live recording. Loads the model if needed.
    pub fn transcribe_segment(&mut self, samples: &[f32]) -> Result<String, Box<dyn std::error::Error>> {
        if samples.len() < MIN_SAMPLES {
            return Ok(String::new());
        }
        self.prepare(None)?;
        let loaded = self.loaded.as_mut().ok_or("Parakeet model not loaded")?;
        let started = Instant::now();
        match loaded.session.run(samples, &RunOptions::default()) {
            Ok(result) => {
                let wall = started.elapsed().as_secs_f64();
                let secs = samples.len() as f64 / 16_000.0;
                debug!(
                    "Segment: {:.1}s audio in {:.0} ms ({:.0}x), encode {:.0} ms",
                    secs,
                    wall * 1000.0,
                    secs / wall.max(1e-6),
                    result.timings.encode_ms
                );
                Ok(result.text.trim().to_string())
            }
            Err(transcribe_cpp::Error::InputTooShort(_)) => Ok(String::new()),
            // The decoder stopped early (budget or a loop). A retry gives the same
            // result, so keep the text it had instead of failing the segment.
            Err(transcribe_cpp::Error::OutputTruncated { message, partial })
            | Err(transcribe_cpp::Error::OutputRepetition { message, partial }) => {
                warn!("Parakeet stopped early ({}); keeping the partial text", message);
                Ok(partial.map(|p| p.text.trim().to_string()).unwrap_or_default())
            }
            Err(e) => Err(format!("Parakeet: {}", e).into()),
        }
    }

    fn load(&self) -> Result<Loaded, String> {
        init_runtime()?;
        let path = resolve_model(&self.model_spec)?;
        let size_mb = std::fs::metadata(&path).map(|m| m.len() >> 20).unwrap_or(0);
        let devices = transcribe_cpp::devices();
        if self.use_gpu {
            match pick_gpu(&devices) {
                Some(gpu) => match Loaded::open(&path, gpu.clone(), self.cpu_threads, size_mb) {
                    Ok(loaded) => return Ok(loaded),
                    Err(e) if self.require_gpu => return Err(e),
                    Err(e) => warn!("Parakeet on {} failed ({}); using the CPU", gpu.description, e),
                },
                None if self.require_gpu => return Err("No Vulkan GPU found (require_gpu = true)".into()),
                None => warn!("No Vulkan GPU found; using the CPU"),
            }
        }
        let cpu = devices
            .into_iter()
            .find(|d| d.kind == "cpu")
            .ok_or("transcribe.cpp registered no CPU device")?;
        Loaded::open(&path, cpu, self.cpu_threads, size_mb)
    }
}

impl Loaded {
    fn open(path: &Path, device: Device, cpu_threads: u32, size_mb: u64) -> Result<Loaded, String> {
        let started = Instant::now();
        let on_gpu = device.device_type != DeviceType::Cpu;
        let where_ = if on_gpu {
            format!("{} ({})", device.kind, device.description)
        } else {
            "CPU".to_string()
        };
        let model = Model::load_with(path, &ModelOptions { device: Some(device), ..Default::default() })
            .map_err(|e| format!("Failed to load {}: {}", path.display(), e))?;
        let mut session = model
            .session_with(&SessionOptions { n_threads: cpu_threads as i32, ..Default::default() })
            .map_err(|e| format!("Failed to create a Parakeet session: {}", e))?;
        if on_gpu {
            // Vulkan compiles each pipeline on first use, and the matmuls pick
            // their pipelines by input length, so warm every length class. With a
            // cold driver shader cache (new driver or machine) one short run left
            // ~10 s of compiles for the first real segments on the 3060. The model
            // is preloaded when recording starts, so pay it here, not on a segment.
            // A failure here means the GPU cannot run the model; returning it lets
            // load() fall back to the CPU.
            for secs in WARMUP_SECS {
                match session.run(&vec![0.0f32; secs * 16_000], &RunOptions::default()) {
                    Ok(_) | Err(transcribe_cpp::Error::InputTooShort(_)) => {}
                    Err(e) => return Err(format!("Parakeet warm-up on {} failed: {}", where_, e)),
                }
            }
        }
        let name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let label = format!("{} on {}", name, where_);
        info!("Loaded {} in {:.1}s", label, started.elapsed().as_secs_f64());
        Ok(Loaded { session, _model: model, on_gpu, label, size_mb })
    }
}

impl AsrEngine for ParakeetEngine {
    fn prepare(
        &mut self,
        progress: Option<&mut dyn FnMut(InferenceProgress)>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.loaded.is_some() {
            return Ok(());
        }
        if let Some(cb) = progress {
            cb(InferenceProgress {
                stage: InferenceStage::LoadingModel,
                message: "Loading Parakeet...".to_string(),
            });
        }
        self.loaded = Some(self.load()?);
        Ok(())
    }

    /// Whole files (crash recovery, `--test`, the API without ranges) are cut at
    /// pauses exactly like a live recording, so each piece picks its language.
    fn transcribe_file_with_progress(
        &mut self,
        path: &Path,
        mut progress: Option<&mut dyn FnMut(InferenceProgress)>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let samples = load_audio_file(path)?;
        self.prepare(progress.as_deref_mut().map(|p| p as &mut dyn FnMut(InferenceProgress)))?;
        let bounds = segment_all(&samples, self.segmenter);
        let mut texts = Vec::new();
        for (i, b) in bounds.iter().enumerate() {
            if let Some(cb) = progress.as_deref_mut() {
                cb(InferenceProgress {
                    stage: InferenceStage::Transcribing,
                    message: format!("Transcribing {}/{}...", i + 1, bounds.len()),
                });
            }
            let text = self.transcribe_segment(&samples[b.start..b.end.min(samples.len())])?;
            if !text.is_empty() {
                texts.push(text);
            }
        }
        Ok(texts.join(" "))
    }

    fn unload(&mut self) {
        if let Some(loaded) = self.loaded.take() {
            info!("Unloading {}", loaded.label);
        }
    }

    fn active_gpu(&self) -> bool {
        self.loaded.as_ref().map(|l| l.on_gpu).unwrap_or(false)
    }

    fn model_label(&self) -> String {
        match &self.loaded {
            Some(l) => format!("transcribe.cpp ({})", l.label),
            None => format!("transcribe.cpp ({})", self.model_spec),
        }
    }

    fn model_size_mb(&self) -> u64 {
        self.loaded.as_ref().map(|l| l.size_mb).unwrap_or(750)
    }
}

/// Discrete GPU first, then an integrated one (Intel Arc on laptops).
fn pick_gpu(devices: &[Device]) -> Option<&Device> {
    [DeviceType::Gpu, DeviceType::Igpu]
        .iter()
        .find_map(|want| devices.iter().find(|d| d.kind == "vulkan" && d.device_type == *want))
}

/// A local `.gguf` file, or `<owner>/<repo>/<file>` on Hugging Face (cached
/// after the first download).
fn resolve_model(spec: &str) -> Result<PathBuf, String> {
    let local = Path::new(spec);
    if local.is_file() {
        return Ok(local.to_path_buf());
    }
    if local.is_absolute() || spec.contains('\\') {
        return Err(format!("parakeet_model not found: {}", spec));
    }
    let (repo, file) = spec
        .rsplit_once('/')
        .filter(|(repo, _)| repo.contains('/'))
        .ok_or_else(|| format!("parakeet_model must be a .gguf path or <owner>/<repo>/<file>, got {}", spec))?;
    let api = hf_hub::api::sync::Api::new().map_err(|e| format!("Failed to create HF API: {}", e))?;
    info!("Fetching {} from {} (cached after the first download)", file, repo);
    api.model(repo.to_string())
        .get(file)
        .map_err(|e| format!("Failed to download {}: {}", spec, e))
}

/// Load the runtime once per process: check the DLL is there (a missing
/// delay-loaded DLL would otherwise kill the process), route native logs into
/// tracing, and register the CPU and Vulkan backend modules.
fn init_runtime() -> Result<(), String> {
    static INIT: OnceLock<Result<(), String>> = OnceLock::new();
    INIT.get_or_init(|| {
        #[cfg(windows)]
        {
            use windows_sys::Win32::System::LibraryLoader::LoadLibraryW;
            let name: Vec<u16> = "transcribe.dll\0".encode_utf16().collect();
            if unsafe { LoadLibraryW(name.as_ptr()) }.is_null() {
                return Err("transcribe.dll not found next to voclaude.exe (reinstall, or run fetch-transcribe.ps1 for a dev build)".to_string());
            }
        }
        unsafe { sys::transcribe_log_set(Some(native_log), std::ptr::null_mut()) };
        transcribe_cpp::init_backends_default().map_err(|e| format!("transcribe.cpp backends: {}", e))?;
        for d in transcribe_cpp::devices() {
            info!("transcribe.cpp device: {} [{}] {}", d.name, d.kind, d.description);
        }
        Ok(())
    })
    .clone()
}

/// Native and ggml diagnostics: errors and warnings stay visible, the
/// per-segment decoder chatter goes to debug.
extern "C" fn native_log(level: sys::transcribe_log_level, msg: *const c_char, _userdata: *mut c_void) {
    if msg.is_null() {
        return;
    }
    let text = unsafe { CStr::from_ptr(msg) }.to_string_lossy();
    let text = text.trim_end();
    if text.is_empty() {
        return;
    }
    use sys::transcribe_log_level as L;
    match level {
        L::TRANSCRIBE_LOG_LEVEL_ERROR => error!(target: "transcribe", "{}", text),
        L::TRANSCRIBE_LOG_LEVEL_WARN => warn!(target: "transcribe", "{}", text),
        _ => debug!(target: "transcribe", "{}", text),
    }
}

//! Parakeet speech-to-text through transcribe.cpp (ggml).
//!
//! Runs NVIDIA Parakeet GGUF models (default: Parakeet Ultra, 25 European
//! languages) on the GPU through Vulkan, or on the CPU with ggml's per-ISA
//! kernels. Parakeet is a transducer: there is no prompt or context, and it
//! picks the language of each segment by itself, so short pause-cut segments
//! are what let a recording switch between Russian and English. A Russian
//! phrase too short to get its own segment is caught by the re-check below.
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
use transcribe_cpp::{
    Device, DeviceType, Model, ModelOptions, RunOptions, Session, SessionOptions, TimestampKind, Transcript,
};
use transcribe_cpp_sys as sys;

/// Shorter input is a click or a breath; the encoder rejects it anyway.
const MIN_SAMPLES: usize = 16_000 / 10;

// Re-check thresholds, from the 2026-10-26 runs over 3.4 h of personal
// recordings: English words sit at 0.93-1.00, the Latin spelling of
// "кассовый разрыв" at 0.80-0.86 and its Russian re-decode at 0.99.
/// An English word below this confidence starts a re-check...
const RECHECK_SEED_P: f32 = 0.85;
/// ...which takes in the neighbouring English words below this one.
const RECHECK_GROW_P: f32 = 0.90;
/// The Russian re-decode must be at least this confident...
const RECHECK_ACCEPT_P: f32 = 0.90;
/// ...and this much more confident than the words it replaces.
const RECHECK_MARGIN: f32 = 0.10;
/// Words of context decoded with the run on each side (one or two keep the
/// Russian in Cyrillic, four already tip it back to Latin)...
const RECHECK_CONTEXT_WORDS: usize = 2;
/// ...unless a pause this long (seconds) comes first.
const RECHECK_PAUSE_SECS: f64 = 0.2;

/// Silent inputs (seconds) run once at load on the GPU: one per Vulkan
/// pipeline size class that real segments (up to 45 s) reach.
const WARMUP_SECS: [usize; 4] = [1, 5, 12, 30];

pub struct ParakeetEngine {
    model_spec: String,
    use_gpu: bool,
    require_gpu: bool,
    cpu_threads: u32,
    recheck: bool,
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
            recheck: config.parakeet_recheck,
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
        match loaded.session.run(samples, &token_options()) {
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
                if self.recheck {
                    if let Some(text) = recheck_russian(&mut loaded.session, samples, &result) {
                        return Ok(text);
                    }
                }
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

/// Token timestamps carry the per-token confidence the re-check reads. Same
/// speed as the default for Parakeet (measured 2026-10-26).
fn token_options() -> RunOptions {
    RunOptions { timestamps: TimestampKind::Token, ..Default::default() }
}

/// A transcribed word: seconds from the start of the decoded audio, and the
/// lowest confidence among its tokens.
#[derive(Debug, Clone)]
struct TimedWord {
    t0: f64,
    t1: f64,
    p: f32,
    text: String,
}

fn timed_words(result: &Transcript) -> Vec<TimedWord> {
    let mut min_p = vec![f32::INFINITY; result.words.len()];
    for token in &result.tokens {
        if let Some(p) = usize::try_from(token.word_index).ok().and_then(|i| min_p.get_mut(i)) {
            *p = p.min(token.p);
        }
    }
    result
        .words
        .iter()
        .zip(min_p)
        // NaN (no confidence) compares false everywhere, so it never triggers a re-check.
        .map(|(w, p)| TimedWord {
            t0: w.t0_ms as f64 / 1000.0,
            t1: w.t1_ms as f64 / 1000.0,
            p,
            text: w.text.trim().to_string(),
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Script {
    Latin,
    Cyrillic,
    Neither,
}

fn script(word: &str) -> Script {
    let cyrillic = word.chars().filter(|c| ('\u{0400}'..='\u{04FF}').contains(c)).count();
    let latin = word.chars().filter(|c| c.is_ascii_alphabetic()).count();
    if cyrillic > latin {
        Script::Cyrillic
    } else if latin > 0 {
        Script::Latin
    } else {
        Script::Neither
    }
}

/// Parakeet writes a whole segment in the alphabet of its main language, so a
/// Russian phrase inside English comes out spelled in Latin letters ("Casa
/// Veras Riv" for "кассовый разрыв"), and the model is unsure of those words.
/// Decoded again with only a word or two of English around it, the same audio
/// comes out in Cyrillic. Returns the segment text with such phrases replaced,
/// or `None` if nothing changed.
///
/// One way only: re-decoding unsure words of a Russian segment in Latin
/// letters gave Polish and Slovak, not English.
fn recheck_russian(session: &mut Session, samples: &[f32], first: &Transcript) -> Option<String> {
    let words = timed_words(first);
    let secs = samples.len() as f64 / 16_000.0;
    let mut replacements: Vec<(usize, usize, String)> = Vec::new();
    for (i, j) in shaky_runs(&words) {
        if replacements.last().is_some_and(|&(_, last, _)| i <= last) {
            continue;
        }
        let (a, b) = recheck_window(&words, i, j, secs);
        let start = ((a * 16_000.0) as usize).min(samples.len());
        let end = ((b * 16_000.0) as usize).min(samples.len());
        // Too short or a decoder stop: keep the first pass.
        let Ok(result) = session.run(&samples[start..end], &token_options()) else {
            continue;
        };
        if let Some(found) = russian_replacement(&words, i, j, &timed_words(&result), a) {
            if replacements.last().map_or(true, |&(_, last, _)| found.0 > last) {
                debug!("Re-check: words {}-{} of {} replaced with Russian", found.0, found.1, words.len());
                replacements.push(found);
            }
        }
    }
    (!replacements.is_empty()).then(|| splice(&words, &replacements))
}

/// Runs of unsure Latin words in a mostly Latin segment, as (first, last)
/// word indexes: each starts at a word below `RECHECK_SEED_P` and takes in
/// neighbours below `RECHECK_GROW_P`.
fn shaky_runs(words: &[TimedWord]) -> Vec<(usize, usize)> {
    let latin = words.iter().filter(|w| script(&w.text) == Script::Latin).count();
    let cyrillic = words.iter().filter(|w| script(&w.text) == Script::Cyrillic).count();
    if latin <= cyrillic {
        return Vec::new();
    }
    let weak = |w: &TimedWord| w.p < RECHECK_GROW_P && script(&w.text) == Script::Latin;
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for (i, w) in words.iter().enumerate() {
        if !(w.p < RECHECK_SEED_P && script(&w.text) == Script::Latin) {
            continue;
        }
        if runs.last().is_some_and(|&(_, last)| last >= i) {
            continue;
        }
        let (mut first, mut last) = (i, i);
        while first > 0 && weak(&words[first - 1]) {
            first -= 1;
        }
        while last + 1 < words.len() && weak(&words[last + 1]) {
            last += 1;
        }
        match runs.last_mut() {
            Some((_, prev_last)) if *prev_last + 1 >= first => *prev_last = last,
            _ => runs.push((first, last)),
        }
    }
    runs
}

/// The audio (seconds) to decode again for `words[i..=j]`: up to
/// `RECHECK_CONTEXT_WORDS` words on each side, stopping at a pause, with the
/// edges in the middle of the gap between words (at most 0.3 s out).
fn recheck_window(words: &[TimedWord], i: usize, j: usize, total_secs: f64) -> (f64, f64) {
    let mut first = i;
    while first > 0 && i - first < RECHECK_CONTEXT_WORDS && words[first].t0 - words[first - 1].t1 < RECHECK_PAUSE_SECS {
        first -= 1;
    }
    let start = if first == 0 {
        0.0
    } else {
        ((words[first - 1].t1 + words[first].t0) / 2.0).max(words[first].t0 - 0.3)
    };
    let mut last = j;
    while last + 1 < words.len()
        && last - j < RECHECK_CONTEXT_WORDS
        && words[last + 1].t0 - words[last].t1 < RECHECK_PAUSE_SECS
    {
        last += 1;
    }
    let end = if last + 1 == words.len() {
        total_secs
    } else {
        ((words[last].t1 + words[last + 1].t0) / 2.0).min(words[last].t1 + 0.3)
    };
    (start, end)
}

/// The Cyrillic words of a re-decode (starting `offset` seconds into the
/// segment) that cover `words[i..=j]`, as (first, last, text) over the original
/// words they replace. `None` unless they are confident enough to trust.
fn russian_replacement(
    words: &[TimedWord],
    i: usize,
    j: usize,
    redecoded: &[TimedWord],
    offset: f64,
) -> Option<(usize, usize, String)> {
    let cyrillic = |k: usize| script(&redecoded[k].text) == Script::Cyrillic;
    let (lo, hi) = (words[i].t0 - 0.15, words[j].t1 + 0.15);
    let mut hits = (0..redecoded.len()).filter(|&k| {
        let mid = offset + (redecoded[k].t0 + redecoded[k].t1) / 2.0;
        mid >= lo && mid <= hi && cyrillic(k)
    });
    let mut block_first = hits.next()?;
    let mut block_last = hits.last().unwrap_or(block_first);
    while block_first > 0 && cyrillic(block_first - 1) {
        block_first -= 1;
    }
    while block_last + 1 < redecoded.len() && cyrillic(block_last + 1) {
        block_last += 1;
    }
    let block = &redecoded[block_first..=block_last];

    let old_p = words[i..=j].iter().map(|w| w.p).fold(f32::INFINITY, f32::min);
    let new_p = block.iter().map(|w| w.p).fold(f32::INFINITY, f32::min);
    if !(new_p >= RECHECK_ACCEPT_P && new_p >= old_p + RECHECK_MARGIN) {
        return None;
    }
    // The Russian may also cover a neighbouring Latin word the model was sure
    // of. Numbers stay: the re-decode can drop a digit the first pass heard.
    let (b0, b1) = (offset + block[0].t0 - 0.1, offset + block[block.len() - 1].t1 + 0.1);
    let latin = |w: &TimedWord| script(&w.text) == Script::Latin;
    let (mut first, mut last) = (i, j);
    while first > 0 && latin(&words[first - 1]) && (words[first - 1].t0 + words[first - 1].t1) / 2.0 >= b0 {
        first -= 1;
    }
    while last + 1 < words.len() && latin(&words[last + 1]) && (words[last + 1].t0 + words[last + 1].t1) / 2.0 <= b1 {
        last += 1;
    }
    let text = block.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ");
    Some((first, last, text))
}

/// The segment text with each (first, last, text) replacement spliced in.
/// Punctuation after a replacement comes from the first pass, which heard the
/// whole sentence; capitalisation follows the sentence position.
fn splice(words: &[TimedWord], replacements: &[(usize, usize, String)]) -> String {
    const ENDS: [char; 6] = ['.', ',', '?', '!', ';', ':'];
    let mut parts: Vec<String> = Vec::new();
    let mut next = 0;
    for (first, last, text) in replacements {
        parts.extend(words[next..*first].iter().map(|w| w.text.clone()));
        let trailing = &words[*last].text[words[*last].text.trim_end_matches(ENDS).len()..];
        let mut text = format!("{}{}", text.trim_end_matches(ENDS), trailing);
        let starts_sentence = *first == 0 || words[*first - 1].text.ends_with(['.', '?', '!']);
        let mut chars = text.chars();
        if let Some(c) = chars.next() {
            if starts_sentence {
                text = c.to_uppercase().chain(chars).collect();
            } else if !words[*first].text.starts_with(char::is_uppercase) {
                text = c.to_lowercase().chain(chars).collect();
            }
        }
        parts.push(text);
        next = last + 1;
    }
    parts.extend(words[next..].iter().map(|w| w.text.clone()));
    parts.join(" ")
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

#[cfg(test)]
mod tests {
    use super::*;

    fn w(t0: f64, t1: f64, p: f32, text: &str) -> TimedWord {
        TimedWord { t0, t1, p, text: text.to_string() }
    }

    /// The first pass over "And uh it's called кассовый разрыв to cover
    /// everyone's salaries correctly." (recording of 2026-10-26, segment 5).
    fn casa_veras_riv() -> Vec<TimedWord> {
        vec![
            w(0.64, 0.96, 0.92, "And"),
            w(1.12, 1.44, 0.929, "uh"),
            w(1.84, 2.16, 0.999, "it's"),
            w(2.16, 2.56, 1.0, "called"),
            w(2.56, 2.88, 0.852, "Casa"),
            w(3.04, 3.36, 0.864, "Veras"),
            w(3.52, 3.84, 0.801, "Riv"),
            w(5.12, 5.36, 1.0, "to"),
            w(5.36, 5.76, 1.0, "cover"),
            w(5.76, 6.24, 0.993, "everyone's"),
            w(6.24, 6.64, 1.0, "salaries"),
            w(6.64, 7.04, 0.984, "correctly."),
        ]
    }

    /// The re-decode of 1.64-4.14 s, times relative to the window.
    fn redecode() -> Vec<TimedWord> {
        vec![w(0.52, 0.92, 1.0, "called"), w(0.92, 1.52, 0.994, "кассовый"), w(1.60, 2.20, 0.995, "разрыв.")]
    }

    #[test]
    fn script_of_words() {
        assert_eq!(script("кассовый"), Script::Cyrillic);
        assert_eq!(script("everyone's"), Script::Latin);
        assert_eq!(script("2000"), Script::Neither);
    }

    #[test]
    fn unsure_run_grows_to_the_whole_phrase() {
        assert_eq!(shaky_runs(&casa_veras_riv()), vec![(4, 6)]);
    }

    #[test]
    fn russian_segments_are_left_alone() {
        let words = vec![w(0.0, 0.4, 1.0, "что"), w(0.4, 0.8, 1.0, "за"), w(0.8, 1.2, 0.7, "утром,")];
        assert!(shaky_runs(&words).is_empty());
    }

    #[test]
    fn window_takes_two_words_of_context_and_stops_at_the_pause() {
        let (a, b) = recheck_window(&casa_veras_riv(), 4, 6, 7.5);
        assert!((a - 1.64).abs() < 1e-9, "start {}", a);
        assert!((b - 4.14).abs() < 1e-9, "end {}", b);
    }

    #[test]
    fn confident_russian_replaces_the_latin_spelling() {
        let words = casa_veras_riv();
        let found = russian_replacement(&words, 4, 6, &redecode(), 1.64);
        assert_eq!(found, Some((4, 6, "кассовый разрыв.".to_string())));
        assert_eq!(
            splice(&words, &[found.unwrap()]),
            "And uh it's called кассовый разрыв to cover everyone's salaries correctly."
        );
    }

    #[test]
    fn unsure_or_latin_redecodes_are_rejected() {
        let words = casa_veras_riv();
        let unsure = vec![w(0.92, 1.52, 0.83, "Фиф")];
        assert_eq!(russian_replacement(&words, 4, 6, &unsure, 1.64), None);
        let latin = vec![w(0.92, 1.52, 0.99, "kasovy"), w(1.60, 2.20, 0.99, "razriv")];
        assert_eq!(russian_replacement(&words, 4, 6, &latin, 1.64), None);
    }

    #[test]
    fn numbers_next_to_the_russian_are_kept() {
        // "2000 euro mesi." whose re-decode heard only "тысяч евро в месяц."
        let words = vec![w(0.0, 0.6, 1.0, "2000"), w(0.7, 1.0, 0.84, "euro"), w(1.1, 1.5, 0.84, "mesi.")];
        let redecode = vec![
            w(0.2, 0.6, 0.97, "тысяч"),
            w(0.7, 1.0, 0.99, "евро"),
            w(1.05, 1.1, 0.99, "в"),
            w(1.1, 1.5, 0.99, "месяц."),
        ];
        let found = russian_replacement(&words, 1, 2, &redecode, 0.0).expect("confident Russian");
        assert_eq!(found.0, 1, "the number must not be replaced");
        assert_eq!(splice(&words, &[found]), "2000 тысяч евро в месяц.");
    }

    #[test]
    fn splice_keeps_first_pass_punctuation_and_sentence_case() {
        let words = vec![w(0.0, 0.5, 0.7, "Kassovy"), w(0.5, 1.0, 0.7, "razriv,"), w(1.0, 1.5, 1.0, "right.")];
        assert_eq!(splice(&words, &[(0, 1, "кассовый разрыв.".to_string())]), "Кассовый разрыв, right.");
    }
}

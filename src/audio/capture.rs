//! Audio capture using cpal.

use super::{mono_from_interleaved, LinearResampler, RingBuffer};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, SampleRate, Stream, StreamConfig, SupportedStreamConfig};
use crossbeam_channel::Sender;
use directories::ProjectDirs;
use std::fs::{self, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

/// Target sample rate for Whisper (16kHz)
pub const WHISPER_SAMPLE_RATE: u32 = 16000;

/// Ring buffer capacity in seconds of input audio.
const RING_BUFFER_SECONDS: usize = 2;

/// Writer thread chunk size in samples (interleaved).
const WRITER_CHUNK_SAMPLES: usize = 4096;

/// Writer idle sleep to avoid busy spinning.
const WRITER_IDLE_SLEEP_MS: u64 = 2;

pub struct AudioRecording {
    pub path: PathBuf,
    pub sample_rate: u32,
    pub sample_count: usize,
}

impl AudioRecording {
    pub fn is_empty(&self) -> bool {
        self.sample_count == 0
    }
}

pub struct AudioCapture {
    device_name: String,
    device: Device,
    config: StreamConfig,
    sample_format: SampleFormat,
    buffer: Arc<RingBuffer>,
    stream: Arc<Mutex<Option<Stream>>>,
    is_recording: Arc<AtomicBool>,
    writer_stop: Arc<AtomicBool>,
    writer_handle: Arc<Mutex<Option<JoinHandle<WriterThreadResult>>>>,
    dropped_samples: Arc<AtomicUsize>,
    level_tx: Option<Sender<f32>>,
}

struct WriterResult {
    path: PathBuf,
    sample_count: usize,
    dropped_input_samples: usize,
}

type WriterThreadResult = Result<WriterResult, String>;

impl AudioCapture {
    pub fn new(level_tx: Option<Sender<f32>>) -> Result<Self, Box<dyn std::error::Error>> {
        let host = cpal::default_host();

        let device = host
            .default_input_device()
            .ok_or("No input device available")?;

        let device_name = device.name()?;
        info!("Using audio device: {}", device_name);

        let supported_configs = device.supported_input_configs()?;
        let config = Self::find_best_config(supported_configs)?;
        info!(
            "Audio config: {} Hz, {} channel(s), {:?}",
            config.sample_rate().0,
            config.channels(),
            config.sample_format()
        );

        let ring_capacity = config.sample_rate().0 as usize
            * config.channels() as usize
            * RING_BUFFER_SECONDS;

        let buffer = Arc::new(RingBuffer::new(ring_capacity));
        let is_recording = Arc::new(AtomicBool::new(false));
        let stream = Arc::new(Mutex::new(None));
        let writer_stop = Arc::new(AtomicBool::new(false));
        let writer_handle = Arc::new(Mutex::new(None));
        let dropped_samples = Arc::new(AtomicUsize::new(0));

        Ok(Self {
            device_name,
            device,
            config: config.clone().into(),
            sample_format: config.sample_format(),
            buffer,
            stream,
            is_recording,
            writer_stop,
            writer_handle,
            dropped_samples,
            level_tx,
        })
    }

    fn find_best_config(
        configs: cpal::SupportedInputConfigs,
    ) -> Result<SupportedStreamConfig, Box<dyn std::error::Error>> {
        let target_rate = SampleRate(WHISPER_SAMPLE_RATE);

        let configs: Vec<_> = configs.collect();

        for config in &configs {
            if config.channels() == 1
                && config.min_sample_rate() <= target_rate
                && config.max_sample_rate() >= target_rate
            {
                return Ok(config.with_sample_rate(target_rate));
            }
        }

        for config in &configs {
            if config.min_sample_rate() <= target_rate
                && config.max_sample_rate() >= target_rate
            {
                return Ok(config.with_sample_rate(target_rate));
            }
        }

        if let Some(config) = configs.into_iter().max_by_key(|c| c.max_sample_rate().0) {
            let rate = config.max_sample_rate();
            warn!(
                "Using non-ideal sample rate: {} Hz (will resample)",
                rate.0
            );
            return Ok(config.with_sample_rate(rate));
        }

        Err("No supported audio config found".into())
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Start recording
    pub fn start(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.is_recording.load(Ordering::Relaxed) {
            return Ok(());
        }

        self.buffer.clear();
        self.dropped_samples.store(0, Ordering::Relaxed);
        self.writer_stop.store(false, Ordering::SeqCst);

        let channels = self.config.channels as usize;
        let source_rate = self.config.sample_rate.0;
        let output_path = audio_output_path()?;

        let buffer = self.buffer.clone();
        let stop = self.writer_stop.clone();
        let dropped = self.dropped_samples.clone();
        let level_tx = self.level_tx.clone();

        let writer_handle = thread::Builder::new()
            .name("audio-writer".to_string())
            .spawn(move || {
                writer_thread(
                    buffer,
                    stop,
                    dropped,
                    channels,
                    source_rate,
                    output_path,
                    level_tx,
                )
            })?;

        *self.writer_handle.lock().unwrap() = Some(writer_handle);

        let err_fn = |err| {
            error!("Audio stream error: {}", err);
        };

        let buffer = self.buffer.clone();
        let is_recording = self.is_recording.clone();
        let dropped = self.dropped_samples.clone();

        let stream = match self.sample_format {
            SampleFormat::F32 => {
                let buffer = buffer.clone();
                let is_recording = is_recording.clone();
                let dropped = dropped.clone();
                self.device.build_input_stream(
                    &self.config,
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        Self::handle_input_f32(data, &buffer, &is_recording, &dropped);
                    },
                    err_fn,
                    None,
                )?
            }
            SampleFormat::I16 => {
                let buffer = buffer.clone();
                let is_recording = is_recording.clone();
                let dropped = dropped.clone();
                self.device.build_input_stream(
                    &self.config,
                    move |data: &[i16], _: &cpal::InputCallbackInfo| {
                        Self::handle_input_mapped(
                            data,
                            &buffer,
                            &is_recording,
                            &dropped,
                            Self::i16_to_f32,
                        );
                    },
                    err_fn,
                    None,
                )?
            }
            SampleFormat::U16 => {
                let buffer = buffer.clone();
                let is_recording = is_recording.clone();
                let dropped = dropped.clone();
                self.device.build_input_stream(
                    &self.config,
                    move |data: &[u16], _: &cpal::InputCallbackInfo| {
                        Self::handle_input_mapped(
                            data,
                            &buffer,
                            &is_recording,
                            &dropped,
                            Self::u16_to_f32,
                        );
                    },
                    err_fn,
                    None,
                )?
            }
            other => {
                return Err(format!("Unsupported audio sample format: {:?}", other).into());
            }
        };

        self.is_recording.store(true, Ordering::SeqCst);
        stream.play()?;

        *self.stream.lock().unwrap() = Some(stream);

        debug!("Recording started");
        Ok(())
    }

    /// Stop recording and return recording info
    pub fn stop(&self) -> Result<AudioRecording, Box<dyn std::error::Error>> {
        self.is_recording.store(false, Ordering::SeqCst);

        if let Ok(mut stream_guard) = self.stream.lock() {
            if let Some(stream) = stream_guard.take() {
                let _ = stream.pause();
                drop(stream);
                debug!("Stream stopped and dropped");
            }
        }

        self.writer_stop.store(true, Ordering::SeqCst);

        let result = if let Some(handle) = self.writer_handle.lock().unwrap().take() {
            match handle.join() {
                Ok(Ok(result)) => result,
                Ok(Err(err)) => return Err(err.into()),
                Err(_) => return Err("Writer thread panicked".into()),
            }
        } else {
            WriterResult {
                path: audio_output_path()?,
                sample_count: 0,
                dropped_input_samples: 0,
            }
        };

        if result.dropped_input_samples > 0 {
            warn!(
                "Dropped {} input samples due to ring overflow",
                result.dropped_input_samples
            );
        }

        debug!(
            "Recording stopped, got {} samples, file at {:?}",
            result.sample_count, result.path
        );

        Ok(AudioRecording {
            path: result.path,
            sample_rate: WHISPER_SAMPLE_RATE,
            sample_count: result.sample_count,
        })
    }

    fn handle_input_f32(
        data: &[f32],
        buffer: &RingBuffer,
        is_recording: &AtomicBool,
        dropped: &AtomicUsize,
    ) {
        if !is_recording.load(Ordering::Relaxed) {
            return;
        }

        let written = buffer.push_slice(data);
        if written < data.len() {
            dropped.fetch_add(data.len() - written, Ordering::Relaxed);
        }
    }

    fn handle_input_mapped<T, F>(
        data: &[T],
        buffer: &RingBuffer,
        is_recording: &AtomicBool,
        dropped: &AtomicUsize,
        to_f32: F,
    ) where
        T: Copy,
        F: Fn(T) -> f32 + Copy,
    {
        if !is_recording.load(Ordering::Relaxed) {
            return;
        }

        let written = buffer.push_mapped(data, to_f32);
        if written < data.len() {
            dropped.fetch_add(data.len() - written, Ordering::Relaxed);
        }
    }

    fn i16_to_f32(sample: i16) -> f32 {
        sample as f32 / 32768.0
    }

    fn u16_to_f32(sample: u16) -> f32 {
        (sample as f32 - 32768.0) / 32768.0
    }
}

fn audio_output_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    ProjectDirs::from("com", "voclaude", "Voclaude")
        .map(|dirs| dirs.data_dir().join("audio.f32"))
        .ok_or_else(|| "Could not determine audio output path".into())
}

fn write_f32_le<W: Write>(
    writer: &mut W,
    samples: &[f32],
    scratch: &mut Vec<u8>,
) -> io::Result<()> {
    scratch.clear();
    scratch.reserve(samples.len() * 4);
    for sample in samples {
        scratch.extend_from_slice(&sample.to_le_bytes());
    }
    writer.write_all(scratch)
}

fn writer_thread(
    buffer: Arc<RingBuffer>,
    stop: Arc<AtomicBool>,
    dropped: Arc<AtomicUsize>,
    channels: usize,
    source_rate: u32,
    output_path: PathBuf,
    level_tx: Option<Sender<f32>>,
) -> WriterThreadResult {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }

    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&output_path)
        .map_err(|e| e.to_string())?;
    let mut writer = BufWriter::new(file);

    let mut resampler = if source_rate != WHISPER_SAMPLE_RATE {
        Some(LinearResampler::new(source_rate, WHISPER_SAMPLE_RATE))
    } else {
        None
    };

    let mut scratch = vec![0.0f32; WRITER_CHUNK_SAMPLES];
    let mut pending: Vec<f32> = Vec::with_capacity(WRITER_CHUNK_SAMPLES * 2);
    let mut bytes: Vec<u8> = Vec::with_capacity(WRITER_CHUNK_SAMPLES * 4);
    let mut sample_count = 0usize;
    let level_interval = Duration::from_millis(60);
    let mut last_level_sent = Instant::now() - level_interval;

    loop {
        let read = buffer.pop_slice(&mut scratch);
        if read == 0 {
            if stop.load(Ordering::Acquire) && buffer.is_empty() {
                break;
            }
            thread::sleep(Duration::from_millis(WRITER_IDLE_SLEEP_MS));
            continue;
        }

        pending.extend_from_slice(&scratch[..read]);
        let frames = pending.len() / channels;
        if frames == 0 {
            continue;
        }
        let take = frames * channels;
        let mono = mono_from_interleaved(&pending[..take], channels, |v| v);
        pending.drain(0..take);

        let resampled = match resampler.as_mut() {
            Some(resampler) => resampler.process(&mono),
            None => mono,
        };

        if !resampled.is_empty() {
            write_f32_le(&mut writer, &resampled, &mut bytes).map_err(|e| e.to_string())?;
            sample_count += resampled.len();

            if let Some(tx) = &level_tx {
                if last_level_sent.elapsed() >= level_interval {
                    let mut peak = 0.0f32;
                    for sample in &resampled {
                        let value = sample.abs();
                        if value > peak {
                            peak = value;
                        }
                    }
                    let _ = tx.try_send(peak.min(1.0));
                    last_level_sent = Instant::now();
                }
            }
        }
    }

    writer.flush().map_err(|e| e.to_string())?;

    Ok(WriterResult {
        path: output_path,
        sample_count,
        dropped_input_samples: dropped.load(Ordering::Relaxed),
    })
}

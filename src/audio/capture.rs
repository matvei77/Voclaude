//! Audio capture using cpal.

use super::{mono_from_interleaved, resample_linear, RingBuffer};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, SampleRate, Stream, StreamConfig, SupportedStreamConfig};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tracing::{debug, error, info, warn};

/// Target sample rate for Whisper (16kHz)
pub const WHISPER_SAMPLE_RATE: u32 = 16000;

/// Initial buffer capacity.
const INITIAL_BUFFER_SECONDS: usize = 60;
const INITIAL_BUFFER_SIZE: usize = WHISPER_SAMPLE_RATE as usize * INITIAL_BUFFER_SECONDS;

/// Guardrail to avoid runaway memory usage.
const MAX_BUFFER_SECONDS: usize = 600;

pub struct AudioCapture {
    device: Device,
    config: StreamConfig,
    sample_format: SampleFormat,
    buffer: Arc<RingBuffer>,
    stream: Arc<Mutex<Option<Stream>>>,
    is_recording: Arc<AtomicBool>,
}

impl AudioCapture {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let host = cpal::default_host();

        let device = host
            .default_input_device()
            .ok_or("No input device available")?;

        info!("Using audio device: {}", device.name()?);

        // Get supported config, prefer 16kHz mono
        let supported_configs = device.supported_input_configs()?;

        let config = Self::find_best_config(supported_configs)?;
        info!(
            "Audio config: {} Hz, {} channel(s), {:?}",
            config.sample_rate().0,
            config.channels(),
            config.sample_format()
        );

        let buffer = Arc::new(RingBuffer::new(INITIAL_BUFFER_SIZE));
        let is_recording = Arc::new(AtomicBool::new(false));
        let stream = Arc::new(Mutex::new(None));

        Ok(Self {
            device,
            config: config.clone().into(),
            sample_format: config.sample_format(),
            buffer,
            stream,
            is_recording,
        })
    }

    fn find_best_config(
        configs: cpal::SupportedInputConfigs,
    ) -> Result<SupportedStreamConfig, Box<dyn std::error::Error>> {
        // Try to find a config that supports our target sample rate
        let target_rate = SampleRate(WHISPER_SAMPLE_RATE);

        // Collect all configs
        let configs: Vec<_> = configs.collect();

        // First, try to find mono 16kHz
        for config in &configs {
            if config.channels() == 1
                && config.min_sample_rate() <= target_rate
                && config.max_sample_rate() >= target_rate
            {
                return Ok(config.with_sample_rate(target_rate));
            }
        }

        // Try stereo 16kHz (we'll convert to mono)
        for config in &configs {
            if config.min_sample_rate() <= target_rate
                && config.max_sample_rate() >= target_rate
            {
                return Ok(config.with_sample_rate(target_rate));
            }
        }

        // Fall back to any config with highest quality, we'll resample
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

    /// Start recording
    pub fn start(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.is_recording.load(Ordering::Relaxed) {
            return Ok(()); // Already recording
        }

        // Clear buffer
        self.buffer.clear();

        let buffer = self.buffer.clone();
        let is_recording = self.is_recording.clone();
        let channels = self.config.channels as usize;
        let source_rate = self.config.sample_rate.0;
        let target_rate = WHISPER_SAMPLE_RATE;
        let max_samples = WHISPER_SAMPLE_RATE as usize * MAX_BUFFER_SECONDS;
        let overflowed = Arc::new(AtomicBool::new(false));

        let err_fn = |err| {
            error!("Audio stream error: {}", err);
        };

        // Build the stream
        let stream = match self.sample_format {
            SampleFormat::F32 => {
                let buffer = buffer.clone();
                let is_recording = is_recording.clone();
                let overflowed = overflowed.clone();
                self.device.build_input_stream(
                    &self.config,
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        Self::handle_input(
                            data,
                            channels,
                            source_rate,
                            target_rate,
                            &buffer,
                            &is_recording,
                            max_samples,
                            &overflowed,
                            |sample| sample,
                        );
                    },
                    err_fn,
                    None,
                )?
            }
            SampleFormat::I16 => {
                let buffer = buffer.clone();
                let is_recording = is_recording.clone();
                let overflowed = overflowed.clone();
                self.device.build_input_stream(
                    &self.config,
                    move |data: &[i16], _: &cpal::InputCallbackInfo| {
                        Self::handle_input(
                            data,
                            channels,
                            source_rate,
                            target_rate,
                            &buffer,
                            &is_recording,
                            max_samples,
                            &overflowed,
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
                let overflowed = overflowed.clone();
                self.device.build_input_stream(
                    &self.config,
                    move |data: &[u16], _: &cpal::InputCallbackInfo| {
                        Self::handle_input(
                            data,
                            channels,
                            source_rate,
                            target_rate,
                            &buffer,
                            &is_recording,
                            max_samples,
                            &overflowed,
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

        // CRITICAL: Set is_recording BEFORE starting stream
        // Otherwise the callback will ignore samples until this flag is set
        self.is_recording.store(true, Ordering::SeqCst);

        stream.play()?;

        // Store stream so we can stop it later
        *self.stream.lock().unwrap() = Some(stream);

        debug!("Recording started");
        Ok(())
    }

    /// Stop recording and return captured samples
    pub fn stop(&self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        self.is_recording.store(false, Ordering::SeqCst);

        // Small delay to ensure last samples are captured
        std::thread::sleep(std::time::Duration::from_millis(50));

        // Stop and drop the stream
        if let Ok(mut stream_guard) = self.stream.lock() {
            if let Some(stream) = stream_guard.take() {
                // Pause the stream before dropping
                let _ = stream.pause();
                drop(stream);
                debug!("Stream stopped and dropped");
            }
        }

        let samples = self.buffer.pop_all();
        debug!("Recording stopped, got {} samples", samples.len());

        Ok(samples)
    }

    /// Check if currently recording
    pub fn is_recording(&self) -> bool {
        self.is_recording.load(Ordering::Relaxed)
    }

    fn handle_input<T, F>(
        data: &[T],
        channels: usize,
        source_rate: u32,
        target_rate: u32,
        buffer: &RingBuffer,
        is_recording: &AtomicBool,
        max_samples: usize,
        overflowed: &AtomicBool,
        to_f32: F,
    ) where
        T: Copy,
        F: Fn(T) -> f32 + Copy,
    {
        if !is_recording.load(Ordering::Relaxed) {
            return;
        }

        let mono = mono_from_interleaved(data, channels, to_f32);
        let samples = if source_rate != target_rate {
            resample_linear(&mono, source_rate, target_rate)
        } else {
            mono
        };

        let current_len = buffer.len();
        if current_len >= max_samples {
            Self::mark_overflow(is_recording, overflowed, max_samples);
            return;
        }

        let remaining = max_samples - current_len;
        if samples.len() > remaining {
            buffer.push(&samples[..remaining]);
            Self::mark_overflow(is_recording, overflowed, max_samples);
            return;
        }

        buffer.push(&samples);
    }

    fn mark_overflow(is_recording: &AtomicBool, overflowed: &AtomicBool, max_samples: usize) {
        if !overflowed.swap(true, Ordering::Relaxed) {
            warn!(
                "Reached max recording duration ({:.1}s); pausing capture",
                max_samples as f32 / WHISPER_SAMPLE_RATE as f32
            );
        }
        is_recording.store(false, Ordering::SeqCst);
    }

    fn i16_to_f32(sample: i16) -> f32 {
        sample as f32 / 32768.0
    }

    fn u16_to_f32(sample: u16) -> f32 {
        (sample as f32 - 32768.0) / 32768.0
    }
}

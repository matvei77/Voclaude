//! Audio capture using cpal.

use super::RingBuffer;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleRate, Stream, StreamConfig};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tracing::{debug, error, info, warn};

/// Target sample rate for Whisper (16kHz)
pub const WHISPER_SAMPLE_RATE: u32 = 16000;

/// Initial buffer capacity (grows as needed for unlimited recording)
const INITIAL_BUFFER_SECONDS: usize = 60;
const INITIAL_BUFFER_SIZE: usize = WHISPER_SAMPLE_RATE as usize * INITIAL_BUFFER_SECONDS;

pub struct AudioCapture {
    device: Device,
    config: StreamConfig,
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
            "Audio config: {} Hz, {} channel(s)",
            config.sample_rate.0, config.channels
        );

        let buffer = Arc::new(RingBuffer::new(INITIAL_BUFFER_SIZE));
        let is_recording = Arc::new(AtomicBool::new(false));
        let stream = Arc::new(Mutex::new(None));

        Ok(Self {
            device,
            config,
            buffer,
            stream,
            is_recording,
        })
    }

    fn find_best_config(
        mut configs: cpal::SupportedInputConfigs,
    ) -> Result<StreamConfig, Box<dyn std::error::Error>> {
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
                return Ok(config.with_sample_rate(target_rate).into());
            }
        }

        // Try stereo 16kHz (we'll convert to mono)
        for config in &configs {
            if config.min_sample_rate() <= target_rate
                && config.max_sample_rate() >= target_rate
            {
                return Ok(config.with_sample_rate(target_rate).into());
            }
        }

        // Fall back to any config with highest quality, we'll resample
        if let Some(config) = configs.into_iter().max_by_key(|c| c.max_sample_rate().0) {
            let rate = config.max_sample_rate();
            warn!(
                "Using non-ideal sample rate: {} Hz (will resample)",
                rate.0
            );
            return Ok(config.with_sample_rate(rate).into());
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

        // Build the stream
        let stream = self.device.build_input_stream(
            &self.config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                if !is_recording.load(Ordering::Relaxed) {
                    return;
                }

                // Convert to mono if needed
                let mono: Vec<f32> = if channels > 1 {
                    data.chunks(channels)
                        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
                        .collect()
                } else {
                    data.to_vec()
                };

                // Resample if needed (simple linear interpolation)
                let samples = if source_rate != target_rate {
                    Self::resample(&mono, source_rate, target_rate)
                } else {
                    mono
                };

                buffer.push(&samples);
            },
            |err| {
                error!("Audio stream error: {}", err);
            },
            None,
        )?;

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

    /// Simple linear resampling
    fn resample(samples: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
        if source_rate == target_rate {
            return samples.to_vec();
        }

        let ratio = source_rate as f64 / target_rate as f64;
        let new_len = (samples.len() as f64 / ratio) as usize;
        let mut output = Vec::with_capacity(new_len);

        for i in 0..new_len {
            let src_idx = i as f64 * ratio;
            let idx = src_idx as usize;
            let frac = src_idx - idx as f64;

            let sample = if idx + 1 < samples.len() {
                samples[idx] * (1.0 - frac as f32) + samples[idx + 1] * frac as f32
            } else if idx < samples.len() {
                samples[idx]
            } else {
                0.0
            };

            output.push(sample);
        }

        output
    }
}

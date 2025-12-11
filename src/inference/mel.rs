//! Mel spectrogram computation for Parakeet TDT 0.6B v2.
//!
//! The istupakov ONNX export expects:
//! - n_fft: 512
//! - hop_length: 160 (10ms at 16kHz)
//! - win_length: 400 (25ms at 16kHz)
//! - n_mels: 128 (FastConformer uses 128 mels)
//! - fmin: 0
//! - fmax: 8000

use ndarray::{Array1, Array2};
use realfft::{RealFftPlanner, RealToComplex};
use std::f32::consts::PI;
use std::sync::Arc;

/// Mel spectrogram parameters for Parakeet TDT 0.6B v2
pub const N_FFT: usize = 512;
pub const HOP_LENGTH: usize = 160;
pub const WIN_LENGTH: usize = 400;
pub const N_MELS: usize = 128;  // FastConformer encoder expects 128 mels
pub const SAMPLE_RATE: u32 = 16000;
pub const F_MIN: f32 = 0.0;
pub const F_MAX: f32 = 8000.0;

/// Mel spectrogram extractor
pub struct MelSpectrogram {
    fft: Arc<dyn RealToComplex<f32>>,
    mel_filterbank: Array2<f32>,
    window: Vec<f32>,
}

impl MelSpectrogram {
    pub fn new() -> Self {
        // Create FFT planner
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);

        // Create Hann window
        let window: Vec<f32> = (0..WIN_LENGTH)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / WIN_LENGTH as f32).cos()))
            .collect();

        // Create mel filterbank
        let mel_filterbank = Self::create_mel_filterbank();

        Self {
            fft,
            mel_filterbank,
            window,
        }
    }

    /// Convert Hz to Mel scale
    fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    /// Convert Mel to Hz scale
    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }

    /// Create mel filterbank matrix
    fn create_mel_filterbank() -> Array2<f32> {
        let n_fft_bins = N_FFT / 2 + 1;
        let mut filterbank = Array2::<f32>::zeros((N_MELS, n_fft_bins));

        let mel_min = Self::hz_to_mel(F_MIN);
        let mel_max = Self::hz_to_mel(F_MAX);

        // Create mel points
        let mel_points: Vec<f32> = (0..=N_MELS + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (N_MELS + 1) as f32)
            .collect();

        // Convert to Hz and then to FFT bin indices
        let hz_points: Vec<f32> = mel_points.iter().map(|&m| Self::mel_to_hz(m)).collect();
        let bin_points: Vec<usize> = hz_points
            .iter()
            .map(|&hz| ((N_FFT as f32 + 1.0) * hz / SAMPLE_RATE as f32).floor() as usize)
            .collect();

        // Create triangular filters
        for m in 0..N_MELS {
            let left = bin_points[m];
            let center = bin_points[m + 1];
            let right = bin_points[m + 2];

            // Rising edge
            for k in left..center {
                if center > left {
                    filterbank[[m, k]] = (k - left) as f32 / (center - left) as f32;
                }
            }

            // Falling edge
            for k in center..right {
                if right > center {
                    filterbank[[m, k]] = (right - k) as f32 / (right - center) as f32;
                }
            }
        }

        filterbank
    }

    /// Compute mel spectrogram from audio samples
    /// Input: f32 samples at 16kHz
    /// Output: [n_mels, n_frames] mel spectrogram
    pub fn compute(&self, samples: &[f32]) -> Array2<f32> {
        let n_frames = (samples.len().saturating_sub(WIN_LENGTH)) / HOP_LENGTH + 1;
        if n_frames == 0 {
            return Array2::zeros((N_MELS, 1));
        }

        let n_fft_bins = N_FFT / 2 + 1;
        let mut spectrogram = Array2::<f32>::zeros((N_MELS, n_frames));

        // Prepare FFT buffers
        let mut fft_input = vec![0.0f32; N_FFT];
        let mut fft_output = self.fft.make_output_vec();

        for frame_idx in 0..n_frames {
            let start = frame_idx * HOP_LENGTH;
            let end = (start + WIN_LENGTH).min(samples.len());

            // Clear and fill FFT input with windowed samples
            fft_input.fill(0.0);
            for (i, &sample) in samples[start..end].iter().enumerate() {
                fft_input[i] = sample * self.window[i];
            }

            // Compute FFT
            self.fft
                .process(&mut fft_input, &mut fft_output)
                .expect("FFT failed");

            // Compute power spectrum
            let power_spectrum: Vec<f32> = fft_output
                .iter()
                .map(|c| c.norm_sqr())
                .collect();

            // Apply mel filterbank
            for mel_idx in 0..N_MELS {
                let mut mel_energy = 0.0f32;
                for bin_idx in 0..n_fft_bins.min(power_spectrum.len()) {
                    mel_energy += self.mel_filterbank[[mel_idx, bin_idx]] * power_spectrum[bin_idx];
                }
                // Log mel spectrogram (add small epsilon to avoid log(0))
                spectrogram[[mel_idx, frame_idx]] = (mel_energy + 1e-10).ln();
            }
        }

        // Normalize per feature (mean=0, std=1)
        self.normalize(&mut spectrogram);

        spectrogram
    }

    /// Per-feature normalization
    fn normalize(&self, spec: &mut Array2<f32>) {
        let (n_mels, n_frames) = spec.dim();
        if n_frames == 0 {
            return;
        }

        for mel_idx in 0..n_mels {
            // Compute mean
            let mut sum = 0.0f32;
            for frame_idx in 0..n_frames {
                sum += spec[[mel_idx, frame_idx]];
            }
            let mean = sum / n_frames as f32;

            // Compute std
            let mut var_sum = 0.0f32;
            for frame_idx in 0..n_frames {
                let diff = spec[[mel_idx, frame_idx]] - mean;
                var_sum += diff * diff;
            }
            let std = (var_sum / n_frames as f32).sqrt().max(1e-10);

            // Normalize
            for frame_idx in 0..n_frames {
                spec[[mel_idx, frame_idx]] = (spec[[mel_idx, frame_idx]] - mean) / std;
            }
        }
    }
}

impl Default for MelSpectrogram {
    fn default() -> Self {
        Self::new()
    }
}

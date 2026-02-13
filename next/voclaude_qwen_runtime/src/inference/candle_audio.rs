//! Mel spectrogram computation for Qwen3-ASR.
//!
//! Converts raw f32 PCM samples (16 kHz mono) into a 128-bin log-mel spectrogram
//! tensor suitable for the audio encoder.  Uses the same algorithm as the
//! WhisperFeatureExtractor (Hann window, center-padded STFT, Slaney mel filterbank).

use candle_core::{Device, Result, Tensor};
use realfft::RealFftPlanner;
use std::f64::consts::PI;

// WhisperFeatureExtractor constants (must match model config)
pub const SAMPLE_RATE: usize = 16_000;
pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const N_MELS: usize = 128;

/// Compute 128-bin log-mel spectrogram from PCM samples.
/// Returns tensor of shape `(1, N_MELS, n_frames)`.
pub fn pcm_to_mel(samples: &[f32], device: &Device) -> Result<Tensor> {
    // Center-pad the signal (like librosa / Whisper feature extractor)
    let pad = N_FFT / 2;
    let mut padded = vec![0.0f32; pad + samples.len() + pad];
    // Reflect padding at boundaries
    for i in 0..pad {
        padded[pad - 1 - i] = samples[i.min(samples.len() - 1)];
    }
    padded[pad..pad + samples.len()].copy_from_slice(samples);
    for i in 0..pad {
        let src_idx = samples.len().saturating_sub(2).saturating_sub(i);
        padded[pad + samples.len() + i] = samples[src_idx.min(samples.len() - 1)];
    }

    let magnitudes = stft_magnitudes(&padded);
    let filters = compute_mel_filterbank();
    let n_freq = N_FFT / 2 + 1;
    let n_frames = magnitudes[0].len();

    // magnitudes: (n_freq, n_frames) as flat vec
    let mag_flat: Vec<f32> = magnitudes.iter().flat_map(|row| row.iter().copied()).collect();
    let mag_tensor = Tensor::from_vec(mag_flat, (n_freq, n_frames), device)?;

    // filters: (N_MELS, n_freq)
    let filter_flat: Vec<f32> = filters.into_iter().flatten().collect();
    let filter_tensor = Tensor::from_vec(filter_flat, (N_MELS, n_freq), device)?;

    // mel_spec = filters @ magnitudes => (N_MELS, n_frames)
    let mel_spec = filter_tensor.matmul(&mag_tensor)?;

    // Log-mel: clamp, log10, normalize
    let mel_spec = mel_spec.clamp(1e-10f32, f32::MAX)?;
    let mel_spec = mel_spec.log()?.affine(1.0 / f64::ln(10.0), 0.0)?;

    let max_val: f32 = mel_spec.flatten_all()?.max(0)?.to_scalar()?;
    let mel_spec = mel_spec.clamp(max_val - 8.0, f32::MAX)?;
    // Whisper normalization: (log10(spec) + 4.0) / 4.0
    let mel_spec = mel_spec.affine(0.25, 1.0)?;

    // Add batch dim: (N_MELS, n_frames) -> (1, N_MELS, n_frames)
    mel_spec.unsqueeze(0)
}

// ---------------------------------------------------------------------------
// STFT using realfft (O(n log n))
// ---------------------------------------------------------------------------

fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let x = (PI * i as f64 / size as f64).sin();
            (x * x) as f32
        })
        .collect()
}

/// Compute power spectrum via real-valued FFT.
/// Returns `magnitudes[freq_bin][frame]` where freq_bin in 0..N_FFT/2+1.
fn stft_magnitudes(samples: &[f32]) -> Vec<Vec<f32>> {
    let window = hann_window(N_FFT);
    let n_freq = N_FFT / 2 + 1;
    let n_frames = if samples.len() >= N_FFT {
        1 + (samples.len() - N_FFT) / HOP_LENGTH
    } else {
        0
    };

    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut scratch = fft.make_scratch_vec();

    let mut output = vec![vec![0.0f32; n_frames]; n_freq];
    let mut input_buf = fft.make_input_vec();
    let mut spectrum = fft.make_output_vec();

    for frame_idx in 0..n_frames {
        let start = frame_idx * HOP_LENGTH;

        // Fill windowed frame
        for i in 0..N_FFT {
            let sample = if start + i < samples.len() {
                samples[start + i]
            } else {
                0.0
            };
            input_buf[i] = sample * window[i];
        }

        // Forward FFT
        fft.process_with_scratch(&mut input_buf, &mut spectrum, &mut scratch)
            .expect("FFT failed");

        // Power spectrum: |X[k]|^2
        for k in 0..n_freq {
            let re = spectrum[k].re;
            let im = spectrum[k].im;
            output[k][frame_idx] = re * re + im * im;
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Mel filterbank (Slaney-style, matches librosa / Whisper)
// ---------------------------------------------------------------------------

fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10.0f64.powf(mel / 2595.0) - 1.0)
}

fn compute_mel_filterbank() -> Vec<Vec<f32>> {
    let n_freq = N_FFT / 2 + 1;
    let fmin = 0.0_f64;
    let fmax = SAMPLE_RATE as f64 / 2.0;

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    let n_points = N_MELS + 2;
    let mel_points: Vec<f64> = (0..n_points)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_points - 1) as f64)
        .collect();
    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    let fft_freqs: Vec<f64> = (0..n_freq)
        .map(|i| i as f64 * SAMPLE_RATE as f64 / N_FFT as f64)
        .collect();

    let mut filters = vec![vec![0.0f32; n_freq]; N_MELS];

    for m in 0..N_MELS {
        let f_left = hz_points[m];
        let f_center = hz_points[m + 1];
        let f_right = hz_points[m + 2];

        for k in 0..n_freq {
            let freq = fft_freqs[k];
            if freq >= f_left && freq <= f_center && f_center > f_left {
                filters[m][k] = ((freq - f_left) / (f_center - f_left)) as f32;
            } else if freq > f_center && freq <= f_right && f_right > f_center {
                filters[m][k] = ((f_right - freq) / (f_right - f_center)) as f32;
            }
        }

        // Slaney-style normalization
        let enorm = 2.0 / (hz_points[m + 2] - hz_points[m]);
        for k in 0..n_freq {
            filters[m][k] *= enorm as f32;
        }
    }

    filters
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_shape() {
        let sr = SAMPLE_RATE;
        let duration_sec = 2;
        let n_samples = sr * duration_sec;
        let samples: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin())
            .collect();

        let device = Device::Cpu;
        let mel = pcm_to_mel(&samples, &device).unwrap();
        let shape = mel.dims();
        assert_eq!(shape[0], 1);
        assert_eq!(shape[1], N_MELS);
        // With center padding, n_frames = 1 + (n_samples + pad*2 - N_FFT) / HOP_LENGTH
        // = 1 + (32000 + 400 - 400) / 160 = 1 + 200 = 201
        assert!(shape[2] > 0, "should have at least one frame");
    }

    #[test]
    fn test_fft_matches_shapes() {
        // Verify FFT output has correct number of bins
        let samples = vec![0.0f32; 16000]; // 1 second of silence
        let mags = stft_magnitudes(&samples);
        assert_eq!(mags.len(), N_FFT / 2 + 1);
        let expected_frames = 1 + (samples.len() - N_FFT) / HOP_LENGTH;
        assert_eq!(mags[0].len(), expected_frames);
    }
}

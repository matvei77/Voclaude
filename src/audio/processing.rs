//! Shared audio processing helpers.

/// Convert interleaved samples into mono f32 samples.
pub fn mono_from_interleaved<T, F>(data: &[T], channels: usize, to_f32: F) -> Vec<f32>
where
    T: Copy,
    F: Fn(T) -> f32 + Copy,
{
    if data.is_empty() || channels == 0 {
        return Vec::new();
    }

    if channels == 1 {
        return data.iter().map(|&sample| to_f32(sample)).collect();
    }

    data.chunks(channels)
        .map(|frame| {
            let sum: f32 = frame.iter().map(|&sample| to_f32(sample)).sum();
            sum / channels as f32
        })
        .collect()
}

/// Simple linear resampling from source_rate to target_rate.
pub fn resample_linear(samples: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if source_rate == target_rate {
        return samples.to_vec();
    }

    if samples.is_empty() || source_rate == 0 || target_rate == 0 {
        return Vec::new();
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

#[cfg(test)]
mod tests {
    use super::{mono_from_interleaved, resample_linear};

    #[test]
    fn mono_from_interleaved_stereo() {
        let input = [0.0_f32, 1.0_f32, 2.0_f32, 3.0_f32];
        let mono = mono_from_interleaved(&input, 2, |v| v);
        assert_eq!(mono, vec![0.5, 2.5]);
    }

    #[test]
    fn resample_linear_upsample() {
        let input = [0.0_f32, 1.0_f32];
        let output = resample_linear(&input, 2, 4);
        assert_eq!(output.len(), 4);
        assert!((output[1] - 0.5).abs() < 1e-6);
        assert!((output[3] - 1.0).abs() < 1e-6);
    }
}

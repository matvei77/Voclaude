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
#[allow(dead_code)]
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

/// Streaming linear resampler for incremental processing.
pub struct LinearResampler {
    ratio: f64,
    pos: f64,
    src_offset: u64,
    prev: Option<f32>,
}

impl LinearResampler {
    pub fn new(source_rate: u32, target_rate: u32) -> Self {
        let ratio = source_rate as f64 / target_rate as f64;
        Self {
            ratio,
            pos: 0.0,
            src_offset: 0,
            prev: None,
        }
    }

    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }

        if (self.ratio - 1.0).abs() < f64::EPSILON {
            return input.to_vec();
        }

        let start = self.src_offset as f64;
        let end = start + input.len() as f64;
        let estimate = ((input.len() as f64) / self.ratio).ceil() as usize + 2;
        let mut output = Vec::with_capacity(estimate);

        while self.pos + 1.0 < end {
            let idx = self.pos.floor() as u64;
            let frac = self.pos - idx as f64;

            let s0 = if idx < self.src_offset {
                self.prev.unwrap_or(input[0])
            } else {
                input[(idx - self.src_offset) as usize]
            };

            let s1_idx = idx + 1;
            let s1 = if s1_idx < self.src_offset {
                self.prev.unwrap_or(input[0])
            } else {
                input[(s1_idx - self.src_offset) as usize]
            };

            let sample = s0 + (s1 - s0) * frac as f32;
            output.push(sample);

            self.pos += self.ratio;
        }

        self.src_offset += input.len() as u64;
        self.prev = Some(*input.last().unwrap());

        output
    }
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

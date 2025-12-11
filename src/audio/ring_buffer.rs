//! Growable audio buffer for unlimited recording length.

use std::sync::Mutex;

/// Thread-safe growable buffer for audio samples (unlimited length)
pub struct RingBuffer {
    samples: Mutex<Vec<f32>>,
}

impl RingBuffer {
    /// Create a new growable buffer
    /// The capacity parameter is used as initial capacity hint
    pub fn new(initial_capacity: usize) -> Self {
        Self {
            samples: Mutex::new(Vec::with_capacity(initial_capacity)),
        }
    }

    /// Push samples into the buffer (called from audio callback)
    /// Returns the number of samples pushed (always all of them)
    pub fn push(&self, new_samples: &[f32]) -> usize {
        if let Ok(mut samples) = self.samples.lock() {
            samples.extend_from_slice(new_samples);
            new_samples.len()
        } else {
            0
        }
    }

    /// Pop all available samples from the buffer
    pub fn pop_all(&self) -> Vec<f32> {
        if let Ok(mut samples) = self.samples.lock() {
            std::mem::take(&mut *samples)
        } else {
            Vec::new()
        }
    }

    /// Clear the buffer
    pub fn clear(&self) {
        if let Ok(mut samples) = self.samples.lock() {
            samples.clear();
        }
    }

    /// Get the number of samples available
    pub fn len(&self) -> usize {
        if let Ok(samples) = self.samples.lock() {
            samples.len()
        } else {
            0
        }
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get recording duration in seconds
    pub fn duration_secs(&self, sample_rate: u32) -> f32 {
        self.len() as f32 / sample_rate as f32
    }
}

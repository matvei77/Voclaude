//! Growable audio buffer for unlimited recording length.

use std::sync::{Mutex, PoisonError};
use tracing::error;

/// Error type for ring buffer operations
#[derive(Debug)]
pub struct BufferError(String);

impl std::fmt::Display for BufferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Buffer error: {}", self.0)
    }
}

impl std::error::Error for BufferError {}

impl<T> From<PoisonError<T>> for BufferError {
    fn from(e: PoisonError<T>) -> Self {
        BufferError(format!("Mutex poisoned: {}", e))
    }
}

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
    /// Returns the number of samples pushed
    ///
    /// Note: In the audio callback, we can't propagate errors easily,
    /// so we log and return 0 on failure. The main thread should check
    /// for buffer errors via pop_all().
    pub fn push(&self, new_samples: &[f32]) -> usize {
        match self.samples.lock() {
            Ok(mut samples) => {
                samples.extend_from_slice(new_samples);
                new_samples.len()
            }
            Err(e) => {
                // Log error - this is serious but we can't panic in audio callback
                error!("CRITICAL: Audio buffer mutex poisoned, samples lost: {}", e);
                0
            }
        }
    }

    /// Pop all available samples from the buffer
    /// Returns error if mutex is poisoned (indicates audio thread panicked)
    pub fn pop_all(&self) -> Result<Vec<f32>, BufferError> {
        let mut samples = self.samples.lock()?;
        Ok(std::mem::take(&mut *samples))
    }

    /// Clear the buffer
    pub fn clear(&self) {
        match self.samples.lock() {
            Ok(mut samples) => samples.clear(),
            Err(e) => error!("Failed to clear buffer: {}", e),
        }
    }

    /// Get the number of samples available
    pub fn len(&self) -> usize {
        match self.samples.lock() {
            Ok(samples) => samples.len(),
            Err(e) => {
                error!("Failed to get buffer length: {}", e);
                0
            }
        }
    }

    /// Check if buffer is empty
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get recording duration in seconds
    #[allow(dead_code)]
    pub fn duration_secs(&self, sample_rate: u32) -> f32 {
        self.len() as f32 / sample_rate as f32
    }
}

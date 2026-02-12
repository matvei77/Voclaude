//! Audio capture with lock-free ring buffer.

mod capture;
mod processing;
mod ring_buffer;

pub use capture::{AudioCapture, TARGET_SAMPLE_RATE};
pub use processing::{mono_from_interleaved, resample_linear, LinearResampler};
pub use ring_buffer::RingBuffer;

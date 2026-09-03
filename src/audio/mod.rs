//! Audio capture with lock-free ring buffer.

mod capture;
mod processing;
mod ring_buffer;
pub mod segmenter;

pub use capture::{AudioCapture, AudioRecording, SegmentReady, TARGET_SAMPLE_RATE};
pub use segmenter::{SegmentBounds, SegmenterConfig};
pub use processing::{mono_from_interleaved, LinearResampler};
pub use ring_buffer::RingBuffer;

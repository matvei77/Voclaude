//! Audio capture with lock-free ring buffer.

mod capture;
mod ring_buffer;

pub use capture::AudioCapture;
pub use ring_buffer::RingBuffer;

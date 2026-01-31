//! Lock-free single-producer/single-consumer ring buffer for audio samples.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};

/// SPSC ring buffer for f32 samples.
pub struct RingBuffer {
    buffer: Box<[UnsafeCell<f32>]>,
    capacity: usize,
    mask: usize,
    head: AtomicUsize,
    tail: AtomicUsize,
}

unsafe impl Send for RingBuffer {}
unsafe impl Sync for RingBuffer {}

impl RingBuffer {
    pub fn new(min_capacity: usize) -> Self {
        let capacity = min_capacity.max(2).next_power_of_two();
        let mut buf = Vec::with_capacity(capacity);
        buf.resize_with(capacity, || UnsafeCell::new(0.0));
        Self {
            buffer: buf.into_boxed_slice(),
            capacity,
            mask: capacity - 1,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        head.wrapping_sub(tail)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&self) {
        let head = self.head.load(Ordering::Relaxed);
        self.tail.store(head, Ordering::Release);
    }

    pub fn push_slice(&self, input: &[f32]) -> usize {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        let used = head.wrapping_sub(tail);
        let free = self.capacity.saturating_sub(used);
        let write_len = input.len().min(free);

        for i in 0..write_len {
            let idx = (head + i) & self.mask;
            unsafe {
                *self.buffer.get_unchecked(idx).get() = input[i];
            }
        }

        self.head.store(head.wrapping_add(write_len), Ordering::Release);
        write_len
    }

    pub fn push_mapped<T, F>(&self, input: &[T], map: F) -> usize
    where
        T: Copy,
        F: Fn(T) -> f32 + Copy,
    {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        let used = head.wrapping_sub(tail);
        let free = self.capacity.saturating_sub(used);
        let write_len = input.len().min(free);

        for i in 0..write_len {
            let idx = (head + i) & self.mask;
            unsafe {
                *self.buffer.get_unchecked(idx).get() = map(input[i]);
            }
        }

        self.head.store(head.wrapping_add(write_len), Ordering::Release);
        write_len
    }

    pub fn pop_slice(&self, output: &mut [f32]) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Relaxed);
        let available = head.wrapping_sub(tail);
        let read_len = output.len().min(available);

        for i in 0..read_len {
            let idx = (tail + i) & self.mask;
            output[i] = unsafe { *self.buffer.get_unchecked(idx).get() };
        }

        self.tail.store(tail.wrapping_add(read_len), Ordering::Release);
        read_len
    }
}

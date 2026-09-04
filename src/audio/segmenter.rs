//! Incremental pause-based segmenter for live transcription.
//!
//! The recorder feeds 16 kHz mono samples in as they arrive. Once a segment is
//! at least `min_secs` long, the first pause of `pause_secs` closes it (the cut
//! lands in the middle of the pause). If no pause shows up, the segment is cut
//! at the quietest frame of the last few seconds once `max_secs` is reached.
//! The same code segments a finished file for crash recovery.

/// Sample rate the segmenter assumes (matches the recorder output).
pub const SEGMENTER_SAMPLE_RATE: usize = 16_000;

/// Analysis frame: 50 ms.
const FRAME_LEN: usize = SEGMENTER_SAMPLE_RATE / 20;

/// Window (in frames) inspected for the quietest point on a forced cut.
const FORCED_CUT_LOOKBACK_FRAMES: usize = 60; // 3 s

/// Absolute floor for the speech threshold (~-54 dBFS RMS).
const MIN_THRESHOLD: f32 = 0.002;
/// Ceiling for the speech threshold (~-34 dBFS RMS) so a noisy floor can't
/// swallow quiet speech.
const MAX_THRESHOLD: f32 = 0.02;
/// Speech must be this many times louder than the tracked noise floor.
const FLOOR_RATIO: f32 = 3.0;

#[derive(Debug, Clone, Copy)]
pub struct SegmenterConfig {
    pub min_secs: f32,
    pub max_secs: f32,
    pub pause_secs: f32,
}

impl Default for SegmenterConfig {
    fn default() -> Self {
        Self {
            min_secs: 20.0,
            max_secs: 45.0,
            pause_secs: 0.5,
        }
    }
}

/// A closed segment: sample offsets `[start, end)` into the recording.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentBounds {
    pub start: usize,
    pub end: usize,
}

impl SegmentBounds {
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }
}

pub struct Segmenter {
    min_samples: usize,
    max_samples: usize,
    pause_frames: usize,
    /// Samples not yet forming a full analysis frame.
    partial: Vec<f32>,
    /// Total samples consumed (frame-aligned; `partial` is not included).
    consumed: usize,
    /// Start of the open segment.
    seg_start: usize,
    /// Consecutive quiet frames at the end of the consumed stream.
    silence_run: usize,
    /// Tracked noise floor (RMS).
    noise_floor: f32,
    /// Recent frame RMS values (ring) for the forced-cut search.
    recent_rms: Vec<f32>,
    recent_pos: usize,
}

impl Segmenter {
    pub fn new(cfg: SegmenterConfig) -> Self {
        let min_secs = cfg.min_secs.max(1.0);
        let max_secs = cfg.max_secs.max(min_secs + 1.0);
        Self {
            min_samples: (min_secs * SEGMENTER_SAMPLE_RATE as f32) as usize,
            max_samples: (max_secs * SEGMENTER_SAMPLE_RATE as f32) as usize,
            pause_frames: ((cfg.pause_secs.max(0.1) * SEGMENTER_SAMPLE_RATE as f32) as usize / FRAME_LEN).max(1),
            partial: Vec::with_capacity(FRAME_LEN),
            consumed: 0,
            seg_start: 0,
            silence_run: 0,
            noise_floor: MIN_THRESHOLD,
            recent_rms: vec![0.0; FORCED_CUT_LOOKBACK_FRAMES],
            recent_pos: 0,
        }
    }

    /// Feed samples. Returns every segment closed by this call, in order.
    pub fn push(&mut self, samples: &[f32]) -> Vec<SegmentBounds> {
        let mut closed = Vec::new();
        let mut idx = 0;
        while idx < samples.len() {
            let need = FRAME_LEN - self.partial.len();
            let take = need.min(samples.len() - idx);
            self.partial.extend_from_slice(&samples[idx..idx + take]);
            idx += take;
            if self.partial.len() == FRAME_LEN {
                let rms = frame_rms(&self.partial);
                self.partial.clear();
                if let Some(seg) = self.consume_frame(rms) {
                    closed.push(seg);
                }
            }
        }
        closed
    }

    /// Close whatever is open. Returns `None` if nothing is left.
    pub fn finish(&mut self) -> Option<SegmentBounds> {
        let end = self.consumed + self.partial.len();
        self.partial.clear();
        self.consumed = end;
        if end > self.seg_start {
            let seg = SegmentBounds { start: self.seg_start, end };
            self.seg_start = end;
            self.silence_run = 0;
            Some(seg)
        } else {
            None
        }
    }

    /// Total samples fed so far.
    #[allow(dead_code)]
    pub fn position(&self) -> usize {
        self.consumed + self.partial.len()
    }

    fn consume_frame(&mut self, rms: f32) -> Option<SegmentBounds> {
        self.consumed += FRAME_LEN;
        self.recent_rms[self.recent_pos] = rms;
        self.recent_pos = (self.recent_pos + 1) % FORCED_CUT_LOOKBACK_FRAMES;

        // Noise floor: drops immediately, rises slowly.
        if rms < self.noise_floor {
            self.noise_floor = rms.max(1e-5);
        } else {
            self.noise_floor += (rms - self.noise_floor) * 0.003;
        }
        let threshold = (self.noise_floor * FLOOR_RATIO).clamp(MIN_THRESHOLD, MAX_THRESHOLD);

        if rms < threshold {
            self.silence_run += 1;
        } else {
            self.silence_run = 0;
        }

        let seg_len = self.consumed - self.seg_start;
        if seg_len >= self.min_samples && self.silence_run >= self.pause_frames {
            // Cut inside the pause, half a pause length back from now. In a
            // long silence this keeps the cut near the current position
            // instead of drifting back to the middle of the whole quiet run.
            let pause_samples = self.silence_run.min(self.pause_frames) * FRAME_LEN;
            let cut = self.consumed - pause_samples / 2;
            let cut = cut.max(self.seg_start + FRAME_LEN);
            let seg = SegmentBounds { start: self.seg_start, end: cut };
            self.seg_start = cut;
            self.silence_run = 0;
            return Some(seg);
        }

        if seg_len >= self.max_samples {
            // No pause found: cut at the quietest frame in the lookback window.
            let lookback = FORCED_CUT_LOOKBACK_FRAMES.min(seg_len / FRAME_LEN).max(1);
            let mut best_frames_back = 0;
            let mut best_rms = f32::MAX;
            for back in 0..lookback {
                let pos = (self.recent_pos + FORCED_CUT_LOOKBACK_FRAMES - 1 - back) % FORCED_CUT_LOOKBACK_FRAMES;
                let v = self.recent_rms[pos];
                if v < best_rms {
                    best_rms = v;
                    best_frames_back = back;
                }
            }
            let cut = self.consumed - best_frames_back * FRAME_LEN;
            let cut = cut.max(self.seg_start + FRAME_LEN);
            let seg = SegmentBounds { start: self.seg_start, end: cut };
            self.seg_start = cut;
            self.silence_run = 0;
            return Some(seg);
        }
        None
    }
}

fn frame_rms(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    let sum: f32 = frame.iter().map(|s| s * s).sum();
    (sum / frame.len() as f32).sqrt()
}

/// Segment a complete recording (used for crash recovery).
pub fn segment_all(samples: &[f32], cfg: SegmenterConfig) -> Vec<SegmentBounds> {
    let mut seg = Segmenter::new(cfg);
    let mut out = Vec::new();
    for chunk in samples.chunks(4096) {
        out.extend(seg.push(chunk));
    }
    if let Some(tail) = seg.finish() {
        out.push(tail);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(secs: f32, amp: f32) -> Vec<f32> {
        let n = (secs * SEGMENTER_SAMPLE_RATE as f32) as usize;
        (0..n)
            .map(|i| amp * ((i as f32 * 0.05).sin()))
            .collect()
    }

    #[test]
    fn cuts_at_first_pause_after_min() {
        let cfg = SegmenterConfig { min_secs: 2.0, max_secs: 10.0, pause_secs: 0.5 };
        let mut audio = tone(3.0, 0.3);
        audio.extend(tone(1.0, 0.0)); // 1 s pause
        audio.extend(tone(3.0, 0.3));
        let segs = segment_all(&audio, cfg);
        assert_eq!(segs.len(), 2, "{:?}", segs);
        // First cut should land inside the pause (3.0 s .. 4.0 s).
        let cut = segs[0].end as f32 / SEGMENTER_SAMPLE_RATE as f32;
        assert!(cut > 3.0 && cut < 4.0, "cut at {}", cut);
        assert_eq!(segs[1].end, audio.len());
        assert_eq!(segs[0].end, segs[1].start);
    }

    #[test]
    fn forced_cut_without_pause() {
        let cfg = SegmenterConfig { min_secs: 2.0, max_secs: 4.0, pause_secs: 0.5 };
        let audio = tone(9.0, 0.3);
        let segs = segment_all(&audio, cfg);
        assert!(segs.len() >= 2, "{:?}", segs);
        for s in &segs {
            assert!(s.len() <= (4.1 * SEGMENTER_SAMPLE_RATE as f32) as usize);
        }
        assert_eq!(segs.last().unwrap().end, audio.len());
    }

    #[test]
    fn short_audio_is_one_segment() {
        let cfg = SegmenterConfig::default();
        let audio = tone(1.5, 0.2);
        let segs = segment_all(&audio, cfg);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0], SegmentBounds { start: 0, end: audio.len() });
    }

    #[test]
    fn empty_audio_has_no_segments() {
        assert!(segment_all(&[], SegmenterConfig::default()).is_empty());
    }

    #[test]
    fn incremental_matches_batch() {
        let cfg = SegmenterConfig { min_secs: 2.0, max_secs: 6.0, pause_secs: 0.4 };
        let mut audio = Vec::new();
        for _ in 0..4 {
            audio.extend(tone(2.5, 0.25));
            audio.extend(tone(0.6, 0.0));
        }
        let batch = segment_all(&audio, cfg);
        let mut seg = Segmenter::new(cfg);
        let mut inc = Vec::new();
        for chunk in audio.chunks(777) {
            inc.extend(seg.push(chunk));
        }
        if let Some(t) = seg.finish() {
            inc.push(t);
        }
        assert_eq!(batch, inc);
    }
}

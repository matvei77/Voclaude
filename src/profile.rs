//! Opt-in profiling of the `--worker` child: `VOCLAUDE_WORKER_PROFILE=1`.
//!
//! When set, the worker logs at DEBUG instead of INFO (to stderr, as always)
//! and writes one machine-readable record per request:
//!
//!     voclaude-profile {"kind":"parakeet_segment","samples":...,"first_ms":...}
//!
//! Records carry sample counts, times, flags and edit counts, never transcript
//! text. Unset (the default), the worker behaves exactly as before.

use std::sync::OnceLock;

/// True when `VOCLAUDE_WORKER_PROFILE` is set to anything but empty or "0" (read once).
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VOCLAUDE_WORKER_PROFILE").map(|v| !v.is_empty() && v != "0").unwrap_or(false))
}

/// Write one record (a JSON object) if profiling is enabled.
pub fn emit(record: serde_json::Value) {
    if enabled() {
        tracing::info!("voclaude-profile {}", record);
    }
}

/// Milliseconds since `t`, rounded to 0.1 ms.
pub fn ms_since(t: std::time::Instant) -> f64 {
    (t.elapsed().as_secs_f64() * 10_000.0).round() / 10.0
}

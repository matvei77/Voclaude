//! Smoke test for the transcribe.cpp runtime: list devices, load a GGUF model
//! on one device and transcribe a raw 16 kHz mono f32 file.
//!
//!     cargo run --release --example parakeet_smoke -- <model.gguf> <audio.f32> [vulkan|cpu]

use std::time::Instant;
use transcribe_cpp::{devices, init_backends_default, Model, ModelOptions, RunOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let (model_path, audio_path) = (&args[1], &args[2]);
    let kind = args.get(3).map(String::as_str).unwrap_or("vulkan");

    init_backends_default()?;
    for d in devices() {
        println!("device: {} [{}] {} ({} MB free)", d.name, d.kind, d.description, d.memory_free >> 20);
    }
    let device = devices().into_iter().find(|d| d.kind == kind).ok_or("no such device")?;

    let t = Instant::now();
    let model = Model::load_with(model_path, &ModelOptions { device: Some(device), ..Default::default() })?;
    let caps = model.capabilities()?;
    println!("model: {} on {} in {:.1}s, max_audio_ms={}, {} languages",
        model.arch(), model.backend(), t.elapsed().as_secs_f64(), caps.max_audio_ms, caps.languages.len());

    let bytes = std::fs::read(audio_path)?;
    let pcm: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let mut session = model.session()?;
    let secs = pcm.len() as f64 / 16000.0;
    for _ in 0..2 {
        let t = Instant::now();
        let r = session.run(&pcm, &RunOptions::default())?;
        let wall = t.elapsed().as_secs_f64();
        println!("\n{:.1}s audio in {:.3}s ({:.0}x), encode {:.0} ms, language {:?}\n{}",
            secs, wall, secs / wall, r.timings.encode_ms, r.language, r.text.trim());
    }
    Ok(())
}

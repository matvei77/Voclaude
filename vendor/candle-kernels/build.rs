//! Vendored candle-kernels 0.9.2 build script, minus the MoE kernels.
//!
//! Upstream also compiles `src/moe/*.cu` into `libmoe.a`; those kernels use
//! bf16 tensor-core fragments that only exist from compute capability 8.0, so
//! the build fails when targeting 7.5 (GTX 16xx / RTX 20xx). Voclaude never
//! runs a mixture-of-experts model, so the FFI entry points are stubbed in
//! `src/ffi.rs` and only the PTX modules are built here.
use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");
    let builder = bindgen_cuda::Builder::default()
        .kernel_paths_glob("src/*.cu")
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");
    let bindings = builder.build_ptx().unwrap();
    bindings.write(&ptx_path).unwrap();
}

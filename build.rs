fn main() {
    // Link cudart_static for CUDA 12+ API (cudaGetDeviceProperties_v2)
    // Required because whisper.cpp uses cuda runtime functions not in cudart.lib
    println!("cargo:rustc-link-search=native=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/lib/x64");
    println!("cargo:rustc-link-lib=cudart_static");
}

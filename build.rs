fn main() {
    // Link cudart_static for CUDA 12+ API (cudaGetDeviceProperties_v2)
    // Required because whisper.cpp uses cuda runtime functions not in cudart.lib

    #[cfg(target_os = "windows")]
    {
        println!("cargo:rustc-link-search=native=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/lib/x64");
        println!("cargo:rustc-link-lib=cudart_static");
    }

    #[cfg(target_os = "linux")]
    {
        // Standard CUDA installation paths on Linux
        let cuda_paths = [
            "/usr/local/cuda/lib64",       // Default NVIDIA installer location
            "/usr/lib64",                   // Fedora/RHEL system packages
            "/usr/lib/x86_64-linux-gnu",   // Debian/Ubuntu system packages
            "/opt/cuda/lib64",              // Alternative location
        ];

        let mut cuda_found = false;
        for path in &cuda_paths {
            let cudart_path = format!("{}/libcudart.so", path);
            if std::path::Path::new(&cudart_path).exists() {
                println!("cargo:rustc-link-search=native={}", path);
                println!("cargo:rustc-link-lib=cudart");
                cuda_found = true;
                eprintln!("Found CUDA at: {}", path);
                break;
            }
        }

        if !cuda_found {
            // CUDA not found - whisper-rs will handle this gracefully
            // and fall back to CPU inference
            eprintln!("Warning: CUDA not found, GPU acceleration will be unavailable");
        }
    }
}

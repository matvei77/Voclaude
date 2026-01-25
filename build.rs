use std::env;

fn main() {
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-search=native=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/lib/x64");
    } else {
        let cuda_root = env::var("CUDA_PATH").or_else(|_| env::var("CUDA_HOME"));
        if let Ok(cuda_root) = cuda_root {
            println!("cargo:rustc-link-search=native={}/lib64", cuda_root);
        } else {
            println!("cargo:warning=CUDA feature enabled but CUDA_PATH/CUDA_HOME not set; relying on system linker paths");
        }
    }

    // CUDA runtime linkage for whisper.cpp on Windows.
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cudart_static");
    println!("cargo:rustc-link-lib=cudadevrt");
    println!("cargo:rustc-link-lib=cuda");
}

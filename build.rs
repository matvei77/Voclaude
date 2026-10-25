/// Build script: embeds git hash into the binary via VOCLAUDE_GIT_HASH env var.
fn main() {
    // Get git hash for version stamping
    let git_hash = std::process::Command::new("git")
        .args(["rev-parse", "--short=8", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout).ok().map(|s| s.trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=VOCLAUDE_GIT_HASH={}", git_hash);

    // Re-run if git HEAD changes (new commits)
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/refs/heads");
    println!("cargo:rerun-if-changed=.git/packed-refs");

    // candle-kernels links `cudart` for its MoE kernels; with cudarc in
    // dynamic-loading mode nobody else adds the CUDA library directory to the
    // linker search path, so do it here (harmless on the CPU build).
    if let Ok(cuda) = std::env::var("CUDA_PATH") {
        println!("cargo:rustc-link-search=native={}/lib/x64", cuda);
    }
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    // transcribe.dll (and the ggml DLLs it pulls in) load on the first
    // transcribe.cpp call, which only happens in a `--worker` child running a
    // GGUF model. The tray process never maps them, and a missing DLL is an
    // engine error instead of a launch failure.
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        println!("cargo:rustc-link-arg=/DELAYLOAD:transcribe.dll");
        println!("cargo:rustc-link-arg=delayimp.lib");
    }

    // Application icon for Explorer, the taskbar and the installer (Windows only).
    #[cfg(target_os = "windows")]
    {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/icon.ico");
        res.set("ProductName", "Voclaude");
        res.set("FileDescription", "Voclaude voice input");
        if let Err(e) = res.compile() {
            println!("cargo:warning=icon not embedded: {}", e);
        }
        println!("cargo:rerun-if-changed=assets/icon.ico");
    }
}

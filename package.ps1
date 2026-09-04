# package.ps1 - Build Voclaude and package it with CUDA DLLs for distribution.
#
# Usage:
#   .\package.ps1            # GPU build (default, bundles CUDA runtime DLLs)
#   .\package.ps1 -Cpu       # CPU-only build (no CUDA DLLs, smaller)
#
# Output:
#   dist\voclaude-gpu.zip    or    dist\voclaude-cpu.zip

param(
    [switch]$Cpu
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$DistDir     = Join-Path $ProjectRoot "dist"
$Version     = (Select-String -Path (Join-Path $ProjectRoot "Cargo.toml") -Pattern '^version\s*=\s*"(.+)"' |
                Select-Object -First 1).Matches.Groups[1].Value
$GitHash     = (git -C $ProjectRoot rev-parse --short=8 HEAD 2>$null)
if (-not $GitHash) { $GitHash = "unknown" }

# Build CUDA kernels for compute capability 7.5 (GTX 16xx / RTX 20xx) unless
# told otherwise; newer GPUs JIT the PTX. Without this, candle-kernels asks
# nvidia-smi and targets only the build machine's GPU generation.
if (-not $env:CUDA_COMPUTE_CAP) { $env:CUDA_COMPUTE_CAP = "75" }

if ($Cpu) {
    $Variant = "cpu"
    Write-Host "Building Voclaude v$Version (CPU-only)..." -ForegroundColor Cyan
    cargo build --release --manifest-path "$ProjectRoot\Cargo.toml" --no-default-features --features cpu
} else {
    $Variant = "gpu"
    Write-Host "Building Voclaude v$Version (GPU/CUDA)..." -ForegroundColor Cyan
    cargo build --release --manifest-path "$ProjectRoot\Cargo.toml"
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed." -ForegroundColor Red
    exit 1
}

$Exe = Join-Path $ProjectRoot "target\release\voclaude.exe"
if (-not (Test-Path $Exe)) {
    Write-Host "Binary not found at $Exe" -ForegroundColor Red
    exit 1
}

# Stage files
$StageDir = Join-Path $DistDir "voclaude-$Variant"
if (Test-Path $StageDir) { Remove-Item -Recurse -Force $StageDir }
New-Item -ItemType Directory -Force -Path $StageDir | Out-Null

Copy-Item $Exe $StageDir

# For GPU builds, bundle the required CUDA DLLs
if (-not $Cpu) {
    # DLLs we need (nvcuda.dll ships with NVIDIA drivers, no need to bundle).
    # Keep this broader than the current import table because NVIDIA DLLs can
    # depend on sibling runtime DLLs depending on CUDA version.
    $CudaDlls = @(
        "cublas64_*.dll",
        "cublasLt64_*.dll",
        "curand64_*.dll",
        "cudart64_*.dll",
        "nvrtc64_*.dll"
    )

    # Search paths: CUDA_PATH, then common install locations
    $SearchRoots = @()
    if ($env:CUDA_PATH) { $SearchRoots += "$env:CUDA_PATH\bin" }
    $SearchRoots += "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    # Also check PATH
    $SearchRoots += ($env:PATH -split ';' | Where-Object { $_ -match 'CUDA|nvidia' })

    $Found = @{}
    foreach ($pattern in $CudaDlls) {
        foreach ($root in $SearchRoots) {
            if (-not (Test-Path $root)) { continue }
            $match = Get-ChildItem -Path $root -Filter $pattern -Recurse -ErrorAction SilentlyContinue |
                     Select-Object -First 1
            if ($match) {
                $Found[$pattern] = $match.FullName
                break
            }
        }
    }

    $Missing = $CudaDlls | Where-Object { -not $Found.ContainsKey($_) }
    if ($Missing) {
        Write-Host "`nWARNING: Could not find these CUDA DLLs:" -ForegroundColor Yellow
        $Missing | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow }
        Write-Host "The package may not run on machines without the CUDA Toolkit." -ForegroundColor Yellow
        Write-Host "Install CUDA Toolkit or set CUDA_PATH to fix this.`n" -ForegroundColor Yellow
    }

    # S-4: Verify CUDA DLL major version matches the build environment to prevent
    # silent crashes from mismatched cublas/candle ABI.
    $ExpectedCudaMajor = $null
    if ($env:CUDA_PATH -match 'v(\d+)\.\d+') {
        $ExpectedCudaMajor = $Matches[1]
    } elseif ((Get-Command nvcc -ErrorAction SilentlyContinue)) {
        $nvccOut = nvcc --version 2>&1 | Select-String 'release (\d+)\.\d+'
        if ($nvccOut) { $ExpectedCudaMajor = $nvccOut.Matches.Groups[1].Value }
    }

    foreach ($entry in $Found.GetEnumerator()) {
        $dll = Split-Path -Leaf $entry.Value
        # Check DLL version number matches expected CUDA major version
        if ($ExpectedCudaMajor -and $dll -match '_(\d+)\.dll$') {
            $dllMajor = $Matches[1]
            if ($dllMajor -ne $ExpectedCudaMajor) {
                Write-Host "  WARNING: $dll is CUDA $dllMajor but build environment is CUDA $ExpectedCudaMajor - ABI mismatch!" -ForegroundColor Red
            }
        }
        Copy-Item $entry.Value $StageDir
        Write-Host "  Bundled $dll" -ForegroundColor DarkGray
    }

    # MSVC runtime is frequently missing on clean corporate Windows images.
    # Bundle it when present so users do not have to install VC++ redistributables.
    $MsvcDlls = @("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll")
    $MsvcRoots = @(
        "$env:WINDIR\System32",
        "$env:WINDIR\SysWOW64"
    ) + ($env:PATH -split ';')
    foreach ($dllName in $MsvcDlls) {
        $dllPath = $null
        foreach ($root in $MsvcRoots) {
            if (-not $root -or -not (Test-Path $root)) { continue }
            $candidate = Join-Path $root $dllName
            if (Test-Path $candidate) {
                $dllPath = $candidate
                break
            }
        }
        if ($dllPath) {
            Copy-Item $dllPath $StageDir
            Write-Host "  Bundled $dllName" -ForegroundColor DarkGray
        } else {
            Write-Host "  WARNING: Could not find $dllName" -ForegroundColor Yellow
        }
    }
}

# Copy example config
$ExampleConfig = Join-Path $ProjectRoot "config.example.toml"
if (Test-Path $ExampleConfig) {
    Copy-Item $ExampleConfig $StageDir
}

$ReadmePath = Join-Path $StageDir "README-FIRST.txt"
$ReadmeText = if ($Cpu) {
@"
Voclaude CPU build
==================

Run voclaude.exe. No Rust, Python, CUDA Toolkit, or NVIDIA GPU is required.

The model downloads on first transcription. For normal corporate laptops, use
model_tier = "medium" in config.toml so Voclaude uses Qwen3-ASR-0.6B.

Useful checks:
  voclaude.exe --validate --cpu --model-tier medium
  voclaude.exe --list-models
"@
} else {
@"
Voclaude GPU build
==================

Run voclaude.exe. Do NOT install the CUDA Toolkit on the user's machine.
This package bundles the CUDA runtime DLLs that Voclaude imports.

Requirements:
  - Windows 10/11 64-bit
  - NVIDIA GPU
  - Recent NVIDIA driver

Useful checks:
  voclaude.exe --validate --gpu --model-tier medium
  voclaude.exe --list-models

If validation says CUDA is unavailable, update the NVIDIA driver. The CUDA
Toolkit is a developer dependency, not an end-user install step.
"@
}
Set-Content -Path $ReadmePath -Value $ReadmeText -Encoding ASCII

# Create zip (include git hash for traceability)
$ZipName = "voclaude-v$Version-$GitHash-$Variant.zip"
$ZipPath = Join-Path $DistDir $ZipName
if (Test-Path $ZipPath) { Remove-Item -Force $ZipPath }

Write-Host "`nCompressing to $ZipName..." -ForegroundColor Cyan
Compress-Archive -Path "$StageDir\*" -DestinationPath $ZipPath

# Summary
$ZipSize = (Get-Item $ZipPath).Length / 1MB
$FileCount = (Get-ChildItem $StageDir).Count

Write-Host "`nPackaged $FileCount files -> $ZipName ($([math]::Round($ZipSize, 1)) MB)" -ForegroundColor Green
Write-Host "Location: $ZipPath" -ForegroundColor Green

# Cleanup staging dir
Remove-Item -Recurse -Force $StageDir

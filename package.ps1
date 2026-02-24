# package.ps1 — Build Voclaude and package it with CUDA DLLs for distribution.
#
# Usage:
#   .\package.ps1            # GPU build (default)
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
    # DLLs we need (nvcuda.dll ships with NVIDIA drivers, no need to bundle)
    $CudaDlls = @("cublas64_*.dll", "cublasLt64_*.dll", "curand64_*.dll")

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

    foreach ($entry in $Found.GetEnumerator()) {
        $dll = Split-Path -Leaf $entry.Value
        Copy-Item $entry.Value $StageDir
        Write-Host "  Bundled $dll" -ForegroundColor DarkGray
    }
}

# Copy example config
$ExampleConfig = Join-Path $ProjectRoot "config.example.toml"
if (Test-Path $ExampleConfig) {
    Copy-Item $ExampleConfig $StageDir
}

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

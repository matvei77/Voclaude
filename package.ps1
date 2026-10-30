# package.ps1 - Build Voclaude and package it for distribution.
#
# Usage:
#   .\package.ps1                    # standard build: Parakeet on Vulkan or CPU, Qwen on CPU
#   .\package.ps1 -Installer         # also build the Inno Setup installer
#   .\package.ps1 -Cuda -Installer   # CUDA build: adds the Qwen engine on NVIDIA GPUs
#
# Output (dist\):
#   voclaude-vX.Y.Z-<hash>.zip          voclaude-vX.Y.Z-setup.exe
#   voclaude-vX.Y.Z-<hash>-cuda.zip     voclaude-vX.Y.Z-cuda-setup.exe
#
# The standard build needs no CUDA toolkit. The CUDA build needs the toolkit,
# MSVC's cl.exe (found through vswhere) and bundles the CUDA runtime DLLs.
# Keep this file ASCII only: PowerShell 5.1 misreads UTF-8 without a BOM.

param(
    [switch]$Cuda,
    [switch]$Installer
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$DistDir     = Join-Path $ProjectRoot "dist"
$Version     = (Select-String -Path (Join-Path $ProjectRoot "Cargo.toml") -Pattern '^version\s*=\s*"(.+)"' |
                Select-Object -First 1).Matches.Groups[1].Value
$GitHash     = (git -C $ProjectRoot rev-parse --short=8 HEAD 2>$null)
if (-not $GitHash) { $GitHash = "unknown" }

if ($Cuda) {
    $Variant = "cuda"
    $Suffix  = "-cuda"
} else {
    $Variant = "standard"
    $Suffix  = ""
}

if ($Cuda) {
    # Build CUDA kernels for compute capability 7.5 (GTX 16xx / RTX 20xx) unless
    # told otherwise; newer GPUs JIT the PTX. Without this, candle-kernels asks
    # nvidia-smi and targets only the build machine's GPU generation.
    if (-not $env:CUDA_COMPUTE_CAP) { $env:CUDA_COMPUTE_CAP = "75" }

    # nvcc needs MSVC's cl.exe on PATH; find it through vswhere when the shell
    # is not a Visual Studio developer prompt.
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        $VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
        if (Test-Path $VsWhere) {
            $Cl = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find "VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe" | Select-Object -First 1
            if ($Cl) {
                $env:PATH = (Split-Path $Cl) + ";" + $env:PATH
                Write-Host "Using MSVC compiler: $Cl" -ForegroundColor DarkGray
            }
        }
    }
}

# Prebuilt transcribe.cpp (CPU + Vulkan) that the build links against.
& (Join-Path $ProjectRoot "fetch-transcribe.ps1")
if ($LASTEXITCODE -ne 0) { exit 1 }

if ($Cuda) {
    Write-Host "Building Voclaude v$Version (CUDA variant)..." -ForegroundColor Cyan
    cargo build --release --manifest-path "$ProjectRoot\Cargo.toml" --features cuda
} else {
    Write-Host "Building Voclaude v$Version (standard)..." -ForegroundColor Cyan
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

# transcribe.cpp runtime for the Parakeet engine: transcribe.dll, ggml, the
# per-ISA CPU modules and the Vulkan module (delay-loaded by the worker).
$TranscribeBin = Join-Path $ProjectRoot "vendor\transcribe-prebuilt\bin"
Copy-Item "$TranscribeBin\*.dll" $StageDir
New-Item -ItemType Directory -Force -Path (Join-Path $StageDir "licenses") | Out-Null
Copy-Item -Recurse (Join-Path $ProjectRoot "vendor\transcribe-prebuilt\licenses") (Join-Path $StageDir "licenses\transcribe.cpp")
Write-Host "  Bundled transcribe.cpp runtime ($((Get-ChildItem "$TranscribeBin\*.dll").Count) DLLs)" -ForegroundColor DarkGray

# CUDA variant: bundle the CUDA runtime DLLs the Qwen engine imports.
if ($Cuda) {
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
}

# MSVC runtime is frequently missing on clean corporate Windows images.
# Bundle it when present so users do not have to install VC++ redistributables.
# voclaude.exe and transcribe.dll both import it.
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

# Copy example config
$ExampleConfig = Join-Path $ProjectRoot "config.example.toml"
if (Test-Path $ExampleConfig) {
    Copy-Item $ExampleConfig $StageDir
}

$ReadmePath = Join-Path $StageDir "README-FIRST.txt"
$CudaNote = if ($Cuda) {
@"

This CUDA build also runs the optional Qwen3-ASR engine on NVIDIA GPUs
(engine = "qwen" in config.toml). The CUDA runtime DLLs are bundled. Do NOT
install the CUDA Toolkit; a 2025 or newer NVIDIA driver is enough.
"@
} else { "" }
$ReadmeText = @"
Voclaude $Version
==================

Run voclaude.exe. It lives in the system tray.
Press F4, speak, press F4 again, then paste with Ctrl+V.

Requirements:
  - Windows 10/11, 64-bit
  - a microphone
  - internet for the first run: the speech model (about 0.6 GB) downloads once
  - any GPU with a Vulkan driver makes it faster; without one it runs on the CPU

Settings: right-click the tray icon > Edit Settings (config.toml).
Checks from a terminal:
  voclaude.exe --validate
  voclaude.exe --list-models
$CudaNote
Documentation: https://github.com/matvei77/Voclaude
"@
Set-Content -Path $ReadmePath -Value $ReadmeText -Encoding ASCII

# Create zip (include git hash for traceability)
$ZipName = "voclaude-v$Version-$GitHash$Suffix.zip"
$ZipPath = Join-Path $DistDir $ZipName
if (Test-Path $ZipPath) { Remove-Item -Force $ZipPath }

Write-Host "`nCompressing to $ZipName..." -ForegroundColor Cyan
Compress-Archive -Path "$StageDir\*" -DestinationPath $ZipPath

# Summary
$ZipSize = (Get-Item $ZipPath).Length / 1MB
$FileCount = (Get-ChildItem $StageDir).Count

Write-Host "`nPackaged $FileCount files -> $ZipName ($([math]::Round($ZipSize, 1)) MB)" -ForegroundColor Green
Write-Host "Location: $ZipPath" -ForegroundColor Green

# One-click installer (Inno Setup 6). Per-user install, no admin prompt,
# optional "start with Windows", launches the app when done.
if ($Installer) {
    $IsccCandidates = @(
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
    )
    $Iscc = $IsccCandidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
    if (-not $Iscc) {
        Write-Host "Inno Setup 6 not found (winget install JRSoftware.InnoSetup); skipping installer." -ForegroundColor Yellow
    } else {
        Write-Host "`nBuilding installer..." -ForegroundColor Cyan
        & $Iscc /Q "/DAppVersion=$Version" "/DStageDir=$StageDir" "/DOutputDir=$DistDir" "/DVariant=$Variant" (Join-Path $ProjectRoot "installer\voclaude.iss")
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Installer build failed." -ForegroundColor Red
            exit 1
        }
        $SetupPath = Join-Path $DistDir "voclaude-v$Version$Suffix-setup.exe"
        $SetupSize = (Get-Item $SetupPath).Length / 1MB
        Write-Host "Installer: $SetupPath ($([math]::Round($SetupSize, 1)) MB)" -ForegroundColor Green
    }
}

# Cleanup staging dir
Remove-Item -Recurse -Force $StageDir

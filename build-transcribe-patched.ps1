# build-transcribe-patched.ps1 - Build transcribe.dll 0.4.1 with the Parakeet
# phrase-bias patch (RunOptions::vocabulary for Parakeet TDT), for
# parakeet_phrases. Only transcribe.dll is built; it runs with the official
# ggml DLLs from vendor\transcribe-prebuilt\bin. No ABI change.
#
# Source: the transcribe-cpp-sys 0.4.1 crate (cargo registry) plus
# docs\bench\intel-optimization\native\parakeet-phrase-bias-b3.patch.
# Flags: the official wheel-windows-cpu-vulkan lane without Vulkan.
# Needs: MSVC 2022 Build Tools, CMake and Ninja on PATH, git.
#
# Usage:
#   .\build-transcribe-patched.ps1          # once; output in vendor\transcribe-patched
#   .\build-transcribe-patched.ps1 -Force   # rebuild

param([switch]$Force)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$OutDir      = Join-Path $ProjectRoot "vendor\transcribe-patched"
$OutDll      = Join-Path $OutDir "transcribe.dll"
$Patch       = Join-Path $ProjectRoot "docs\bench\intel-optimization\native\parakeet-phrase-bias-b3.patch"
# Every build has a new hash: the MSVC linker writes a timestamp into the DLL,
# and source file paths are compiled in. Builds from this patch on the desktop
# (MSVC 14.43) gave transcription output byte-identical to the evaluated build
# B3 (4234e7ae...). Compare outputs, not bytes.

if ((Test-Path $OutDll) -and -not $Force) {
    Write-Host "Patched transcribe.dll already at $OutDll (use -Force to rebuild)" -ForegroundColor DarkGray
    exit 0
}

foreach ($tool in @("cmake", "ninja", "git")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "$tool not found on PATH" }
}

# Crate source from the cargo registry (fetch it if this machine never built Voclaude).
$Registry = Join-Path $env:USERPROFILE ".cargo\registry\src"
$Crate = Get-ChildItem $Registry -Directory -ErrorAction SilentlyContinue |
    ForEach-Object { Join-Path $_.FullName "transcribe-cpp-sys-0.4.1" } |
    Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Crate) {
    cargo fetch --manifest-path (Join-Path $ProjectRoot "Cargo.toml")
    $Crate = Get-ChildItem $Registry -Directory | ForEach-Object { Join-Path $_.FullName "transcribe-cpp-sys-0.4.1" } |
        Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $Crate) { throw "transcribe-cpp-sys 0.4.1 source not found in $Registry" }
}

$Work  = Join-Path ([IO.Path]::GetTempPath()) "voclaude-transcribe-patched"
$Src   = Join-Path $Work "src"
$Build = Join-Path $Work "build"
if (Test-Path $Work) { Remove-Item -Recurse -Force $Work }
New-Item -ItemType Directory -Force $Work | Out-Null
Copy-Item -Recurse $Crate $Src
Write-Host "Applying $Patch" -ForegroundColor Cyan
git -C $Src -c core.autocrlf=false apply --whitespace=nowarn $Patch
if ($LASTEXITCODE -ne 0) { throw "patch did not apply to $Crate" }

$VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$VsPath = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $VsPath) { throw "MSVC build tools not found" }
$VcVars = Join-Path $VsPath "VC\Auxiliary\Build\vcvars64.bat"

$Configure = "cmake -S `"$Src`" -B `"$Build`" -G Ninja -DCMAKE_BUILD_TYPE=Release " +
    "-DTRANSCRIBE_BUILD_SHARED=ON -DTRANSCRIBE_BUILD_TESTS=OFF -DTRANSCRIBE_BUILD_EXAMPLES=OFF " +
    "-DTRANSCRIBE_BUILD_TOOLS=OFF -DTRANSCRIBE_USE_OPENMP=OFF -DTRANSCRIBE_USE_SYSTEM_BLAS=OFF " +
    "-DTRANSCRIBE_GGML_BACKEND_DL=ON -DTRANSCRIBE_VULKAN=OFF -DGGML_CPU_ALL_VARIANTS=ON " +
    "-DTRANSCRIBE_X86_CONSERVATIVE=ON -DGGML_NATIVE=OFF"
Write-Host "Building transcribe.dll (about 1 minute)..." -ForegroundColor Cyan
cmd /c "`"$VcVars`" >nul && $Configure >nul && cmake --build `"$Build`" --target transcribe"
if ($LASTEXITCODE -ne 0) { throw "transcribe.dll build failed" }

New-Item -ItemType Directory -Force $OutDir | Out-Null
Copy-Item (Join-Path $Build "bin\transcribe.dll") $OutDll -Force
$Hash = (Get-FileHash $OutDll -Algorithm SHA256).Hash
Set-Content -Path (Join-Path $OutDir "transcribe.dll.sha256") -Value $Hash -Encoding ascii
Write-Host "Patched transcribe.dll: $OutDll sha256 $Hash" -ForegroundColor Green

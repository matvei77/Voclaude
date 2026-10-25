# fetch-transcribe.ps1 - Download the official transcribe.cpp Windows runtime
# (CPU + Vulkan) and lay it out as the install prefix that transcribe-cpp-sys
# links against (TRANSCRIBE_DIR, exported for this shell).
#
# The release ships DLLs only, so the import library is generated from the
# DLL's export table and the link manifest is written here. Building ggml's
# Vulkan backend from source would need the Vulkan SDK; the official DLLs are
# also the exact binaries the 2026-10-23 benchmark measured.
#
# Usage:
#   .\fetch-transcribe.ps1          # once, before the first cargo build
#   .\fetch-transcribe.ps1 -Force   # rebuild the prefix

param([switch]$Force)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$Version = "0.4.1"
$Sha256  = "272EB97795048CCB557C85BB9DD44C9E530E1C928443114C39816A8A5F76269D"
$Name    = "transcribe-native-$Version-windows-x86_64-cpu-vulkan"
$Url     = "https://github.com/handy-computer/transcribe.cpp/releases/download/v$Version/$Name.tar.gz"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Prefix      = Join-Path $ProjectRoot "vendor\transcribe-prebuilt"
$env:TRANSCRIBE_DIR = $Prefix

if ((Test-Path "$Prefix\lib\transcribe-link.json") -and -not $Force) {
    Write-Host "transcribe.cpp $Version already at $Prefix (use -Force to redo)" -ForegroundColor DarkGray
    exit 0
}

$Tmp = Join-Path ([IO.Path]::GetTempPath()) "voclaude-transcribe-$Version"
New-Item -ItemType Directory -Force $Tmp | Out-Null
$Tar = Join-Path $Tmp "$Name.tar.gz"
if (-not (Test-Path $Tar)) {
    Write-Host "Downloading $Url" -ForegroundColor Cyan
    Invoke-WebRequest $Url -OutFile $Tar
}
if ((Get-FileHash $Tar -Algorithm SHA256).Hash -ne $Sha256) {
    Remove-Item $Tar
    throw "Checksum mismatch for $Name.tar.gz"
}
# -m: the release archive stores 1970 timestamps, which zip files cannot hold.
& "$env:SystemRoot\System32\tar.exe" -xmzf $Tar -C $Tmp
if ($LASTEXITCODE -ne 0) { throw "tar failed" }
$Src = Join-Path $Tmp "transcribe-native-windows-x86_64-cpu-vulkan"   # the archive's top directory has no version

if (Test-Path $Prefix) { Remove-Item -Recurse -Force $Prefix }
New-Item -ItemType Directory -Force "$Prefix\bin", "$Prefix\lib", "$Prefix\include" | Out-Null
Copy-Item "$Src\*.dll" "$Prefix\bin"
Copy-Item "$Src\contract.json" "$Prefix\bin"
Copy-Item -Recurse "$Src\licenses" "$Prefix\licenses"

# MSVC dumpbin + lib from the newest installed toolset.
$VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$Dumpbin = & $VsWhere -latest -products * -find "VC\Tools\MSVC\*\bin\Hostx64\x64\dumpbin.exe" | Select-Object -Last 1
if (-not $Dumpbin) { throw "MSVC build tools not found (need dumpbin.exe and lib.exe)" }
$LibExe = Join-Path (Split-Path $Dumpbin) "lib.exe"

$Exports = & $Dumpbin /nologo /exports "$Prefix\bin\transcribe.dll" |
    ForEach-Object { if ($_ -match '^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]{8}\s+(\w+)') { $Matches[1] } }
if ($Exports.Count -lt 50) { throw "Only $($Exports.Count) exports found in transcribe.dll" }
$Def = Join-Path $Tmp "transcribe.def"
@("LIBRARY transcribe.dll", "EXPORTS") + ($Exports | ForEach-Object { "    $_" }) | Set-Content -Encoding ascii $Def
& $LibExe /nologo /def:$Def /machine:x64 /out:"$Prefix\lib\transcribe.lib" | Out-Null
if ($LASTEXITCODE -ne 0) { throw "lib.exe failed" }
Remove-Item "$Prefix\lib\transcribe.exp" -ErrorAction SilentlyContinue

# The manifest transcribe-cpp-sys reads (same keys as cmake/transcribe-install.cmake).
@"
{
  "version": "$Version",
  "commit": "release-v$Version",
  "shared": true,
  "backend_dl": true,
  "module_dir": "bin",
  "backends": ["cpu", "vulkan"],
  "metal_embed": false,
  "include_dir": "include",
  "lib_dir": "lib",
  "libraries": ["transcribe"],
  "library_paths": [],
  "system_libs": [],
  "frameworks": [],
  "link_flags": []
}
"@ | Set-Content -Encoding ascii "$Prefix\lib\transcribe-link.json"

Write-Host "transcribe.cpp $Version prefix ready: $Prefix ($($Exports.Count) exports)" -ForegroundColor Green

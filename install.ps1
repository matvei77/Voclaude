# install.ps1 - Install or update Voclaude with one command:
#
#   irm https://raw.githubusercontent.com/matvei77/Voclaude/master/install.ps1 | iex
#
# Downloads the newest installer from GitHub Releases and runs it silently:
# per-user, no administrator rights, into %LOCALAPPDATA%\Programs\Voclaude.
# A first install makes Voclaude start when you sign in; an update keeps your
# choice. Voclaude starts when the installer is done.
# Keep this file ASCII only: PowerShell 5.1 misreads UTF-8 without a BOM.

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is slow with a progress bar
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Url = "https://github.com/matvei77/Voclaude/releases/latest/download/voclaude-setup.exe"
$Setup = Join-Path ([IO.Path]::GetTempPath()) "voclaude-setup.exe"

Write-Host "Downloading Voclaude..." -ForegroundColor Cyan
Invoke-WebRequest -Uri $Url -OutFile $Setup -UseBasicParsing
Unblock-File $Setup

Write-Host "Installing..." -ForegroundColor Cyan
# WaitForExit, not -Wait: Windows PowerShell's -Wait also waits for the
# processes setup starts, and setup starts Voclaude.
$p = Start-Process -FilePath $Setup -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART" -PassThru
$p.WaitForExit()
Remove-Item $Setup -Force -ErrorAction SilentlyContinue
if ($p.ExitCode -ne 0) { throw "The installer failed (exit code $($p.ExitCode))." }

$Exe = Join-Path $env:LOCALAPPDATA "Programs\Voclaude\voclaude.exe"
Write-Host "Voclaude is installed: $Exe" -ForegroundColor Green
Write-Host "It runs in the system tray. Press F4 to dictate; left-click the tray icon for the window."

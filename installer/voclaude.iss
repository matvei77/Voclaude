; Inno Setup script for Voclaude.
;
; Built by package.ps1 -Installer, which passes the staging directory that
; already holds voclaude.exe, the transcribe.cpp and MSVC runtime DLLs (plus
; the CUDA DLLs for the -Cuda variant), config.example.toml and README-FIRST.txt:
;
;   ISCC.exe /DAppVersion=0.6.0 /DStageDir=..\dist\voclaude-standard ^
;            /DOutputDir=..\dist /DVariant=standard installer\voclaude.iss
;
; Variant "standard" produces voclaude-v0.6.0-setup.exe; "cuda" produces
; voclaude-v0.6.0-cuda-setup.exe. Both install to the same place (same AppId).
;
; Design: one click. A per-user install (no admin prompt) into
; %LOCALAPPDATA%\Programs with no wizard pages: double-click, a progress bar,
; and the app starts. The first install makes Voclaude start at sign-in; an
; upgrade keeps whatever the user chose (Settings > Start with Windows).
; Silent install: voclaude-setup.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef StageDir
  #define StageDir "..\dist\voclaude-standard"
#endif
#ifndef OutputDir
  #define OutputDir "..\dist"
#endif
#ifndef Variant
  #define Variant "standard"
#endif
#if Variant == "cuda"
  #define Suffix "-cuda"
#else
  #define Suffix ""
#endif

[Setup]
AppId={{6F3C1A2E-9B7D-4E5A-8C21-0A4D5E6F7A8B}
AppName=Voclaude
AppVersion={#AppVersion}
AppVerName=Voclaude {#AppVersion}
AppPublisher=Voclaude
AppPublisherURL=https://github.com/matvei77/Voclaude
AppSupportURL=https://github.com/matvei77/Voclaude/issues
DefaultDirName={localappdata}\Programs\Voclaude
DefaultGroupName=Voclaude
PrivilegesRequired=lowest
DisableWelcomePage=yes
DisableDirPage=yes
DisableProgramGroupPage=yes
DisableReadyPage=yes
DisableFinishedPage=yes
OutputDir={#OutputDir}
OutputBaseFilename=voclaude-v{#AppVersion}{#Suffix}-setup
Compression=lzma2/max
SolidCompression=yes
LZMAUseSeparateProcess=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes
RestartApplications=no
UninstallDisplayIcon={app}\voclaude.exe
UninstallDisplayName=Voclaude
SetupIconFile=..\assets\icon.ico
WizardStyle=modern
ShowLanguageDialog=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "{#StageDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs

#if Variant != "cuda"
[InstallDelete]
; CUDA runtime DLLs left by a CUDA build or a release before 0.6.0 (~650 MB).
; The standard build never loads them.
Type: files; Name: "{app}\cublas64_*.dll"
Type: files; Name: "{app}\cublasLt64_*.dll"
Type: files; Name: "{app}\curand64_*.dll"
Type: files; Name: "{app}\nvrtc64_*.dll"
Type: files; Name: "{app}\nvrtc-builtins64_*.dll"
Type: files; Name: "{app}\cudart64_*.dll"
#endif

[Icons]
Name: "{autoprograms}\Voclaude"; Filename: "{app}\voclaude.exe"; Comment: "Voice input anywhere (press F4)"

[Registry]
; First install only: an upgrade keeps the user's choice.
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "Voclaude"; ValueData: """{app}\voclaude.exe"""; Flags: uninsdeletevalue; Check: not IsUpgrade

[Run]
; Start the app when setup ends (also after a silent install).
Filename: "{app}\voclaude.exe"; Flags: nowait

[UninstallRun]
Filename: "{sys}\taskkill.exe"; Parameters: "/IM voclaude.exe /F"; Flags: runhidden; RunOnceId: "StopVoclaude"

[Code]
// Voclaude was installed before (by this or an older installer).
function IsUpgrade: Boolean;
var
  S: String;
begin
  Result := RegQueryStringValue(HKCU, 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{6F3C1A2E-9B7D-4E5A-8C21-0A4D5E6F7A8B}_is1', 'UninstallString', S);
end;

// Stop a running Voclaude before files are replaced (the tray app keeps
// voclaude.exe locked; Restart Manager handles it too, but this is quicker
// and avoids the "close applications" page).
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  ResultCode: Integer;
begin
  Exec(ExpandConstant('{sys}\taskkill.exe'), '/IM voclaude.exe /F', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  Result := '';
end;

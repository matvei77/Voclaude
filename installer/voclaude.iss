; Inno Setup script for Voclaude.
;
; Built by package.ps1 -Installer, which passes the staging directory that
; already holds voclaude.exe, the CUDA/MSVC runtime DLLs, config.example.toml
; and README-FIRST.txt:
;
;   ISCC.exe /DAppVersion=0.4.0 /DStageDir=..\dist\voclaude-gpu ^
;            /DOutputDir=..\dist /DVariant=gpu installer\voclaude.iss
;
; Design: a per-user install (no admin prompt) into %LOCALAPPDATA%\Programs,
; one wizard page ("start with Windows" checkbox), then the app launches.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef StageDir
  #define StageDir "..\dist\voclaude-gpu"
#endif
#ifndef OutputDir
  #define OutputDir "..\dist"
#endif
#ifndef Variant
  #define Variant "gpu"
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
OutputDir={#OutputDir}
OutputBaseFilename=voclaude-v{#AppVersion}-{#Variant}-setup
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

[Tasks]
Name: "autostart"; Description: "Start Voclaude when I sign in to Windows"; GroupDescription: "Startup:"
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"; Flags: unchecked

[Files]
Source: "{#StageDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{autoprograms}\Voclaude"; Filename: "{app}\voclaude.exe"; Comment: "Voice input anywhere (press F4)"
Name: "{autoprograms}\Voclaude README"; Filename: "{app}\README-FIRST.txt"
Name: "{autodesktop}\Voclaude"; Filename: "{app}\voclaude.exe"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "Voclaude"; ValueData: """{app}\voclaude.exe"""; Flags: uninsdeletevalue; Tasks: autostart

[Run]
Filename: "{app}\voclaude.exe"; Description: "Launch Voclaude now (press F4 to dictate)"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "{sys}\taskkill.exe"; Parameters: "/IM voclaude.exe /F"; Flags: runhidden; RunOnceId: "StopVoclaude"

[Code]
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

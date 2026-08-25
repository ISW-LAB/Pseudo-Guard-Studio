;  PG-Label — Inno Setup installer script
; ---------------------------------------------------------------------------------------------
; Compiled by packaging\build.py --installer, which passes the version and the training flag:
;     ISCC.exe /DMyAppVersion=1.0.0 /DWithTraining=1 packaging\installer.iss
; Compile by hand only after a build — it packages packaging\dist\PG-Label.
;
; Installs PER USER by default: no administrator rights, no UAC prompt, nothing written outside
; the user's own profile. That matters on managed lab machines where annotators are not admins.

#define MyAppName "PG-Label"
#define MyAppPublisher "Pseudo-Guard Studio"
#define MyAppExeName "PG-Label.exe"

#ifndef MyAppVersion
  #define MyAppVersion "1.0.0"
#endif
#ifndef WithTraining
  #define WithTraining "1"
#endif

[Setup]
; Keep this GUID stable forever — it is how Windows recognises an upgrade of the same product
; rather than a second, parallel installation.
AppId={{8AF9FEB8-6B13-4D6B-82DB-35FCC1595DAA}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=dist
OutputBaseFilename=PG-Label-Setup-{#MyAppVersion}
SetupIconFile=assets\pglabel.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; The app is a local web server on 127.0.0.1 — no inbound firewall rule, so no security prompt.
AppComments=Collaborative auto-labeling for object detection (local web app)

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; The whole PyInstaller onedir output. Ship all of it: PG-Label.exe alone does not run.
Source: "dist\{#MyAppName}\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist\{#MyAppName}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\docs\WINDOWS.md"; DestDir: "{app}"; Flags: ignoreversion isreadme
#if WithTraining == "1"
Source: "install_training_pack.bat"; DestDir: "{app}"; Flags: ignoreversion
#endif

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{#MyAppName} — where are my files?"; Filename: "{app}\{#MyAppExeName}"; Parameters: "--where"; Comment: "Print the folders this install uses"
#if WithTraining == "1"
; Optional second step, deliberately NOT run during setup: it downloads up to 2.5 GB of CUDA
; wheels, which must be the user's decision, on their network, at a time they choose.
Name: "{group}\Install training pack"; Filename: "{app}\{#MyAppExeName}"; Parameters: "--install-gpu-pack"; Comment: "Add torch + ultralytics so the Train button works (large download)"
#endif
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; PyInstaller's bundle folder can hold .pyc files written after install; leave nothing behind.
Type: filesandordirs; Name: "{app}\_internal"

; NOTE: the user's labels, settings and training pack live in %LOCALAPPDATA%\PG-Label and are
; intentionally NOT removed by the uninstaller — losing annotation work to an uninstall would be
; unforgivable. docs\WINDOWS.md documents how to delete that folder deliberately.

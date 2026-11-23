; HKL-PDF installer (Inno Setup 6). Built by installer\build.ps1, which passes
; MyAppVersion, SourceDir (the PyInstaller folder) and GpuDownloadMB.
;
; Installs per user without administrator rights (by default into
; %LOCALAPPDATA%\Programs\HKL-PDF); "Install for all users" is offered too.
;
; NVIDIA GPU support is not inside the installer: the optional task runs
; "HKL-PDF.exe --install-gpu", which downloads the pinned CuPy / CUDA wheels
; from PyPI into %LOCALAPPDATA%\HKL-PDF\gpu. It is also available later from
; the program's File menu.

#define MyAppName "HKL-PDF"
#ifndef MyAppVersion
  #define MyAppVersion "0.1.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\dist\HKL-PDF"
#endif
#ifndef GpuDownloadMB
  #define GpuDownloadMB "700"
#endif
#define MyAppPublisher "Marcelo Malagutti"
#define MyAppURL "https://github.com/MalaguttiMarcelo/HKL-PDF"
#define MyAppExeName "HKL-PDF.exe"

[Setup]
AppId={{F9AB3198-6D31-42CF-8BB2-18AAFC21B93E}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
AllowNoIcons=yes
LicenseFile=..\LICENSE
OutputDir=..\installer-output
OutputBaseFilename=HKL-PDF-Setup-{#MyAppVersion}
SetupIconFile=HKL-PDF.ico
Compression=lzma2/max
SolidCompression=yes
LZMANumBlockThreads=4
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Per user, no administrator rights; the user may still choose "all users".
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
UninstallDisplayIcon={app}\{#MyAppExeName}
; A running HKL-PDF (also the one kept loaded in the tray) is closed first.
CloseApplications=force

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked
Name: "gpu"; Description: "Download and install NVIDIA GPU support (about {#GpuDownloadMB} MB, needs internet) - only for computers with an NVIDIA graphics card"; GroupDescription: "Optional:"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\HKL-PDF"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\HKL-PDF (classic window)"; Filename: "{app}\{#MyAppExeName}"; Parameters: "--classic"
Name: "{group}\HKL-PDF examples"; Filename: "{app}\_internal\examples"
Name: "{autodesktop}\HKL-PDF"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Registry]
; "Start at Windows login" is set from inside HKL-PDF (off by default);
; uninstalling removes it.
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: none; ValueName: "HKL-PDF"; Flags: uninsdeletevalue dontcreatekey

[UninstallDelete]
; Downloaded GPU support (per user).
Type: filesandordirs; Name: "{localappdata}\HKL-PDF\gpu"

[Run]
Filename: "{app}\{#MyAppExeName}"; Parameters: "--warmup"; StatusMsg: "Preparing the numerical kernels (about a minute)..."; Flags: runhidden waituntilterminated
Filename: "{app}\{#MyAppExeName}"; Parameters: "--install-gpu"; StatusMsg: "Installing NVIDIA GPU support..."; Tasks: gpu; Flags: waituntilterminated
Filename: "{app}\{#MyAppExeName}"; Description: "Start HKL-PDF"; Flags: nowait postinstall skipifsilent

#define MyAppName "HKL-PDF"
#define MyAppVersion "0.1.0"
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
AllowNoIcons=yes
LicenseFile=..\LICENSE
OutputDir=..\installer-output
OutputBaseFilename=HKL-PDF-Setup-{#MyAppVersion}
SetupIconFile=HKL-PDF.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
Source: "..\dist\HKL-PDF\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\HKL-PDF"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\HKL-PDF"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch HKL-PDF"; Flags: nowait postinstall skipifsilent
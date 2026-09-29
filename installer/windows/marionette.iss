; Inno Setup script for Marionette - produces MarionetteSetup.exe.
;
; A no-CLI install: the wizard copies the app, creates Start Menu and desktop
; shortcuts that launch it with no console window, and (when the SteamVR driver
; was built) offers to register it. Uninstall unregisters the driver.
;
; Build it with Inno Setup 6 (https://jrsoftware.org/isinfo.php):
;
;   iscc installer\windows\marionette.iss ^
;        /DAppVersion=0.1.0 ^
;        /DSourceExe=build\app\Release\marionette.exe ^
;        /DDriverDir=build\dist\marionette
;
; All /D defines are optional; the defaults assume a Release build tree at the
; repository root. Relative paths are resolved from this .iss file's folder.

#ifndef AppVersion
#define AppVersion "0.1.0"
#endif
#ifndef SourceExe
#define SourceExe "..\..\build\app\Release\marionette.exe"
#endif
#ifndef DriverDir
#define DriverDir "..\..\build\dist\marionette"
#endif
#ifndef RepoRoot
#define RepoRoot "..\.."
#endif

#define AppName "Marionette"
#define AppPublisher "Marionette"
#define AppURL "https://github.com/andrewjfiore/kinect-fbt"
#define LaunchVbs "launch-marionette.vbs"

; Was the SteamVR driver folder built? Resolved at compile time so the driver
; task/files/registration only appear when there is a driver to register.
#define HaveDriver DirExists(SourcePath + DriverDir)

[Setup]
AppId={{7C6B4B2E-2F5E-4E6A-9B3D-9E1E6C2A4F10}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\marionette.ico
OutputBaseFilename=MarionetteSetup
OutputDir={#RepoRoot}\dist
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
ArchitecturesAllowed=x64compatible
LicenseFile={#RepoRoot}\LICENSE
SetupIconFile=marionette.ico

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"
#if HaveDriver
Name: "registerdriver"; Description: "Register the &SteamVR driver (needed for PCVR / ALVR / Virtual Desktop)"; GroupDescription: "SteamVR:"
#endif

[Files]
Source: "{#SourceExe}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#LaunchVbs}"; DestDir: "{app}"; Flags: ignoreversion
Source: "marionette.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#RepoRoot}\config\demo.json"; DestDir: "{app}\config"; Flags: ignoreversion
Source: "{#RepoRoot}\scripts\install-driver.ps1"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "{#RepoRoot}\scripts\fix-kinect-services.ps1"; DestDir: "{app}\scripts"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#RepoRoot}\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#RepoRoot}\docs\USER_GUIDE.md"; DestDir: "{app}\docs"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#RepoRoot}\docs\INSTALL.md"; DestDir: "{app}\docs"; Flags: ignoreversion skipifsourcedoesntexist
#if HaveDriver
Source: "{#DriverDir}\*"; DestDir: "{app}\driver\marionette"; Flags: ignoreversion recursesubdirs createallsubdirs
#endif

[Icons]
; Shortcuts run the VBS via wscript so no console window ever appears.
Name: "{group}\{#AppName}"; Filename: "{sys}\wscript.exe"; Parameters: """{app}\{#LaunchVbs}"""; WorkingDir: "{app}"; IconFilename: "{app}\marionette.ico"; Comment: "Start Marionette full-body tracking"
Name: "{group}\Marionette User Guide"; Filename: "{app}\docs\USER_GUIDE.md"; Check: FileExists(ExpandConstant('{app}\docs\USER_GUIDE.md'))
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{sys}\wscript.exe"; Parameters: """{app}\{#LaunchVbs}"""; WorkingDir: "{app}"; IconFilename: "{app}\marionette.ico"; Tasks: desktopicon

[Run]
#if HaveDriver
; Register the SteamVR driver (best effort; a missing SteamVR is reported by
; the script but does not fail the install).
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\scripts\install-driver.ps1"" -DriverPath ""{app}\driver\marionette"""; StatusMsg: "Registering the SteamVR driver..."; Flags: runhidden waituntilterminated; Tasks: registerdriver
#endif
; Offer to launch right after install.
Filename: "{sys}\wscript.exe"; Parameters: """{app}\{#LaunchVbs}"""; Description: "Launch {#AppName} now"; Flags: postinstall nowait skipifsilent

[UninstallRun]
#if HaveDriver
; Unregister the driver on uninstall (ignore errors if SteamVR is gone).
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\scripts\install-driver.ps1"" -Remove -DriverPath ""{app}\driver\marionette"""; Flags: runhidden waituntilterminated; RunOnceId: "UnregisterMarionetteDriver"; Check: FileExists(ExpandConstant('{app}\scripts\install-driver.ps1'))
#endif

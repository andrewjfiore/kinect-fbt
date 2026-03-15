; Inno Setup Script for FBT Server
; Requires Inno Setup 6+: https://jrsoftware.org/isinfo.php

#define MyAppName "FBT Server"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "kinect-fbt"
#define MyAppURL "https://github.com/your-org/kinect-fbt"
#define MyAppExeName "FBT-Server.exe"

[Setup]
AppId={{B8E3F4A2-7C1D-4F5E-9A3B-6D2E8F1C4A5B}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
OutputDir=..\..\dist
OutputBaseFilename=FBT-Server-Setup-{#MyAppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\..\dist\FBT-Server\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(#MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
function IsKinectSDKInstalled: Boolean;
begin
  Result := RegKeyExists(HKLM, 'SOFTWARE\Microsoft\Kinect\v2.0');
  if not Result then
    Result := RegKeyExists(HKLM, 'SOFTWARE\WOW6432Node\Microsoft\Kinect\v2.0');
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    if not IsKinectSDKInstalled then
    begin
      MsgBox('Kinect for Windows SDK 2.0 was not detected.' + #13#10 +
             'FBT Server requires the Kinect SDK runtime to function.' + #13#10#13#10 +
             'Download from: https://www.microsoft.com/en-us/download/details.aspx?id=44561',
             mbInformation, MB_OK);
    end;
  end;
end;

' launch-marionette.vbs - start Marionette with NO console window.
'
' The Start Menu and desktop shortcuts created by the installer point here
' (run via wscript.exe). It sets up a per-user working directory the first
' time it runs, then launches the pipeline; the app itself opens the dashboard
' in the default browser. Nothing about day-to-day use touches a terminal.
Option Explicit

Dim shell, fso, appDir, dataDir, exe, bundledCfg, cfg, cmd
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' This script lives next to marionette.exe in the install directory.
appDir = fso.GetParentFolderName(WScript.ScriptFullName)
exe = appDir & "\marionette.exe"

' Per-user, writable working directory: %LOCALAPPDATA%\Marionette. Calibration
' and config live here so nothing needs write access to Program Files.
dataDir = shell.ExpandEnvironmentStrings("%LOCALAPPDATA%") & "\Marionette"
If Not fso.FolderExists(dataDir) Then fso.CreateFolder(dataDir)

' Seed an editable config from the bundled demo on first run.
cfg = dataDir & "\config.json"
bundledCfg = appDir & "\config\demo.json"
If (Not fso.FileExists(cfg)) And fso.FileExists(bundledCfg) Then
    fso.CopyFile bundledCfg, cfg
End If
If Not fso.FileExists(cfg) Then cfg = bundledCfg  ' fall back if copy failed

If Not fso.FileExists(exe) Then
    MsgBox "Marionette is not installed correctly (marionette.exe not found in " & _
        appDir & ").", vbCritical, "Marionette"
    WScript.Quit 1
End If

' Run from the data directory so calibration.json is written somewhere writable.
shell.CurrentDirectory = dataDir
cmd = """" & exe & """ run -c """ & cfg & """"

' 0 = hidden window, False = do not wait. The app opens the browser itself.
shell.Run cmd, 0, False

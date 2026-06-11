# install-driver.ps1 - register (or remove) the Marionette SteamVR driver via vrpathreg.
#
# Usage:
#   .\install-driver.ps1                          # adddriver on <repo>\build\dist\marionette
#   .\install-driver.ps1 -DriverPath C:\some\dir  # adddriver on an explicit dist folder
#   .\install-driver.ps1 -Remove                  # removedriver
#
# Works on Windows PowerShell 5.1 and PowerShell 7+. Plain ASCII only.

[CmdletBinding()]
param(
    [string]$DriverPath = "",
    [switch]$Remove
)

$ErrorActionPreference = "Stop"

function Get-SteamRoots {
    $roots = @()

    # Registry (covers non-default Steam installs).
    foreach ($regPath in @("HKCU:\Software\Valve\Steam", "HKLM:\SOFTWARE\WOW6432Node\Valve\Steam")) {
        try {
            $props = Get-ItemProperty -Path $regPath -ErrorAction Stop
            foreach ($name in @("SteamPath", "InstallPath")) {
                $value = $props.$name
                if ($value -and (Test-Path $value)) {
                    $roots += (Resolve-Path $value).Path
                }
            }
        } catch {
            # Registry key missing; ignore.
        }
    }

    # Default install location.
    $pf86 = ${env:ProgramFiles(x86)}
    if ($pf86) {
        $default = Join-Path $pf86 "Steam"
        if (Test-Path $default) {
            $roots += $default
        }
    }

    return $roots | Select-Object -Unique
}

function Get-SteamLibraries {
    param([string[]]$SteamRoots)

    $libraries = @()
    foreach ($root in $SteamRoots) {
        $libraries += $root
        $vdf = Join-Path $root "steamapps\libraryfolders.vdf"
        if (-not (Test-Path $vdf)) { continue }

        # Parse "path" entries from libraryfolders.vdf (new format). Old format
        # lines look like:  "1"  "D:\\SteamLibrary"  - catch those too.
        foreach ($line in (Get-Content $vdf)) {
            if ($line -match '^\s*"path"\s+"(.+)"\s*$' -or
                $line -match '^\s*"\d+"\s+"(.+)"\s*$') {
                $candidate = $Matches[1] -replace '\\\\', '\'
                if (Test-Path $candidate) {
                    $libraries += (Resolve-Path $candidate).Path
                }
            }
        }
    }
    return $libraries | Select-Object -Unique
}

function Find-VrPathReg {
    $steamRoots = Get-SteamRoots
    if ($steamRoots.Count -eq 0) {
        throw "Steam installation not found (registry and default path both empty)."
    }
    foreach ($library in (Get-SteamLibraries -SteamRoots $steamRoots)) {
        $candidate = Join-Path $library "steamapps\common\SteamVR\bin\win64\vrpathreg.exe"
        if (Test-Path $candidate) {
            return $candidate
        }
    }
    throw "vrpathreg.exe not found. Is SteamVR installed? Looked under: $($steamRoots -join '; ')"
}

# Resolve the driver dist folder (default: <repo>/build/dist/marionette
# relative to this script, which lives in <repo>/scripts).
if ([string]::IsNullOrWhiteSpace($DriverPath)) {
    $DriverPath = Join-Path $PSScriptRoot "..\build\dist\marionette"
}
if (-not (Test-Path $DriverPath)) {
    throw "Driver folder not found: $DriverPath (build the project first, or pass -DriverPath)."
}
$DriverPath = (Resolve-Path $DriverPath).Path

$manifest = Join-Path $DriverPath "driver.vrdrivermanifest"
if (-not (Test-Path $manifest)) {
    throw "No driver.vrdrivermanifest in $DriverPath - this is not a SteamVR driver folder."
}

$vrpathreg = Find-VrPathReg
Write-Host "vrpathreg: $vrpathreg"
Write-Host "driver:    $DriverPath"

$verb = "adddriver"
if ($Remove) { $verb = "removedriver" }

& $vrpathreg $verb $DriverPath
if ($LASTEXITCODE -ne 0) {
    throw "vrpathreg $verb failed with exit code $LASTEXITCODE."
}

if ($Remove) {
    Write-Host "Removed Marionette driver registration."
} else {
    Write-Host "Registered Marionette driver. Restart SteamVR to load it."
}

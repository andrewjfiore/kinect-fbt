# Marionette: bring half-initialized Kinects back to life.
# - Starts the Kinect v2 runtime services (KinectMonitor / KinectManagement)
# - Power-cycles the Kinect v1 PnP device nodes (clears E_NUI_NOTREADY)
# Self-elevates; results are written to %TEMP%\kinect-fix.log so the calling
# session can read them.

$logPath = Join-Path $env:TEMP 'kinect-fix.log'

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Start-Process -FilePath 'powershell.exe' -Verb RunAs -Wait -ArgumentList @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$PSCommandPath`""
    )
    if (Test-Path $logPath) { Get-Content $logPath }
    exit 0
}

$out = New-Object System.Collections.Generic.List[string]

foreach ($svcName in @('KinectManagement', 'KinectMonitor')) {
    $svc = Get-Service -Name $svcName -ErrorAction SilentlyContinue
    if ($null -eq $svc) { $out.Add("service $svcName : not installed"); continue }
    if ($svc.Status -ne 'Running') {
        try {
            Start-Service -Name $svcName -ErrorAction Stop
            $out.Add("service $svcName : started")
        } catch {
            $out.Add("service $svcName : START FAILED - $($_.Exception.Message)")
        }
    } else {
        $out.Add("service $svcName : already running")
    }
}

# Power-cycle the Kinect v1 nodes (class 'Kinect for Windows').
$v1 = Get-PnpDevice | Where-Object { $_.Class -eq 'Kinect for Windows' }
if ($v1) {
    foreach ($dev in $v1) {
        try {
            Disable-PnpDevice -InstanceId $dev.InstanceId -Confirm:$false -ErrorAction Stop
            $out.Add("v1 disable $($dev.FriendlyName) : ok")
        } catch {
            $out.Add("v1 disable $($dev.FriendlyName) : $($_.Exception.Message)")
        }
    }
    Start-Sleep -Seconds 3
    foreach ($dev in $v1) {
        try {
            Enable-PnpDevice -InstanceId $dev.InstanceId -Confirm:$false -ErrorAction Stop
            $out.Add("v1 enable $($dev.FriendlyName) : ok")
        } catch {
            $out.Add("v1 enable $($dev.FriendlyName) : $($_.Exception.Message)")
        }
    }
} else {
    $out.Add('v1 : no Kinect for Windows class devices found')
}

Start-Sleep -Seconds 2
foreach ($svcName in @('KinectManagement', 'KinectMonitor')) {
    $svc = Get-Service -Name $svcName -ErrorAction SilentlyContinue
    if ($svc) { $out.Add("final: $svcName = $($svc.Status)") }
}

$out | Set-Content -Path $logPath -Encoding utf8

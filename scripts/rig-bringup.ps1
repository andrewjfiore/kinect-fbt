# Marionette rig bring-up orchestrator: waits for both sensors to see a body,
# then runs pair calibration (front=v2 reference, back=v1 target) and body
# model capture through the dashboard API, reporting progress and results.
# ASCII only. Assumes 'marionette run -c config/rig.json' is already running.

$api = 'http://127.0.0.1:8211/api'
$deadline = (Get-Date).AddSeconds(300)

function Get-Json($url) {
    try { return Invoke-RestMethod -Uri $url -TimeoutSec 4 } catch { return $null }
}

Write-Output 'waiting for the dashboard API...'
while ((Get-Date) -lt $deadline) {
    $s = Get-Json "$api/status"
    if ($s) { break }
    Start-Sleep -Seconds 2
}
if (-not $s) { Write-Output 'FAIL: dashboard API never came up'; exit 1 }

Write-Output 'waiting for BOTH sensors to see a body (stand in the overlap)...'
$bothSince = $null
$lastLine = ''
while ((Get-Date) -lt $deadline) {
    $s = Get-Json "$api/status"
    if ($s) {
        $line = ($s.nodes | ForEach-Object { "$($_.id):" + $(if ($_.has_body) { 'BODY' } else { 'no body' }) + " fps=$([math]::Round($_.fps,1))" }) -join '  |  '
        if ($line -ne $lastLine) { Write-Output "  $line"; $lastLine = $line }
        $both = (@($s.nodes | Where-Object { $_.has_body })).Count -eq 2
        if ($both) {
            if (-not $bothSince) { $bothSince = Get-Date }
            elseif (((Get-Date) - $bothSince).TotalSeconds -ge 2) { break }
        } else { $bothSince = $null }
    }
    Start-Sleep -Seconds 1
}
if (-not $bothSince) { Write-Output 'TIMEOUT: both sensors never saw a body together. Rig left running; use the dashboard wizard when ready.'; exit 2 }

Write-Output ''
Write-Output 'both sensors tracking. starting PAIR calibration: HOLD a pose, switch pose every ~2 s (T-pose, arms down, arms up, lean left/right)...'
$body = @{ reference = 'front'; target = 'back'; min_samples = 400; max_seconds = 120 } | ConvertTo-Json
try { Invoke-RestMethod -Uri "$api/calibrate/pair" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 5 | Out-Null }
catch { Write-Output "FAIL: could not start pair job: $($_.Exception.Message)"; exit 1 }

$lastDone = -1
while ($true) {
    Start-Sleep -Seconds 1
    $c = Get-Json "$api/calibrate/status"
    if (-not $c) { continue }
    if ($c.active -and $c.progress.done -ne $lastDone) {
        Write-Output "  pair samples: $($c.progress.done)/$($c.progress.needed)"
        $lastDone = $c.progress.done
    }
    if (-not $c.active -and $c.done) { break }
}
Write-Output "PAIR RESULT: ok=$($c.ok) rmse_cm=$([math]::Round($c.rmse_cm,2)) - $($c.message)"
if (-not $c.ok) { exit 1 }

Write-Output ''
Write-Output 'starting BODY MODEL capture (15 s): stand with arms slightly out, move gently...'
try { Invoke-RestMethod -Uri "$api/calibrate/body" -Method Post -Body (@{ seconds = 15 } | ConvertTo-Json) -ContentType 'application/json' -TimeoutSec 5 | Out-Null }
catch { Write-Output "FAIL: could not start body job: $($_.Exception.Message)"; exit 1 }

while ($true) {
    Start-Sleep -Seconds 2
    $c = Get-Json "$api/calibrate/status"
    if ($c -and -not $c.active -and $c.done) { break }
}
Write-Output "BODY RESULT: ok=$($c.ok) - $($c.message)"

Write-Output ''
Write-Output 'solved calibration (calibration-rig.json):'
Get-Content 'C:\Users\andre\marionette\calibration-rig.json' -ErrorAction SilentlyContinue
exit 0

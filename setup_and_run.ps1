# FBT Server - Setup & Run (Python 3.10)
# Right-click > Open PowerShell in the kinect-fbt folder, then run: .\setup_and_run.ps1

Write-Host ""
Write-Host "=== FBT Server Setup ===" -ForegroundColor Cyan
Write-Host ""

$serverDir = Join-Path $PSScriptRoot "kinect_server"
$venvDir = Join-Path $serverDir "venv"
$reqFile = Join-Path $serverDir "requirements_windows.txt"

# Check Python 3.10 is available
$py310 = $null
try {
    $py310 = & py -3.10 --version 2>&1
} catch {}

if (-not $py310 -or $py310 -notlike "Python 3.10*") {
    Write-Host "ERROR: Python 3.10 not found." -ForegroundColor Red
    Write-Host "Install it from https://www.python.org/downloads/release/python-31011/"
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "Found $py310" -ForegroundColor Green

# Remove old venv if it exists with wrong Python version
if (Test-Path $venvDir) {
    $currentPy = $null
    try {
        $currentPy = & "$venvDir\Scripts\python.exe" --version 2>&1
    } catch {}

    if ($currentPy -notlike "Python 3.10*") {
        Write-Host "Existing venv is $currentPy - removing it..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $venvDir
    }
}

# Create venv if needed
if (-not (Test-Path $venvDir)) {
    Write-Host "Creating Python 3.10 virtual environment..." -ForegroundColor Yellow
    & py -3.10 -m venv $venvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create venv" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "venv created" -ForegroundColor Green
} else {
    Write-Host "venv already exists (Python 3.10)" -ForegroundColor Green
}

# Activate venv
& "$venvDir\Scripts\Activate.ps1"

# Install/update dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
& pip install -r $reqFile --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: pip install failed" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "Dependencies installed" -ForegroundColor Green

# Verify pykinect2
try {
    & python -c "from pykinect2 import PyKinectV2; print('pykinect2 OK')"
    if ($LASTEXITCODE -ne 0) { throw "import failed" }
    Write-Host "pykinect2 verified" -ForegroundColor Green
} catch {
    Write-Host "WARNING: pykinect2 import failed. Make sure Kinect SDK 2.0 is installed:" -ForegroundColor Yellow
    Write-Host "  https://www.microsoft.com/en-us/download/details.aspx?id=44561" -ForegroundColor Yellow
}

# Pull latest code
Write-Host ""
Write-Host "Pulling latest code..." -ForegroundColor Yellow
git pull origin master 2>&1 | Out-Host

# Launch the GUI
Write-Host ""
Write-Host "=== Launching FBT Server ===" -ForegroundColor Cyan
Write-Host ""
Set-Location $serverDir
& python gui.py

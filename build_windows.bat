@echo off
REM Build FBTServer.exe for Windows
REM Requirements: Python 3.10+, pip, Kinect for Windows SDK 2.0

setlocal
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%\kinect_server"

echo ===========================================
echo  FBT Server - Windows Build
echo ===========================================

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ from python.org
    pause
    exit /b 1
)

REM Create/activate venv
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements_windows.txt --quiet

REM Check for pykinect2
python -c "import pykinect2" >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: pykinect2 not installed or Kinect SDK not found.
    echo Install Kinect for Windows SDK 2.0:
    echo   https://www.microsoft.com/en-us/download/details.aspx?id=44561
    echo Then run: pip install pykinect2
    echo.
    echo Build will continue but Kinect hardware will NOT work without the SDK.
    pause
)

REM Build executable
echo Building FBTServer.exe...
pyinstaller --clean fbt_server.spec

if errorlevel 1 (
    echo ERROR: Build failed. Check output above.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Build complete!
echo  Executable: dist\FBTServer.exe
echo ============================================

REM Copy calibration file if exists
if exist "calibration.json" copy calibration.json dist\ >nul

echo.
echo Run dist\FBTServer.exe to launch the GUI.
pause

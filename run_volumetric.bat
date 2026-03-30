@echo off
REM ============================================================
REM  Kinect FBT - Volumetric 3D Monitor Launcher
REM  Starts the server with WebSocket point cloud streaming.
REM  Open http://localhost:8090 in your browser to view.
REM ============================================================
cd /d "%~dp0"

set PYTHON310=C:\Users\andre\AppData\Local\Programs\Python\Python310\python.exe
set VENV=venv310

if not exist "%VENV%\Scripts\python.exe" goto :setup
goto :run

:setup
echo [1/3] Creating Python 3.10 venv...
"%PYTHON310%" -m venv "%VENV%"
if errorlevel 1 goto :no_python
echo [2/3] Installing dependencies...
"%VENV%\Scripts\pip.exe" install --upgrade pip -q
"%VENV%\Scripts\pip.exe" install opencv-python numpy scipy python-osc flask mediapipe aiohttp pykinect2 Pillow -q
echo [3/3] Patching pykinect2...
"%VENV%\Scripts\python.exe" patch_pykinect2.py "%VENV%"
goto :run

:no_python
echo ERROR: Python 3.10 not found at %PYTHON310%
echo Install from https://www.python.org/downloads/release/python-31011/
pause
exit /b 1

:run
echo.
echo ============================================================
echo  Kinect FBT Volumetric Server
echo  Kinect v1 (Xbox 360) + Kinect v2 (Xbox One)
echo  Web UI: http://localhost:8090
echo  Press Ctrl+C to stop
echo ============================================================
echo.

cd kinect_server
"%~dp0%VENV%\Scripts\python.exe" server.py --web-server --num-cameras 2 %*

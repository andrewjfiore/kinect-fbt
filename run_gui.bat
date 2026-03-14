@echo off
REM Quick-start the GUI without building (Windows)
cd /d "%~dp0\kinect_server"
if not exist "venv" (
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install -r requirements_windows.txt -q
) else (
    call venv\Scripts\activate.bat
)
python gui.py %*

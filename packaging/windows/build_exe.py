#!/usr/bin/env python3
"""
PyInstaller build script for FBT Server — Windows single-folder bundle.
Usage: python build_exe.py
"""
import PyInstaller.__main__
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KINECT_SERVER = os.path.join(ROOT, "kinect_server")

PyInstaller.__main__.run([
    os.path.join(KINECT_SERVER, "gui.py"),
    "--name", "FBT-Server",
    "--windowed",
    # Single-folder (faster startup than single-file)
    "--noconfirm",
    "--clean",
    # Collect all kinect_server modules
    "--add-data", f"{KINECT_SERVER}/calibration.py{os.pathsep}.",
    "--add-data", f"{KINECT_SERVER}/camera.py{os.pathsep}.",
    "--add-data", f"{KINECT_SERVER}/debug_http.py{os.pathsep}.",
    "--add-data", f"{KINECT_SERVER}/filter.py{os.pathsep}.",
    "--add-data", f"{KINECT_SERVER}/fusion.py{os.pathsep}.",
    "--add-data", f"{KINECT_SERVER}/osc_output.py{os.pathsep}.",
    "--add-data", f"{KINECT_SERVER}/platform_backend.py{os.pathsep}.",
    "--add-data", f"{KINECT_SERVER}/server.py{os.pathsep}.",
    # Hidden imports
    "--hidden-import", "mediapipe",
    "--hidden-import", "cv2",
    "--hidden-import", "numpy",
    "--hidden-import", "PIL",
    "--hidden-import", "PIL.Image",
    "--hidden-import", "PIL.ImageTk",
    "--hidden-import", "pythonosc",
    "--hidden-import", "flask",
    # MediaPipe data files
    "--collect-data", "mediapipe",
    # Output
    "--distpath", os.path.join(ROOT, "dist"),
    "--workpath", os.path.join(ROOT, "build"),
    "--specpath", os.path.join(ROOT, "build"),
    # Icon (if exists)
    *(["--icon", os.path.join(ROOT, "icon.ico")] if os.path.exists(os.path.join(ROOT, "icon.ico")) else []),
])

print("\n✅ Build complete! Output: dist/FBT-Server/")
print("Run: dist/FBT-Server/FBT-Server.exe")

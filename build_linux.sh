#!/bin/bash
# Build FBTServer (Linux) — single-file executable
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/kinect_server"

echo "==========================================="
echo " FBT Server — Linux Build"
echo "==========================================="

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found."
    exit 1
fi

# Create venv
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements_linux.txt -q

# Check libfreenect2 Python binding
python3 -c "import pylibfreenect2" 2>/dev/null || {
    echo ""
    echo "WARNING: pylibfreenect2 not available."
    echo "Build continues but real Kinect hardware won't work."
    echo "To install: see README.md — build libfreenect2 from source."
    echo ""
}

# Check tkinter
python3 -c "import tkinter" 2>/dev/null || {
    echo "ERROR: tkinter not installed."
    echo "Install: sudo apt install python3-tk"
    exit 1
}

# Build
echo "Building FBTServer executable..."
pyinstaller --clean fbt_server.spec

echo ""
echo "==========================================="
echo " Build complete!"
echo " Executable: dist/FBTServer"
echo "==========================================="

# Copy calibration file if exists
[ -f "calibration.json" ] && cp calibration.json dist/

# Make executable
chmod +x dist/FBTServer

echo ""
echo "Run: ./kinect_server/dist/FBTServer"

#!/bin/bash
# Quick-start the GUI without building an executable (Linux)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/kinect_server"

if [ ! -d "venv" ]; then
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements_linux.txt -q
else
    source venv/bin/activate
fi

python3 gui.py "$@"

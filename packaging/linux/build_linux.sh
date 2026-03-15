#!/usr/bin/env bash
# Build FBT Server for Linux — PyInstaller single-folder bundle + optional AppImage
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
KINECT_SERVER="$ROOT/kinect_server"
DIST="$ROOT/dist"
BUILD="$ROOT/build"
APP_NAME="FBT-Server"
VERSION="${1:-1.0.0}"

echo "=== Building $APP_NAME v$VERSION for Linux ==="

# Ensure dependencies
pip install pyinstaller 2>/dev/null || pip3 install pyinstaller

# Run PyInstaller
pyinstaller \
    "$KINECT_SERVER/gui.py" \
    --name "$APP_NAME" \
    --noconfirm \
    --clean \
    --add-data "$KINECT_SERVER/calibration.py:." \
    --add-data "$KINECT_SERVER/camera.py:." \
    --add-data "$KINECT_SERVER/debug_http.py:." \
    --add-data "$KINECT_SERVER/filter.py:." \
    --add-data "$KINECT_SERVER/fusion.py:." \
    --add-data "$KINECT_SERVER/osc_output.py:." \
    --add-data "$KINECT_SERVER/platform_backend.py:." \
    --add-data "$KINECT_SERVER/server.py:." \
    --hidden-import mediapipe \
    --hidden-import cv2 \
    --hidden-import numpy \
    --hidden-import PIL \
    --hidden-import PIL.Image \
    --hidden-import PIL.ImageTk \
    --hidden-import pythonosc \
    --hidden-import flask \
    --collect-data mediapipe \
    --distpath "$DIST" \
    --workpath "$BUILD" \
    --specpath "$BUILD"

echo "✅ PyInstaller build complete: $DIST/$APP_NAME/"

# Copy desktop file
cp "$SCRIPT_DIR/fbt-server.desktop" "$DIST/$APP_NAME/" 2>/dev/null || true

# Try to build AppImage if appimagetool is available
if command -v appimagetool &>/dev/null; then
    echo "=== Building AppImage ==="
    APPDIR="$BUILD/${APP_NAME}.AppDir"
    rm -rf "$APPDIR"
    mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications" "$APPDIR/usr/share/icons/hicolor/256x256/apps"

    # Copy built files
    cp -r "$DIST/$APP_NAME/"* "$APPDIR/usr/bin/"

    # Desktop file
    cat > "$APPDIR/fbt-server.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=FBT Server
Exec=FBT-Server
Icon=fbt-server
Categories=Utility;
Comment=Kinect Full-Body Tracking for VRChat
EOF
    cp "$APPDIR/fbt-server.desktop" "$APPDIR/usr/share/applications/"

    # AppRun script
    cat > "$APPDIR/AppRun" <<'EOF'
#!/bin/bash
HERE="$(dirname "$(readlink -f "${0}")")"
exec "$HERE/usr/bin/FBT-Server" "$@"
EOF
    chmod +x "$APPDIR/AppRun"

    # Create a simple icon (placeholder)
    if command -v convert &>/dev/null; then
        convert -size 256x256 xc:#7c3aed -fill white -gravity center \
            -pointsize 48 -annotate 0 "FBT" \
            "$APPDIR/fbt-server.png" 2>/dev/null || touch "$APPDIR/fbt-server.png"
    else
        touch "$APPDIR/fbt-server.png"
    fi
    cp "$APPDIR/fbt-server.png" "$APPDIR/usr/share/icons/hicolor/256x256/apps/"

    # Build AppImage
    ARCH="$(uname -m)" appimagetool "$APPDIR" "$DIST/${APP_NAME}-${VERSION}-$(uname -m).AppImage"
    echo "✅ AppImage created: $DIST/${APP_NAME}-${VERSION}-$(uname -m).AppImage"
else
    echo "ℹ️  appimagetool not found — skipping AppImage (standalone binary is in $DIST/$APP_NAME/)"
    echo "   Install: https://github.com/AppImage/AppImageKit/releases"
fi

echo ""
echo "=== Build complete ==="
echo "  Standalone: $DIST/$APP_NAME/$APP_NAME"
if [ -f "$DIST/${APP_NAME}-${VERSION}-$(uname -m).AppImage" ]; then
    echo "  AppImage:   $DIST/${APP_NAME}-${VERSION}-$(uname -m).AppImage"
fi

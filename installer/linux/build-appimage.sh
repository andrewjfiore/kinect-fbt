#!/usr/bin/env bash
# Build Marionette-x86_64.AppImage - a single self-contained file the user can
# download and double-click, no install and no terminal.
#
# Usage (from the repo root, after a Release build):
#   installer/linux/build-appimage.sh
#
# Environment overrides:
#   BIN         path to the marionette binary   (default build/app/marionette)
#   DRIVER_DIR  SteamVR driver folder           (default build/dist/marionette)
#   VERSION     version string                  (default 0.1.0)
#   OUTDIR      where to write the AppImage      (default dist)
set -euo pipefail

repo="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
here="$repo/installer/linux"
BIN="${BIN:-$repo/build/app/marionette}"
DRIVER_DIR="${DRIVER_DIR:-$repo/build/dist/marionette}"
VERSION="${VERSION:-0.1.0}"
OUTDIR="${OUTDIR:-$repo/dist}"

[ -x "$BIN" ] || { echo "error: binary not found at $BIN (build Release first)" >&2; exit 1; }

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
appdir="$work/Marionette.AppDir"
mkdir -p "$appdir"

# Payload laid out to match marionette-launch's expectations.
cp "$BIN" "$appdir/marionette"
cp "$here/marionette-launch" "$appdir/marionette-launch"
chmod +x "$appdir/marionette" "$appdir/marionette-launch"
mkdir -p "$appdir/config"
cp "$repo/config/demo.json" "$appdir/config/"

# Bundle any shared libs sitting next to the binary (e.g. libopenvr_api.so),
# which the binary loads through its $ORIGIN rpath.
for so in "$(dirname "$BIN")"/*.so; do
    [ -e "$so" ] && cp "$so" "$appdir/"
done

if [ -f "$DRIVER_DIR/driver.vrdrivermanifest" ]; then
    mkdir -p "$appdir/driver"
    cp -a "$DRIVER_DIR" "$appdir/driver/marionette"
    mkdir -p "$appdir/scripts"
    cp "$repo/scripts/install-driver.sh" "$appdir/scripts/" 2>/dev/null || true
    chmod +x "$appdir/scripts/install-driver.sh" 2>/dev/null || true
fi

# Icon (top-level, named to match the desktop Icon= key) + the .desktop file.
cp "$here/marionette.png" "$appdir/marionette.png"
cat > "$appdir/marionette.desktop" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=Marionette
GenericName=Full-Body Tracking
Comment=Multi-Kinect full-body tracking for SteamVR and VRChat
Exec=marionette-launch
Icon=marionette
Terminal=false
Categories=Game;Utility;
Keywords=VR;tracking;kinect;steamvr;vrchat;fbt;body;
EOF

# AppRun: the entry point AppImage executes on launch.
cat > "$appdir/AppRun" <<'EOF'
#!/usr/bin/env bash
HERE="$(dirname "$(readlink -f "$0")")"
export APPDIR="$HERE"
exec "$HERE/marionette-launch" "$@"
EOF
chmod +x "$appdir/AppRun"

# Fetch appimagetool if it is not already on PATH.
tool="$(command -v appimagetool || true)"
if [ -z "$tool" ]; then
    tool="$work/appimagetool"
    echo "Downloading appimagetool..."
    curl -fsSL -o "$tool" \
        "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage"
    chmod +x "$tool"
fi

mkdir -p "$OUTDIR"
out="$OUTDIR/Marionette-$VERSION-x86_64.AppImage"
# --appimage-extract-and-run avoids needing FUSE on the build machine.
ARCH=x86_64 "$tool" --appimage-extract-and-run "$appdir" "$out" \
    || ARCH=x86_64 "$tool" "$appdir" "$out"
echo "Built $out"

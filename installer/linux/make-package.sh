#!/usr/bin/env bash
# Assemble the portable Linux package tarball (binary + installer + desktop
# integration files) from a Release build.
#
#   installer/linux/make-package.sh
#
# Environment overrides mirror build-appimage.sh:
#   BIN, DRIVER_DIR, VERSION, OUTDIR
set -euo pipefail

repo="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
here="$repo/installer/linux"
BIN="${BIN:-$repo/build/app/marionette}"
DRIVER_DIR="${DRIVER_DIR:-$repo/build/dist/marionette}"
VERSION="${VERSION:-0.1.0}"
OUTDIR="${OUTDIR:-$repo/dist}"

[ -x "$BIN" ] || { echo "error: binary not found at $BIN (build Release first)" >&2; exit 1; }

name="marionette-$VERSION-linux-x86_64"
stage="$(mktemp -d)/$name"
trap 'rm -rf "$(dirname "$stage")"' EXIT
mkdir -p "$stage/config" "$stage/scripts" "$stage/docs"

cp "$BIN" "$stage/marionette"; chmod +x "$stage/marionette"
cp "$here/marionette-launch" "$here/install.sh" "$here/uninstall.sh" "$stage/"
cp "$here/marionette.desktop" "$here/marionette.png" "$stage/"
chmod +x "$stage/marionette-launch" "$stage/install.sh" "$stage/uninstall.sh"
cp "$repo/config/demo.json" "$stage/config/"

# Shared libs the binary loads through its $ORIGIN rpath (e.g. libopenvr_api).
for so in "$(dirname "$BIN")"/*.so; do
    [ -e "$so" ] && cp "$so" "$stage/"
done

if [ -f "$DRIVER_DIR/driver.vrdrivermanifest" ]; then
    mkdir -p "$stage/driver"
    cp -a "$DRIVER_DIR" "$stage/driver/marionette"
    cp "$repo/scripts/install-driver.sh" "$stage/scripts/" 2>/dev/null || true
    chmod +x "$stage/scripts/install-driver.sh" 2>/dev/null || true
fi

for d in USER_GUIDE.md INSTALL.md USAGE.md CALIBRATION.md TUTORIAL.md; do
    [ -f "$repo/docs/$d" ] && cp "$repo/docs/$d" "$stage/docs/"
done
cp "$repo/LICENSE" "$stage/" 2>/dev/null || true
cp "$repo/README.md" "$stage/" 2>/dev/null || true

mkdir -p "$OUTDIR"
tar -C "$(dirname "$stage")" -czf "$OUTDIR/$name.tar.gz" "$name"
echo "Built $OUTDIR/$name.tar.gz"

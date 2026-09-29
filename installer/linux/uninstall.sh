#!/usr/bin/env bash
# Remove a Marionette installation created by install.sh.
#   ./uninstall.sh [--prefix DIR]
# Leaves your per-user data (~/.local/share/marionette: config + calibration)
# in place; delete that folder by hand if you want a completely clean slate.
set -euo pipefail

prefix="$HOME/.local"
while [ $# -gt 0 ]; do
    case "$1" in
        --prefix) prefix="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

app_dir="$prefix/opt/marionette"
desktop_file="$prefix/share/applications/marionette.desktop"
icon_file="$prefix/share/icons/hicolor/256x256/apps/marionette.png"

# Unregister the SteamVR driver first, while the script is still present.
if [ -x "$app_dir/scripts/install-driver.sh" ] \
   && [ -d "$app_dir/driver/marionette" ]; then
    "$app_dir/scripts/install-driver.sh" --remove "$app_dir/driver/marionette" >/dev/null 2>&1 || true
    echo "  SteamVR driver unregistered"
fi

rm -rf "$app_dir" && echo "  removed $app_dir"
rm -f "$desktop_file" && echo "  removed menu entry"
rm -f "$icon_file" || true
command -v update-desktop-database >/dev/null 2>&1 \
    && update-desktop-database "$prefix/share/applications" >/dev/null 2>&1 || true

echo "Uninstalled. Your settings remain in ${XDG_DATA_HOME:-$HOME/.local/share}/marionette."

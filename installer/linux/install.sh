#!/usr/bin/env bash
# Marionette installer for Linux (desktop integration, no root required).
#
# Run it from the extracted package:  ./install.sh
# Most file managers also offer "Run" on right-click.
#
# It copies the app into ~/.local/opt/marionette, adds a "Marionette" entry to
# your applications menu with an icon, and - when the SteamVR driver is bundled
# and SteamVR is installed - registers the driver. Everything lands under your
# home directory; nothing needs sudo.
#
# Options:
#   --prefix DIR   install under DIR instead of ~/.local
#   --no-driver    skip SteamVR driver registration
#   --uninstall    remove a previous installation
set -euo pipefail

prefix="$HOME/.local"
want_driver=1
do_uninstall=0
while [ $# -gt 0 ]; do
    case "$1" in
        --prefix) prefix="$2"; shift 2 ;;
        --no-driver) want_driver=0; shift ;;
        --uninstall) do_uninstall=1; shift ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

src="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
app_dir="$prefix/opt/marionette"
apps_dir="$prefix/share/applications"
icon_dir="$prefix/share/icons/hicolor/256x256/apps"
desktop_file="$apps_dir/marionette.desktop"

log() { printf '  %s\n' "$*"; }
notify() { command -v notify-send >/dev/null 2>&1 && notify-send "Marionette" "$1" || true; }

if [ "$do_uninstall" -eq 1 ]; then
    exec "$src/uninstall.sh" --prefix "$prefix"
fi

echo "Installing Marionette into $app_dir"

# 1. Copy the payload.
mkdir -p "$app_dir"
cp -f "$src/marionette" "$app_dir/" 2>/dev/null || { echo "error: marionette binary missing from package" >&2; exit 1; }
cp -f "$src/marionette-launch" "$app_dir/"
chmod +x "$app_dir/marionette" "$app_dir/marionette-launch"
mkdir -p "$app_dir/config"
cp -f "$src/config/demo.json" "$app_dir/config/" 2>/dev/null || true
if [ -d "$src/driver" ]; then cp -a "$src/driver" "$app_dir/"; fi
if [ -d "$src/scripts" ]; then cp -a "$src/scripts" "$app_dir/"; chmod +x "$app_dir"/scripts/*.sh 2>/dev/null || true; fi
if [ -d "$src/docs" ]; then cp -a "$src/docs" "$app_dir/"; fi
cp -f "$src/LICENSE" "$app_dir/" 2>/dev/null || true
log "app files copied"

# 2. Icon.
mkdir -p "$icon_dir"
cp -f "$src/marionette.png" "$icon_dir/marionette.png" 2>/dev/null && log "icon installed" || true

# 3. Desktop entry (menu launcher, no terminal).
mkdir -p "$apps_dir"
sed -e "s|@EXEC@|$app_dir/marionette-launch|g" \
    -e "s|@ICON@|marionette|g" \
    "$src/marionette.desktop" > "$desktop_file"
chmod +x "$desktop_file" 2>/dev/null || true
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$apps_dir" >/dev/null 2>&1 || true
command -v gtk-update-icon-cache >/dev/null 2>&1 && gtk-update-icon-cache -q "$prefix/share/icons/hicolor" >/dev/null 2>&1 || true
log "menu entry installed ($desktop_file)"

# 4. SteamVR driver registration (best effort).
if [ "$want_driver" -eq 1 ] && [ -f "$app_dir/driver/marionette/driver.vrdrivermanifest" ] \
   && [ -x "$app_dir/scripts/install-driver.sh" ]; then
    if "$app_dir/scripts/install-driver.sh" "$app_dir/driver/marionette" >/dev/null 2>&1; then
        log "SteamVR driver registered"
    else
        log "SteamVR driver not registered (SteamVR not found - run scripts/install-driver.sh later)"
    fi
fi

echo
echo "Done. Launch 'Marionette' from your applications menu."
notify "Installed. Launch 'Marionette' from your applications menu."

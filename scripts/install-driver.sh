#!/usr/bin/env bash
# install-driver.sh - register (or remove) the Marionette SteamVR driver via vrpathreg.
#
# Usage:
#   ./install-driver.sh                       # adddriver on <repo>/build/dist/marionette
#   ./install-driver.sh /path/to/dist         # adddriver on an explicit dist folder
#   ./install-driver.sh --remove [path]       # removedriver
set -euo pipefail

usage() {
    echo "usage: $0 [--remove] [driver_path]" >&2
    exit 2
}

remove=0
driver_path=""
for arg in "$@"; do
    case "$arg" in
        -r|--remove) remove=1 ;;
        -h|--help) usage ;;
        -*) echo "unknown option: $arg" >&2; usage ;;
        *) driver_path="$arg" ;;
    esac
done

script_dir="$(cd "$(dirname "$0")" && pwd)"
if [ -z "$driver_path" ]; then
    driver_path="$script_dir/../build/dist/marionette"
fi
if [ ! -d "$driver_path" ]; then
    echo "error: driver folder not found: $driver_path (build the project first)" >&2
    exit 1
fi
driver_path="$(cd "$driver_path" && pwd)"
if [ ! -f "$driver_path/driver.vrdrivermanifest" ]; then
    echo "error: no driver.vrdrivermanifest in $driver_path" >&2
    exit 1
fi

# Collect candidate Steam libraries: the two usual roots plus any extra
# libraries listed in libraryfolders.vdf.
steam_roots=()
for root in "$HOME/.steam/steam" "$HOME/.local/share/Steam"; do
    [ -d "$root" ] && steam_roots+=("$root")
done

libraries=()
for root in "${steam_roots[@]+"${steam_roots[@]}"}"; do
    libraries+=("$root")
    vdf="$root/steamapps/libraryfolders.vdf"
    if [ -f "$vdf" ]; then
        while IFS= read -r lib; do
            [ -d "$lib" ] && libraries+=("$lib")
        done < <(sed -n 's/^[[:space:]]*"path"[[:space:]]*"\(.*\)"[[:space:]]*$/\1/p' "$vdf")
    fi
done

steamvr_dir=""
for lib in "${libraries[@]+"${libraries[@]}"}"; do
    candidate="$lib/steamapps/common/SteamVR"
    if [ -d "$candidate" ]; then
        steamvr_dir="$candidate"
        break
    fi
done

if [ -z "$steamvr_dir" ]; then
    echo "error: SteamVR installation not found under ~/.steam/steam or ~/.local/share/Steam" >&2
    exit 1
fi

verb="adddriver"
[ "$remove" -eq 1 ] && verb="removedriver"

echo "steamvr: $steamvr_dir"
echo "driver:  $driver_path"

# Prefer the vrpathreg.sh wrapper (sets up the runtime environment); fall back
# to calling the linux64 binary with its library dir on LD_LIBRARY_PATH.
if [ -x "$steamvr_dir/bin/vrpathreg.sh" ]; then
    "$steamvr_dir/bin/vrpathreg.sh" "$verb" "$driver_path"
elif [ -x "$steamvr_dir/bin/linux64/vrpathreg" ]; then
    LD_LIBRARY_PATH="$steamvr_dir/bin/linux64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$steamvr_dir/bin/linux64/vrpathreg" "$verb" "$driver_path"
else
    echo "error: vrpathreg not found under $steamvr_dir/bin" >&2
    exit 1
fi

if [ "$remove" -eq 1 ]; then
    echo "Removed Marionette driver registration."
else
    echo "Registered Marionette driver. Restart SteamVR to load it."
fi

#!/bin/bash
set -e
APK="$(dirname "$0")/app/build/outputs/apk/debug/app-debug.apk"
if [ ! -f "$APK" ]; then
    echo "APK not found. Run build.sh first."
    exit 1
fi
adb install -r -d "$APK"

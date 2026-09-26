#!/bin/zsh
# Builds the SwiftPM executable and wraps it in a minimal .app bundle.
set -euo pipefail
cd "$(dirname "$0")"

CONFIG="${1:-release}"
APP="build/Pomodoro.app"

# The Command Line Tools toolchain can't initialise the newer Swift Build backend; the native one works fine.
swift build -c "$CONFIG" --build-system native 2>&1 | grep -v "build-system native' has been deprecated"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp ".build/$CONFIG/Pomodoro" "$APP/Contents/MacOS/Pomodoro"
cp Resources/Info.plist "$APP/Contents/Info.plist"
printf 'APPL????' > "$APP/Contents/PkgInfo"
codesign --force --sign - "$APP" >/dev/null 2>&1 || echo "warning: ad-hoc codesign failed (app will still run)"

echo "Built $APP"

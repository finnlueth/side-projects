#!/bin/zsh
# Regenerates Resources/AppIcon.icns from scripts/make_icon.swift.
set -euo pipefail
cd "$(dirname "$0")/.."

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

swiftc -O -parse-as-library scripts/make_icon.swift -o "$TMP/make_icon"
"$TMP/make_icon" "$TMP"
iconutil -c icns "$TMP/AppIcon.iconset" -o Resources/AppIcon.icns
cp "$TMP/AppIcon-preview.png" Resources/AppIcon-preview.png
echo "wrote Resources/AppIcon.icns"

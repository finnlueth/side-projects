#!/usr/bin/env bash
# Package the extension as an .xpi (a plain zip) in ./dist
set -euo pipefail

cd "$(dirname "$0")"

VERSION=$(python3 -c "import json; print(json.load(open('manifest.json'))['version'])")
OUT="dist/socialfocus-custom-${VERSION}.xpi"

mkdir -p dist
rm -f "$OUT"

zip -q -r -X "$OUT" . \
  -x "dist/*" "build.sh" "README.md" ".gitignore" ".DS_Store" "*/.DS_Store" "manifestsFiles.txt"

echo "built $OUT"

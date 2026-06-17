#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNAME="$(uname -s 2>/dev/null || echo unknown)"

case "$UNAME" in
  Linux*)
    exec "$ROOT_DIR/build-ubuntu.sh" "$@"
    ;;
  Darwin*)
    exec "$ROOT_DIR/build-macos.sh" "$@"
    ;;
  *)
    cat >&2 <<'EOF'
build.sh supports Linux and macOS.

On macOS, use:
  ./build-macos.sh
  ./build-macos.sh --wheel
  ./build-macos.sh --monitor

On Windows, use:
  powershell -ExecutionPolicy Bypass -File .\build-windows.ps1

For wheel packaging on Windows, use:
  powershell -ExecutionPolicy Bypass -File .\build-windows.ps1 -Mode wheel
EOF
    exit 1
    ;;
esac

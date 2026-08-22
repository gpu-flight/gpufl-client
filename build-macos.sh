#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
MODE="install"
WHEEL_DIR="$ROOT_DIR/dist"
BUILD_DIR="$ROOT_DIR/build-macos-monitor"
MONITOR_LOG_DIR="${GPUFL_MONITOR_LOG_DIR:-$ROOT_DIR/gpufl-monitor-macos/session}"
OPENSSL_ROOT="${OPENSSL_ROOT_DIR:-}"

if [[ -z "$OPENSSL_ROOT" ]] && command -v brew >/dev/null 2>&1; then
  OPENSSL_ROOT="$(brew --prefix openssl@3 2>/dev/null || true)"
fi

usage() {
  cat <<'EOF'
Usage: ./build-macos.sh [--install|--wheel|--monitor] [--python PATH] [--wheel-dir PATH] [--build-dir PATH]

Defaults:
  --install
  --python    ${PYTHON:-python3}
  --wheel-dir ./dist
  --build-dir ./build-macos-monitor

Examples:
  ./build-macos.sh
  ./build-macos.sh --wheel
  ./build-macos.sh --python .venv/bin/python
  ./build-macos.sh --monitor
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install)
      MODE="install"
      shift
      ;;
    --wheel)
      MODE="wheel"
      shift
      ;;
    --monitor)
      MODE="monitor"
      shift
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --wheel-dir)
      WHEEL_DIR="$2"
      shift 2
      ;;
    --build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Darwin" ]]; then
  echo "build-macos.sh is only supported on macOS." >&2
  exit 2
fi

COMMON_CONFIG=(
  -C cmake.define.BUILD_PYTHON=ON
  -C cmake.define.BUILD_GPUFL_EXAMPLE=OFF
  -C cmake.define.BUILD_TESTING=OFF
  -C cmake.define.PYBIND11_FINDPYTHON=ON
  -C cmake.define.GPUFL_ENABLE_NVIDIA=OFF
  -C cmake.define.GPUFL_ENABLE_AMD=OFF
  -C cmake.define.GPUFL_ENABLE_METAL=ON
)

OPENSSL_CMAKE_ARG=()
if [[ -n "$OPENSSL_ROOT" ]]; then
  COMMON_CONFIG+=(
    -C "cmake.define.OPENSSL_ROOT_DIR=$OPENSSL_ROOT"
    -C "cmake.define.OPENSSL_INCLUDE_DIR=$OPENSSL_ROOT/include"
    -C "cmake.define.OPENSSL_SSL_LIBRARY=$OPENSSL_ROOT/lib/libssl.dylib"
    -C "cmake.define.OPENSSL_CRYPTO_LIBRARY=$OPENSSL_ROOT/lib/libcrypto.dylib"
  )
  OPENSSL_CMAKE_ARG=(
    "-DOPENSSL_ROOT_DIR=$OPENSSL_ROOT"
    "-DOPENSSL_INCLUDE_DIR=$OPENSSL_ROOT/include"
    "-DOPENSSL_SSL_LIBRARY=$OPENSSL_ROOT/lib/libssl.dylib"
    "-DOPENSSL_CRYPTO_LIBRARY=$OPENSSL_ROOT/lib/libcrypto.dylib"
  )
fi

echo "GPUFlight macOS build"
echo "  mode:   $MODE"
echo "  python: $PYTHON_BIN"
if [[ -n "$OPENSSL_ROOT" ]]; then
  echo "  OpenSSL: $OPENSSL_ROOT"
fi

if [[ "$MODE" == "wheel" ]]; then
  mkdir -p "$WHEEL_DIR"
  "$PYTHON_BIN" -m pip wheel "$ROOT_DIR" -w "$WHEEL_DIR" --no-deps -v "${COMMON_CONFIG[@]}"
elif [[ "$MODE" == "monitor" ]]; then
  echo "  build:  $BUILD_DIR"
  cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_GPUFL_EXAMPLE=OFF \
    -DBUILD_TESTING=OFF \
    -DBUILD_PYTHON=OFF \
    -DBUILD_GPUFL_MONITOR=ON \
    -DBUILD_GPUFL_LAUNCHER=OFF \
    -DBUILD_GPUFL_INJECT=OFF \
    -DGPUFL_ENABLE_NVIDIA=OFF \
    -DGPUFL_ENABLE_AMD=OFF \
    -DGPUFL_ENABLE_METAL=ON \
    "${OPENSSL_CMAKE_ARG[@]}"
  cmake --build "$BUILD_DIR" --target gpufl-monitor --parallel
  echo ""
  echo "Built native monitor:"
  echo "  $BUILD_DIR/daemon/monitor/gpufl-monitor"
  echo ""
  echo "Run:"
  echo "  GPUFL_MONITOR_LOG_DIR=\"$MONITOR_LOG_DIR\" \\"
  echo "    GPUFL_MONITOR_BACKEND=metal \"$BUILD_DIR/daemon/monitor/gpufl-monitor\""
  echo ""
  echo "Build and run with defaults:"
  echo "  $ROOT_DIR/scripts/run-monitor-macos.sh"
else
  "$PYTHON_BIN" -m pip install "$ROOT_DIR" -v "${COMMON_CONFIG[@]}"
fi

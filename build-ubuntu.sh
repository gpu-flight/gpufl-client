#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
CUDA_ROOT="${CUDA_ROOT:-${CUDA_PATH:-${CUDA_HOME:-/usr/local/cuda}}}"
MODE="install"
WHEEL_DIR="$ROOT_DIR/dist"

usage() {
  cat <<'EOF'
Usage: ./build-ubuntu.sh [--install|--wheel] [--python PATH] [--cuda-root PATH] [--wheel-dir PATH]

Defaults:
  --install
  --python    ${PYTHON:-python3}
  --cuda-root ${CUDA_ROOT:-${CUDA_PATH:-${CUDA_HOME:-/usr/local/cuda}}}
  --wheel-dir ./dist

Examples:
  ./build-ubuntu.sh
  ./build-ubuntu.sh --wheel
  ./build-ubuntu.sh --python .venv/bin/python --cuda-root /usr/local/cuda-13.2
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
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --cuda-root)
      CUDA_ROOT="$2"
      shift 2
      ;;
    --wheel-dir)
      WHEEL_DIR="$2"
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

if [[ ! -x "$CUDA_ROOT/bin/nvcc" ]]; then
  for candidate in /usr/local/cuda /usr/local/cuda-13.2 /usr/local/cuda-13.1 /usr/local/cuda-13.0; do
    if [[ -x "$candidate/bin/nvcc" ]]; then
      CUDA_ROOT="$candidate"
      break
    fi
  done
fi

if [[ ! -x "$CUDA_ROOT/bin/nvcc" ]]; then
  echo "CUDA nvcc was not found under: $CUDA_ROOT" >&2
  echo "Pass --cuda-root PATH or set CUDA_ROOT/CUDA_PATH/CUDA_HOME." >&2
  exit 1
fi

export CUDA_ROOT
export CUDA_PATH="$CUDA_ROOT"
export CUDA_HOME="$CUDA_ROOT"
export CUDACXX="$CUDA_ROOT/bin/nvcc"
export PATH="$CUDA_ROOT/bin:$CUDA_ROOT/extras/CUPTI/lib64:$PATH"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:$CUDA_ROOT/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}"

COMMON_CONFIG=(
  -C cmake.define.BUILD_PYTHON=ON
  -C cmake.define.BUILD_GPUFL_EXAMPLE=OFF
  -C cmake.define.BUILD_TESTING=OFF
  -C cmake.define.PYBIND11_FINDPYTHON=ON
  -C cmake.define.GPUFL_ENABLE_NVIDIA=ON
  -C cmake.define.GPUFL_ENABLE_AMD=OFF
  -C "cmake.define.CUDAToolkit_ROOT=$CUDA_ROOT"
  -C "cmake.define.CMAKE_CUDA_COMPILER=$CUDA_ROOT/bin/nvcc"
)

echo "GPUFlight build"
echo "  mode:      $MODE"
echo "  python:    $PYTHON_BIN"
echo "  cuda root: $CUDA_ROOT"

if [[ "$MODE" == "wheel" ]]; then
  mkdir -p "$WHEEL_DIR"
  "$PYTHON_BIN" -m pip wheel "$ROOT_DIR" -w "$WHEEL_DIR" --no-deps -v "${COMMON_CONFIG[@]}"
else
  "$PYTHON_BIN" -m pip install "$ROOT_DIR" -v "${COMMON_CONFIG[@]}"
fi

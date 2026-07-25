#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
CUDA_ROOT="${CUDA_ROOT:-${CUDA_PATH:-${CUDA_HOME:-/usr/local/cuda}}}"
MODE="install"
WHEEL_DIR="$ROOT_DIR/dist"

usage() {
  cat <<'EOF'
Usage: ./build-ubuntu.sh [--install|--wheel|--trace] [--python PATH] [--cuda-root PATH] [--wheel-dir PATH]

Defaults:
  --install
  --python    ${PYTHON:-python3}
  --cuda-root ${CUDA_ROOT:-${CUDA_PATH:-${CUDA_HOME:-/usr/local/cuda}}}
  --wheel-dir ./dist

Examples:
  ./build-ubuntu.sh
  ./build-ubuntu.sh --wheel
  ./build-ubuntu.sh --trace
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
    --trace)
      MODE="trace"
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
elif [[ "$MODE" == "trace" ]]; then
  BUILD_DIR="$ROOT_DIR/build-ubuntu"
  TRACE_CONFIG=(
    -DCMAKE_BUILD_TYPE=Release
    -DGPUFL_ENABLE_NVIDIA=ON
    -DGPUFL_ENABLE_AMD=OFF
    -DBUILD_PYTHON=OFF
    -DBUILD_TESTING=OFF
    -DBUILD_GPUFL_EXAMPLE=OFF
    -DBUILD_GPUFL_LAUNCHER=ON
    -DBUILD_GPUFL_INJECT=ON
    "-DCUDAToolkit_ROOT=$CUDA_ROOT"
    "-DCMAKE_CUDA_COMPILER=$CUDA_ROOT/bin/nvcc"
  )

  cmake -S "$ROOT_DIR" -B "$BUILD_DIR" "${TRACE_CONFIG[@]}"
  cmake --build "$BUILD_DIR" --target gpufl_launcher gpufl_inject -j

  LAUNCHER="$BUILD_DIR/daemon/launcher/gpufl"
  INJECT_LIBRARY="$BUILD_DIR/libgpufl_inject.so"
  echo
  echo "Built native trace tooling:"
  echo "  launcher: $LAUNCHER"
  echo "  inject:   $INJECT_LIBRARY"
  echo
  echo "Run:  \"$LAUNCHER\" trace --passes=Trace -- \"$PYTHON_BIN\" <script.py>"
  echo "      \"$LAUNCHER\" trace --passes=PcSampling -- \"$PYTHON_BIN\" <script.py>"
else
  "$PYTHON_BIN" -m pip install "$ROOT_DIR" -v "${COMMON_CONFIG[@]}"
fi

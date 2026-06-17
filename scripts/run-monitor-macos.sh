#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${GPUFL_MONITOR_BUILD_DIR:-$ROOT_DIR/build-macos-monitor}"

cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_GPUFL_EXAMPLE=OFF \
  -DBUILD_TESTING=OFF \
  -DBUILD_PYTHON=OFF \
  -DBUILD_GPUFL_MONITOR=ON \
  -DGPUFL_ENABLE_NVIDIA=OFF \
  -DGPUFL_ENABLE_AMD=OFF \
  -DGPUFL_ENABLE_METAL=ON

cmake --build "$BUILD_DIR" --target gpufl-monitor --parallel

export GPUFL_MONITOR_BACKEND="${GPUFL_MONITOR_BACKEND:-metal}"
export GPUFL_MONITOR_APP="${GPUFL_MONITOR_APP:-gpufl-monitor-macos}"
export GPUFL_MONITOR_LOG_DIR="${GPUFL_MONITOR_LOG_DIR:-$ROOT_DIR/gpufl-monitor-macos/session}"
export GPUFL_MONITOR_INTERVAL_MS="${GPUFL_MONITOR_INTERVAL_MS:-5000}"

mkdir -p "$GPUFL_MONITOR_LOG_DIR"

exec "$BUILD_DIR/daemon/monitor/gpufl-monitor"

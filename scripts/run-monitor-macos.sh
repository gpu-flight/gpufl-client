#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${GPUFL_MONITOR_BUILD_DIR:-$ROOT_DIR/build-macos-monitor}"

"$ROOT_DIR/build-macos.sh" --monitor --build-dir "$BUILD_DIR"

export GPUFL_MONITOR_BACKEND="${GPUFL_MONITOR_BACKEND:-metal}"
export GPUFL_MONITOR_APP="${GPUFL_MONITOR_APP:-gpufl-monitor-macos}"
export GPUFL_MONITOR_LOG_DIR="${GPUFL_MONITOR_LOG_DIR:-$ROOT_DIR/gpufl-monitor-macos/session}"
export GPUFL_MONITOR_INTERVAL_MS="${GPUFL_MONITOR_INTERVAL_MS:-5000}"

mkdir -p "$GPUFL_MONITOR_LOG_DIR"

exec "$BUILD_DIR/daemon/monitor/gpufl-monitor"

#!/bin/bash
# Runs the occupancy_demo in a loop, writing logs to the shared volume
# so the gpufl-agent can tail and forward them to the backend.

LOG_DIR="${GPUFL_DEMO_LOG_DIR:-/var/gpufl/demo}"
INTERVAL="${GPUFL_DEMO_INTERVAL_SEC:-30}"

echo "[demo] Writing logs to ${LOG_DIR}, interval=${INTERVAL}s"

cd "${LOG_DIR}"

while true; do
    echo "[demo] Starting occupancy_demo..."
    occupancy_demo
    echo "[demo] Done. Sleeping ${INTERVAL}s..."
    sleep "${INTERVAL}"
done

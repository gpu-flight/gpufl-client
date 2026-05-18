@echo off
REM Run gpufl-monitor + demo workload via docker-compose.
REM The demo runs occupancy_demo in a loop, generating real GPU events.
REM The monitor tails the logs and sends them to the backend.
REM
REM Usage:
REM   set GPUFL_HTTP_TOKEN=gfl_xxx
REM   run-monitor-local.bat

if not defined GPUFL_HTTP_TOKEN (
    echo WARNING: GPUFL_HTTP_TOKEN is not set. Set it to your API key.
    echo Example: set GPUFL_HTTP_TOKEN=gfl_xxxx
    echo.
)

REM Just the scheme+host — the agent appends /api/{version}/events/<type>.
REM Override the API version with GPUFL_HTTP_API_VERSION when needed.
set GPUFL_HTTP_HOST=http://host.docker.internal:8080

pushd %~dp0..\..
docker compose -f docker-compose.monitor.yml up --build
popd

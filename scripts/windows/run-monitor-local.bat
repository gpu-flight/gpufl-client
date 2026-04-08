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

set GPUFL_HTTP_URL=http://host.docker.internal:8080/api/v1/events/

pushd %~dp0..\..
docker compose -f docker-compose.monitor.yml up --build
popd

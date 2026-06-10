#pragma once

#include <string>

namespace gpufl::daemon {

struct MonitorRunOptions {
    std::string app_name = "gpufl-monitor";
    std::string log_path;
    int interval_ms = 5000;
    bool quiet = false;
};

// Runs telemetry-only monitoring in the current process until SIGINT/SIGTERM.
// Returns 0 on clean shutdown and 3 when gpufl::init() cannot start.
int runMonitorForeground(const MonitorRunOptions& opts);

}  // namespace gpufl::daemon

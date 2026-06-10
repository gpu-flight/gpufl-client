#include <cstdlib>
#include <iostream>
#include <string>

#include "gpufl/core/env_vars.hpp"
#include "monitor_runner.hpp"

namespace {

std::string getenv_or(const char* var, const char* fallback) {
    const char* val = std::getenv(var);
    return val ? std::string(val) : std::string(fallback);
}

int getenv_int_or(const char* var, int fallback) {
    const char* val = std::getenv(var);
    if (!val || !*val) return fallback;
    try {
        return std::stoi(val);
    } catch (...) {
        return fallback;
    }
}

}  // namespace

int main() {
    gpufl::daemon::MonitorRunOptions opts;
    opts.app_name = getenv_or(gpufl::env::kMonitorApp, "gpufl-monitor");
    opts.log_path = getenv_or(gpufl::env::kMonitorLogDir,
                              "/var/gpufl/monitor/session");
    opts.interval_ms = getenv_int_or(gpufl::env::kMonitorIntervalMs, 5000);

    if (opts.interval_ms <= 0) {
        std::cerr << "GPUFL_MONITOR_INTERVAL_MS must be positive.\n";
        return 2;
    }

    return gpufl::daemon::runMonitorForeground(opts);
}

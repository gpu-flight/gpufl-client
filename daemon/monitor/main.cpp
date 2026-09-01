#include <algorithm>
#include <cctype>
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

gpufl::BackendKind parseBackend() {
    const char* raw = std::getenv(gpufl::env::kMonitorBackend);
    if (!raw || raw[0] == '\0') {
#if defined(__APPLE__)
        return gpufl::BackendKind::Metal;
#else
        return gpufl::BackendKind::Auto;
#endif
    }

    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    if (value == "auto") return gpufl::BackendKind::Auto;
    if (value == "nvidia") return gpufl::BackendKind::Nvidia;
    if (value == "amd") return gpufl::BackendKind::Amd;
    if (value == "metal") return gpufl::BackendKind::Metal;
    if (value == "none") return gpufl::BackendKind::None;

    std::cerr << "Unrecognized " << gpufl::env::kMonitorBackend << "='" << raw
              << "'; expected auto, nvidia, amd, metal, or none. Using auto.\n";
    return gpufl::BackendKind::Auto;
}

}  // namespace

int main() {
    gpufl::daemon::MonitorRunOptions opts;
    opts.app_name = getenv_or(gpufl::env::kMonitorApp, "gpufl-monitor");
    opts.log_path = getenv_or(gpufl::env::kMonitorLogDir,
                              "/var/gpufl/monitor/session");
    opts.interval_ms = getenv_int_or(gpufl::env::kMonitorIntervalMs, 5000);
    opts.backend = parseBackend();

    if (opts.interval_ms <= 0) {
        std::cerr << "GPUFL_MONITOR_INTERVAL_MS must be positive.\n";
        return 2;
    }

    return gpufl::daemon::runMonitorForeground(opts);
}

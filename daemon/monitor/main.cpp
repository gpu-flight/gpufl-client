#include <csignal>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

#ifndef _WIN32
#include <pthread.h>
#endif

#include "gpufl/gpufl.hpp"

namespace {

std::string getenv_or(const char* var, const char* fallback) {
    const char* val = std::getenv(var);
    return val ? std::string(val) : std::string(fallback);
}

#ifdef _WIN32
volatile std::sig_atomic_t g_shutdownRequested = 0;

void signalHandler(int /*sig*/) {
    g_shutdownRequested = 1;
}

bool blockTerminationSignals() {
    return true;
}

bool waitForShutdownSignal() {
    while (!g_shutdownRequested) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return true;
}
#else
bool blockTerminationSignals() {
    sigset_t set;
    if (::sigemptyset(&set) != 0) return false;
    if (::sigaddset(&set, SIGINT) != 0) return false;
    if (::sigaddset(&set, SIGTERM) != 0) return false;

    return ::pthread_sigmask(SIG_BLOCK, &set, nullptr) == 0;
}

bool waitForShutdownSignal() {
    sigset_t set;
    if (::sigemptyset(&set) != 0) return false;
    if (::sigaddset(&set, SIGINT) != 0) return false;
    if (::sigaddset(&set, SIGTERM) != 0) return false;

    int sig = 0;
    const int rc = ::sigwait(&set, &sig);
    return rc == 0;
}
#endif

}  // namespace

int main() {
#ifdef _WIN32
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
#endif

    if (!blockTerminationSignals()) {
        std::cerr << "Failed to block termination signals: "
                  << std::strerror(errno) << '\n';
        return 1;
    }

    const std::string appName = getenv_or("GPUFL_MONITOR_APP", "gpufl-monitor");
    const std::string logPath =
        getenv_or("GPUFL_MONITOR_LOG_DIR", "/var/gpufl/monitor/session");
    const int intervalMs =
        std::stoi(getenv_or("GPUFL_MONITOR_INTERVAL_MS", "5000"));

    gpufl::InitOptions opts;
    opts.app_name = appName;
    opts.log_path = logPath;
    opts.profiling_engine = gpufl::ProfilingEngine::None;
    opts.sampling_auto_start = true;
    opts.system_sample_rate_ms = intervalMs;
    opts.enable_debug_output = true;
    opts.flush_logs_always = true;

    if (!gpufl::init(opts)) {
        return 1;
    }

    if (!waitForShutdownSignal()) {
        std::cerr << "Failed while waiting for termination signal.\n";
        gpufl::shutdown();
        return 1;
    }

    gpufl::shutdown();
    return 0;
}

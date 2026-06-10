#include "monitor_runner.hpp"

#include <cerrno>
#include <csignal>
#include <cstring>
#include <iostream>
#include <thread>

#ifndef _WIN32
#include <pthread.h>
#endif

#include "gpufl/gpufl.hpp"

namespace gpufl::daemon {
namespace {

#ifdef _WIN32
volatile std::sig_atomic_t g_shutdownRequested = 0;

void signalHandler(int /*sig*/) {
    g_shutdownRequested = 1;
}

bool setupSignalHandling() {
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    return true;
}

bool waitForShutdownSignal() {
    while (!g_shutdownRequested) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return true;
}
#else
bool setupSignalHandling() {
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

int runMonitorForeground(const MonitorRunOptions& opts) {
    if (!setupSignalHandling()) {
        std::cerr << "gpufl monitor: failed to set signal handling: "
                  << std::strerror(errno) << '\n';
        return 3;
    }

    if (!opts.quiet) {
        std::cerr << "[gpufl] monitor sampling -> " << opts.log_path
                  << " every " << opts.interval_ms << "ms\n";
    }

    gpufl::InitOptions init;
    init.app_name = opts.app_name.empty() ? "gpufl-monitor" : opts.app_name;
    init.log_path = opts.log_path;
    init.profiling_engine = gpufl::ProfilingEngine::Monitor;
    init.continuous_system_sampling = true;
    init.system_sample_rate_ms = opts.interval_ms;
    init.enable_debug_output = false;
    init.enable_stack_trace = false;
    init.enable_source_collection = false;
    init.enable_synchronization = false;
    init.enable_memory_tracking = false;
    init.enable_cuda_graphs_tracking = false;
    init.enable_external_correlation = false;
    init.flush_logs_always = true;

    if (!gpufl::init(init)) {
        return 3;
    }

    const bool wait_ok = waitForShutdownSignal();
    gpufl::shutdown();

    if (!wait_ok) {
        std::cerr << "gpufl monitor: failed while waiting for termination signal.\n";
        return 3;
    }
    if (!opts.quiet) {
        std::cerr << "[gpufl] monitor stopped\n";
    }
    return 0;
}

}  // namespace gpufl::daemon

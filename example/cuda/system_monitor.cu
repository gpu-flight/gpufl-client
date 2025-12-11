#include <gpumon/gpumon.hpp>
#include <thread>
#include <chrono>
#include <iostream>

int main(int argc, char** argv) {
    // 1. Configure for System Monitoring
    gpumon::InitOptions opts;
    opts.appName = "SystemMonitor";

    // Optional: Set a specific log path or let it default
    opts.logPath = "gpumon_system.log";

    gpumon::init(opts);

    std::cout << "Starting GPU System Monitor (Ctrl+C to stop)..." << std::endl;
    GPUMON_SYSTEM_START(1000);

    gpumon::shutdown();
    return 0;
}
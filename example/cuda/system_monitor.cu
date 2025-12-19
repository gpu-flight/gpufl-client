#include <gpufl/gpufl.hpp>
#include <thread>
#include <chrono>
#include <iostream>

int main(int argc, char** argv) {
    // 1. Configure for System Monitoring
    gpufl::InitOptions opts;
    opts.appName = "SystemMonitor";

    // Optional: Set a specific log path or let it default
    opts.logPath = "gpufl_system.log";

    gpufl::init(opts);

    std::cout << "Starting GPU System Monitor (Ctrl+C to stop)..." << std::endl;
    GFL_SYSTEM_START(1000, "system");

    gpufl::shutdown();
    return 0;
}
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace gpufl {
    struct HostSample {
        double cpuUtilPercent = 0.0; // System-wide CPU usage (0.0 - 100.0)
        uint64_t ramUsedMiB = 0;
        uint64_t ramTotalMiB = 0;
    };

    struct DeviceSample {
        int deviceId = 0;
        std::string name;
        std::string uuid;
        int pciBusId = 0;

        size_t freeMiB = 0;
        size_t totalMiB = 0;
        size_t usedMiB = 0;

        unsigned int gpuUtil = 0;      // %
        unsigned int memUtil = 0;      // %
        unsigned int tempC = 0;        // Celsius
        unsigned int powerMw = 0;      // Milliwatts
        unsigned int clockGfx = 0;     // MHz
        unsigned int clockSm = 0;      // MHz
        unsigned int clockMem = 0;     // MHz
    };

    struct InitEvent {
        int pid = 0;
        std::string app;
        std::string logPath;
        int64_t tsNs = 0;
        HostSample host;
        std::vector<DeviceSample> devices;
    };

    struct ShutdownEvent {
        int pid = 0;
        std::string app;
        int64_t tsNs = 0;
    };

    struct KernelBeginEvent {
        int pid = 0;
        std::string app;
        std::string name;
        std::string tag;

        int64_t tsStartNs = 0;
        int64_t tsEndNs = 0;
        int64_t durationNs = 0;

        std::string grid;
        std::string block;
        int dynSharedBytes = 0;
        int numRegs = 0;
        std::size_t staticSharedBytes = 0;
        std::size_t localBytes = 0;
        std::size_t constBytes = 0;
        std::string cudaError;

        HostSample host;
        std::vector<DeviceSample> devices;
    };

    struct KernelEndEvent {
        int pid = 0;
        std::string app;
        std::string name;
        std::string tag;
        int64_t tsNs = 0;
        std::string cudaError;
        HostSample host;
        std::vector<DeviceSample> devices;
    };

    struct KernelSampleEvent {
        int pid;
        std::string app;
        std::string name;
        int64_t tsNs;
        HostSample host;
        std::vector<DeviceSample> devices;
    };

    struct ScopeBeginEvent {
        uint64_t scopeId = 0;
        int pid = 0;
        std::string app;
        std::string name;
        std::string tag;
        int64_t tsNs = 0;

        HostSample host;
        std::vector<DeviceSample> devices;
    };

    struct ScopeEndEvent {
        uint64_t scopeId = 0;
        int pid = 0;
        std::string app;
        std::string name;
        std::string tag;
        int64_t tsNs = 0;

        HostSample host;
        std::vector<DeviceSample> devices;
    };

    struct ScopeSampleEvent {
        uint64_t scopeId = 0;
        int pid = 0;
        std::string app;
        std::string name;
        std::string tag;
        int64_t tsNs = 0;

        HostSample host;
        std::vector<DeviceSample> devices;
    };

    struct SystemStartEvent {
        int pid{};
        std::string app;
        std::string name;
        int64_t tsNs{};

        HostSample host;
        std::vector<DeviceSample> devices;
    };

    struct SystemSampleEvent {
        int pid = 0;
        std::string app;
        std::string name;
        int64_t tsNs = 0;

        HostSample host;
        std::vector<DeviceSample> devices;
    };

    struct SystemStopEvent {
        int pid{};
        std::string app;
        std::string name;
        int64_t tsNs{};

        HostSample host;
        std::vector<DeviceSample> devices;
    };
}
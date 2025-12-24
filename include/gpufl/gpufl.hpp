#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <atomic>
#include "gpufl/core/monitor.hpp"

namespace gpufl {
    enum MetricFlags {
        METRIC_NONE = 0,
        METRIC_SYSTEM = 1 << 0, // PCIe, Power, Temps
        METRIC_KERNEL = 1 << 1,
        METRIC_ALL = 0xFFFFFFFF
    };
    enum class BackendKind { Auto, Nvidia, Amd, None };
    static std::atomic<int> g_systemSampleRateMs{0};

    struct InitOptions {
        std::string appName = "gpufl";
        std::string logPath = "";     // if empty, will default to "<app>.log"
        int systemSampleRateMs = 0;
        BackendKind backend = BackendKind::Auto;
        bool enableKernelDetails = false;
        bool enableDebugOutput = false;
    };

    struct BackendProbeResult {
        bool available;
        std::string reason;
    };

    BackendProbeResult probeNvml();
    BackendProbeResult probeRocm();

    void systemStart(std::string name="system");
    void systemStop(std::string name="system");

    // Start global runtime. Returns true on success.
    bool init(const InitOptions& opts);

    // Stop runtime, flush and close logs.
    void shutdown();


    class ScopedMonitor {
    public:
        explicit ScopedMonitor(std::string name, std::string tag = "");
        ~ScopedMonitor();

        ScopedMonitor(const ScopedMonitor&) = delete;
        ScopedMonitor& operator=(const ScopedMonitor&) = delete;

    private:
        std::string name_;
        std::string tag_;
        int pid_{0};
        int64_t startTs_{0};
        uint64_t scopeId_;
    };

    inline void monitor(const std::string& name, const std::function<void()> &fn) {
        ScopedRange r(name.c_str());
        fn();
    }
} // namespace gpufl

#define GFL_SCOPE(name) \
    if(gpufl::ScopedRange _gpufl_scope{name}; true)

#define GFL_SCOPE_TAGGED(name, tag) \
    if (gpufl::ScopedRange _gpufl_scope{name}; true)

#define GFL_SYSTEM_START(name) ::gpufl::systemStart(name)
#define GFL_SYSTEM_STOP(name)  ::gpufl::systemStop(name)
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <atomic>
#include "gpufl/core/monitor.hpp"

namespace gpufl {
    enum class BackendKind { Auto, Nvidia, Amd, None };

    struct InitOptions {
        std::string appName = "gpufl";
        std::string logPath = "";     // if empty, will default to "<app>.log"
        int systemSampleRateMs = 0; // currently less than 50-100 would not be effective.
        int kernelSampleRateMs = 0;
        BackendKind backend = BackendKind::Auto;
        bool samplingAutoStart = false;
        bool enableKernelDetails = false;
        bool enableDebugOutput = false;
        bool enableProfiling = true;
        bool enableStackTrace = true;
    };

    struct BackendProbeResult {
        bool available;
        std::string reason;
    };

    extern std::atomic<int> g_systemSampleRateMs;
    extern InitOptions g_opts;

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
        ScopedMonitor r(name);
        fn();
    }
    inline void monitor(const std::string& name, const std::string& tag, const std::function<void()> &fn) {
        ScopedMonitor r(name, tag);
        fn();
    }
} // namespace gpufl

#define GFL_SCOPE(name) \
    if(gpufl::ScopedMonitor _gpufl_scope{name}; true)

#define GFL_SCOPE_TAGGED(name, tag) \
    if (gpufl::ScopedMonitor+ _gpufl_scope{name, tag}; true)

#define GFL_SYSTEM_START(name) ::gpufl::systemStart(name)
#define GFL_SYSTEM_STOP(name)  ::gpufl::systemStop(name)
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#if GPUFL_HAS_CUDA
#include "gpufl/cuda/cuda.hpp"
#endif

namespace gpufl {
    enum class BackendKind { Auto, Nvidia, Amd, None };
    struct InitOptions {
        std::string appName = "gpufl";
        std::string logPath = "";     // if empty, will default to "<app>.log"
        int scopeSampleRateMs = 0;
        int systemSampleRateMs = 0;
        BackendKind backend = BackendKind::Auto;
    };

    struct BackendProbeResult {
        bool available;
        std::string reason;
    };

    BackendProbeResult probeNvml();
    BackendProbeResult probeRocm();

    void systemStart(int intervalMs, std::string name="system");
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

    inline void monitor(const std::string& name, const std::function<void()> &fn, const std::string& tag = "") {
        ScopedMonitor monitor(name, tag);
        fn();
    }
} // namespace gpufl

#define GFL_SCOPE(name) \
    if(gpufl::ScopedMonitor _gpufl_scope{name}; true)

#define GFL_SCOPE_TAGGED(name, tag) \
    if (gpufl::ScopedMonitor _gpufl_scope{name, tag}; true)

#define GFL_SYSTEM_START(interval, name) ::gpufl::systemStart(interval, name)
#define GFL_SYSTEM_STOP(name)  ::gpufl::systemStop(name)
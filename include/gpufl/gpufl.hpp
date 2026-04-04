#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>

#include "gpufl/core/monitor.hpp"

namespace gpufl {
enum class BackendKind { Auto, Nvidia, Amd, None };

struct InitOptions {
    std::string app_name = "gpufl";
    std::string log_path = "";  // if empty, will default to "<app>.log"
    int system_sample_rate_ms =
        0;  // currently less than 50-100 would not be effective.
    int kernel_sample_rate_ms = 0;
    BackendKind backend = BackendKind::Auto;
    bool sampling_auto_start = false;
    bool enable_kernel_details = false;
    bool enable_debug_output = false;
    bool enable_stack_trace = true;
    ProfilingEngine profiling_engine = ProfilingEngine::PcSampling;
};

struct BackendProbeResult {
    bool available;
    std::string reason;
};

extern std::atomic<int> g_systemSampleRateMs;
extern InitOptions g_opts;

BackendProbeResult probeNvml();
BackendProbeResult probeRocm();

void systemStart(std::string name = "system");
void systemStop(std::string name = "system");

// Start global runtime. Returns true on success.
bool init(const InitOptions& opts);

// Stop runtime, flush and close logs.
void shutdown();

// Generate a text report from the log files written during this session.
// Call after shutdown().
// - No argument: prints the report to console (stdout).
// - With output_path: saves the report to a file.
void generateReport(const std::string& output_path = "");

class ScopedMonitor {
   public:
    explicit ScopedMonitor(std::string name, std::string tag, bool deep_profiling);
    explicit ScopedMonitor(std::string name, std::string tag);
    explicit ScopedMonitor(std::string name, bool deep_profiling);
    explicit ScopedMonitor(std::string name);
    ~ScopedMonitor();

    ScopedMonitor(const ScopedMonitor&) = delete;
    ScopedMonitor& operator=(const ScopedMonitor&) = delete;

   private:
    std::string name_;
    std::string tag_;
    int pid_{0};
    int64_t start_ns_{0};
    uint64_t scope_id_{};
};

inline void monitor(const std::string& name, const std::function<void()>& fn) {
    ScopedMonitor r(name);
    fn();
}
inline void monitor(const std::string& name, const std::string& tag,
                    const std::function<void()>& fn) {
    ScopedMonitor r(name, tag);
    fn();
}
}  // namespace gpufl

#define GFL_SCOPE(name) if (gpufl::ScopedMonitor _gpufl_scope{name}; true)

#define GFL_SCOPE_TAGGED(name, tag, deep_profiling) \
    if (gpufl::ScopedMonitor _gpufl_scope{name, tag}; true)

#define GFL_SYSTEM_START(name) ::gpufl::systemStart(name)
#define GFL_SYSTEM_STOP(name) ::gpufl::systemStop(name)

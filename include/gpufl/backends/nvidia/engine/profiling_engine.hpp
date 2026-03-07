#pragma once

#include <cuda.h>

#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

#include "gpufl/backends/nvidia/cupti_common.hpp"
#include "gpufl/core/events.hpp"
#include "gpufl/core/monitor.hpp"

namespace gpufl {

/**
 * @brief Context passed to an engine at initialization time (after the CUDA
 * context is available). Gives the engine access to shared backend state that
 * it needs to do its job.
 */
struct EngineContext {
    CUcontext cuda_ctx   = nullptr;
    uint32_t  device_id  = 0;
    std::string chip_name;

    // Cubin map — written by ResourceHandler, read by PC Sampling / SASS
    // Metrics for source-correlation look-ups.
    std::mutex*                               cubin_mu     = nullptr;
    std::unordered_map<uint64_t, CubinInfo>*  cubin_by_crc = nullptr;
};

/**
 * @brief Pure-virtual interface for a profiling engine.
 *
 * Exactly one engine is owned by CuptiBackend at any time. The engine
 * is created during initialize() and torn down during shutdown().
 */
class IProfilingEngine {
   public:
    virtual ~IProfilingEngine() = default;

    /**
     * @brief One-time engine initialization after the CUDA context exists.
     * @return false on failure; the engine will not be started.
     */
    virtual bool initialize(const MonitorOptions& opts,
                            const EngineContext& ctx) = 0;

    /** @brief Begin data collection. */
    virtual void start() = 0;

    /** @brief Stop data collection. */
    virtual void stop() = 0;

    /** @brief Release all engine resources. */
    virtual void shutdown() = 0;

    // ---- Scope hooks (PC Sampling / SASS Metrics) ----
    virtual void onScopeStart(const char* /*name*/) {}
    virtual void onScopeStop(const char* /*name*/) {}

    // ---- Perf-scope hooks (Range Profiler / Perfworks) ----
    virtual void onPerfScopeStart(const char* /*name*/) {}
    virtual void onPerfScopeStop(const char* /*name*/) {}

    /** @brief Consume and return the last decoded perf-metric event. */
    virtual std::optional<PerfMetricEvent> takeLastPerfEvent() {
        return std::nullopt;
    }
};

}  // namespace gpufl

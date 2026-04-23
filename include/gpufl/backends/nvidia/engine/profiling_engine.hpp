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
    virtual const char* name() const = 0;

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

    /** @brief Periodically drain buffered profiling data (e.g. PC Sampling).
     *  Called from the collector thread every ~250ms. */
    virtual void drainData() {}

    // ---- Perf-scope hooks (Range Profiler / Perfworks) ----
    virtual void onPerfScopeStart(const char* /*name*/) {}
    virtual void onPerfScopeStop(const char* /*name*/) {}

    /** @brief Consume and return the last decoded perf-metric event. */
    virtual std::optional<PerfMetricEvent> takeLastPerfEvent() {
        return std::nullopt;
    }

    /**
     * @brief True if this engine attempted to start but was denied by
     * CUPTI with CUPTI_ERROR_INSUFFICIENT_PRIVILEGES (or the virtualized
     * equivalent).
     *
     * On Windows this typically means the user is not an administrator
     * AND "Allow access to the GPU performance counters to all users" is
     * not enabled in the NVIDIA Control Panel. On Linux this means the
     * user is not in the `nvidia` group or the `NVreg_RestrictProfilingToAdminUsers`
     * kernel-module parameter is set.
     *
     * Callers (notably `gpufl::init`) check this after start() to emit
     * a user-facing warning and degrade gracefully rather than crashing
     * on the first kernel launch when CUPTI is half-initialized.
     *
     * Default: false (engine doesn't require special privileges, e.g.
     * the None engine).
     */
    virtual bool hasInsufficientPrivileges() const { return false; }

    /**
     * @brief True if this engine started successfully and is producing
     * data. False if start() was skipped, failed, or the engine is None.
     */
    virtual bool isOperational() const { return true; }
};

}  // namespace gpufl

#pragma once

#include <cuda_runtime.h>

#include <string>

#include "gpufl/core/trace_type.hpp"

namespace gpufl {

/**
 * @brief Selects which profiling engine is active for this session.
 *
 * Exactly one engine may be active at a time (CUPTI mutual-exclusivity).
 */
enum class ProfilingEngine {
    None,           // Monitoring only — no profiling overhead
    PcSampling,     // PC-level stall-reason sampling (default when profiling)
    SassMetrics,    // SASS instruction-level metrics
    RangeProfiler,  // Perfworks hardware counters (requires GPUFL_HAS_PERFWORKS)
};

struct MonitorOptions {
    bool collect_kernel_details = false;
    bool enable_debug_output = false;
    bool enable_stack_trace = false;
    int kernel_sample_rate_ms = 0;
    uint32_t pc_sampling_period = 5000;  // GPU clock cycles between PC samples
    ProfilingEngine profiling_engine = ProfilingEngine::None;
};

/**
 * @brief The central monitoring engine.
 */
class Monitor {
   public:
    /**
     * @brief Initializes the monitoring engine.
     */
    static void Initialize(const MonitorOptions& opts = {});

    /**
     * @brief Shuts down the monitoring engine.
     */
    static void Shutdown();

    /**
     * @brief Starts global collection.
     */
    static void Start();

    /**
     * @brief Stops global collection.
     */
    static void Stop();

    /**
     * @brief Marks the start of a logical section.
     */
    static void PushRange(const char* name);

    /**
     * @brief Marks the end of the current section.
     */
    static void PopRange();

    /**
     * @brief Internal API for backends to record events.
     */
    static void RecordStart(const char* name, cudaStream_t stream,
                            TraceType type, void** outHandle);
    static void RecordStop(void* handle, cudaStream_t stream);

    /**
     * @brief Profiler Scope Control
     */
    static void BeginProfilerScope(const char* name);
    static void EndProfilerScope(const char* name);

    /**
     * @brief Hardware counter (Perfworks) scope control
     */
    static void BeginPerfScope(const char* name);
    static void EndPerfScope(const char* name);

    /**
     * @brief Returns the raw backend pointer (may be null).
     */
    static class IMonitorBackend* GetBackend();

   private:
    Monitor() = delete;
};

}  // namespace gpufl

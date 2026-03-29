#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "gpufl/core/events.hpp"
#include "gpufl/core/stream_handle.hpp"
#include "gpufl/core/trace_type.hpp"

namespace gpufl {

/**
 * @brief Selects which profiling engine is active for this session.
 *
 * Note: RangeProfiler is mutually exclusive with PcSampling (both need
 * hardware perf counters).  PcSampling and SassMetrics can coexist because
 * SassMetrics uses software lazy SASS patching, not hardware counters.
 */
enum class ProfilingEngine {
    None,                // Monitoring only — no profiling overhead
    PcSampling,          // PC-level stall-reason sampling
    SassMetrics,         // SASS instruction-level metrics
    RangeProfiler,       // Perfworks hardware counters (requires GPUFL_HAS_PERFWORKS)
    PcSamplingWithSass,  // PC sampling + SASS metrics in a single run
};

struct MonitorOptions {
    bool collect_kernel_details = false;
    bool enable_debug_output = false;
    bool enable_stack_trace = false;
    int kernel_sample_rate_ms = 0;
    uint32_t pc_sampling_period = 12;  // log2 exponent: 2^N GPU cycles between samples (valid: 5-31; 12 = 4096 cycles)
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
    static void RecordStart(const char* name, StreamHandle stream,
                            TraceType type, void** outHandle);
    static void RecordStop(void* handle, StreamHandle stream);

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

    /**
     * @brief Intern a scope name into the shared dictionary.
     * Returns the uint32 dictionary ID, emitting a dictionary_update if new.
     */
    static uint32_t InternScopeName(const std::string& name);

    /**
     * @brief Enqueue a cubin for SASS disassembly on the next batch flush.
     * Called once per unique cubin (identified by CRC) from ResourceHandler.
     */
    static void EnqueueCubinForDisassembly(uint64_t crc, const uint8_t* data,
                                           size_t size);

    /**
     * @brief Push a pre-built ScopeBatchRow into the scope batch buffer.
     */
    static void PushScopeRow(const ScopeBatchRow& row);

   private:
    Monitor() = delete;
};

}  // namespace gpufl

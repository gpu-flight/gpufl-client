#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "gpufl/core/activity_record.hpp"
#include "gpufl/core/events.hpp"
#include "gpufl/core/ring_buffer.hpp"
#include "gpufl/core/stream_handle.hpp"
#include "gpufl/core/trace_type.hpp"

namespace gpufl {

/// Size of the global monitor ring buffer.
///
/// The ring buffer decouples CUPTI callback threads (which must return
/// fast) from the collector thread.  After SASS metric drain was moved
/// off the ring buffer onto a direct-to-batch path (Monitor::PushProfileSamples,
/// taken on the user thread inside onScopeStop), the only remaining
/// producers are CUPTI callbacks: kernel activity records, memcpy
/// activity records, NVTX markers, and PC sampling drain.  These are
/// well below the 8K ceiling in normal use.  RingBuffer::Push retains
/// a brief spin/yield on overrun as defense in depth.
inline constexpr size_t kMonitorBufferSize = 8192;

/// Global ring buffer shared between profiling engines (producers) and
/// the monitor collector thread (consumer).  Declared here so all
/// translation units use the same template instantiation.
extern RingBuffer<ActivityRecord, kMonitorBufferSize> g_monitorBuffer;

enum class MonitorBackendKind {
    Auto,
    Nvidia,
    Amd,
    None,
};

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
    bool enable_source_collection = true;
    // Gate for CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION. Mirror of
    // InitOptions::enable_external_correlation; copied across in the
    // gpufl::init() → CuptiBackend::initialize() conversion path.
    bool enable_external_correlation = true;
    // Gate for CUPTI_ACTIVITY_KIND_SYNCHRONIZATION. Mirror of
    // InitOptions::enable_synchronization. Backend honors this in
    // CuptiBackend::start() — flag false means we never call
    // cuptiActivityEnable for the kind, so zero overhead.
    bool enable_synchronization = true;
    // Gate for CUPTI_ACTIVITY_KIND_MEMORY2. Mirror of
    // InitOptions::enable_memory_tracking. Default-off in v1.
    bool enable_memory_tracking = false;
    // Gate for CUPTI_ACTIVITY_KIND_GRAPH_TRACE. Mirror of
    // InitOptions::enable_cuda_graphs_tracking. Default-off in v1.
    bool enable_cuda_graphs_tracking = false;
    int kernel_sample_rate_ms = 0;
    uint32_t pc_sampling_period = 12;  // log2 exponent: 2^N GPU cycles between samples (valid: 5-31; 12 = 4096 cycles)
    ProfilingEngine profiling_engine = ProfilingEngine::None;
    MonitorBackendKind backend_kind = MonitorBackendKind::Auto;
};

/**
 * @brief Input shape for Monitor::PushProfileSamples.
 *
 * Mirrors the field set produced by the CollectorLoop's PC_SAMPLE branch
 * when translating an ActivityRecord into a ProfileSampleBatchRow — the
 * dictionary interns (function, metric, source_file) happen inside
 * PushProfileSamples so callers don't need access to the dict manager.
 */
struct ProfileSampleInput {
    int64_t  ts_ns         = 0;
    uint32_t corr_id       = 0;
    uint32_t device_id     = 0;
    std::string function_key;   // "function_name@source_file"
    uint32_t pc_offset     = 0;
    std::string metric_name;    // populated for SASS samples; empty for PC sampling
    uint64_t metric_value  = 0;
    uint32_t stall_reason  = 0;  // populated for PC sampling; 0 for SASS
    uint8_t  sample_kind   = 0;  // 0 = pc_sampling, 1 = sass_metric
    std::string source_file;
    uint32_t source_line   = 0;
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

    /**
     * @brief Bulk-push profile samples (SASS metric rows, etc.) directly
     * into the profile batch buffer.
     *
     * Bypasses the lock-free g_monitorBuffer ring — intended for producers
     * that run on the user's app thread (inside onScopeStop) and emit
     * thousands of samples in a tight loop.  Routing those bursts through
     * the ring buffer overruns it and silently drops later activity records
     * (kernel records arriving from CUPTI activity flush).
     *
     * Single bulk call per drain amortizes the g_scopeBatchMu cost over
     * the entire burst.  The 250 ms periodic flush in CollectorLoop picks
     * up the pushed rows; this function never triggers a flush itself
     * (would deadlock on g_scopeBatchMu re-entry).
     *
     * Safe to call from any thread.
     */
    static void PushProfileSamples(
        const std::vector<ProfileSampleInput>& samples);

   private:
    Monitor() = delete;
};

}  // namespace gpufl

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
 * Compatibility matrix (NVIDIA backend):
 *
 *   PcSampling   + SassMetrics   → Deep (shipping). PC sampling uses
 *                                  hardware counters; SASS metrics uses
 *                                  software lazy cubin patching — disjoint
 *                                  mechanisms, can coexist.
 *
 *   SassMetrics  + RangeProfiler → Possible in principle (no resource
 *                                  conflict — SASS is software-patching,
 *                                  Range uses hardware counters). Not yet
 *                                  implemented; see prior plan notes for
 *                                  the composite-engine sketch.
 *
 *   PcSampling   + RangeProfiler → IMPOSSIBLE on current NVIDIA drivers.
 *                                  Both require exclusive access to SM
 *                                  hardware perf counters; whichever calls
 *                                  cuptiProfilerInitialize() first pre-empts
 *                                  the other, and the loser fails with
 *                                  CUPTI_ERROR_INVALID_OPERATION or
 *                                  CUPTI_ERROR_NOT_INITIALIZED. No
 *                                  client-side fix — would require NVIDIA
 *                                  to expose a counter-multiplexing API.
 *
 *   By extension: Deep + RangeProfiler is also impossible (the
 *   PC-sampling half conflicts; the SASS half would be fine).
 *
 *   PmSampling is a time-series hardware metric sampler. It is introduced
 *   as a standalone engine first because it also consumes PM hardware state;
 *   compatibility with SASS / PC sampling is validated separately.
 */
enum class ProfilingEngine {
    // A monotonic ladder of increasing capture depth + cost. One name
    // per level, no aliases — the names are chosen so the option list
    // reads clearly to someone who isn't a CUPTI expert. The plain
    // intent is in each comment; the precise CUPTI mechanism is named
    // where it's the real, doc-searchable term (PC sampling, SASS).
    Monitor,        // No CUPTI at all — GPU/host health metrics only
                    // (util, mem, temp, power). Lowest overhead and immune
                    // to every CUPTI kernel-path failure mode (per-launch
                    // cost, the CUDA 13.1 symbolName segfault, activity
                    // buffer leakage). "Just watch my GPU." The default.
    Trace,          // + Activity trace: every kernel, memcpy/memset, and
                    // sync — name, duration, stream, grid/block. NOT
                    // kernel-exclusive (hence "Trace", not "KernelTrace").
                    // "What ran and how long."
    PcSampling,     // + PC stall-reason sampling: where in each kernel the
                    // GPU stalls. "Why is this kernel slow." (CUPTI PC Sampling)
    SassMetrics,    // + Per-instruction SASS counters: executed / divergent
                    // instruction counts per source line. (CUPTI SASS Metrics)
    PmSampling,     // + Time-series PM hardware samples: SM / memory /
                    // tensor utilization over time. (CUPTI PM Sampling)
    RangeProfiler,  // + Hardware throughput counters: SM / memory / tensor
                    // utilization, L1/L2 hit rates, DRAM bandwidth.
                    // (CUPTI Range Profiler; requires GPUFL_HAS_PERFWORKS)
    RangeProfilerKernelReplay,  // Kernel-owned counters via AutoRange + KernelReplay.
    Deep,           // PcSampling + SassMetrics in a single run — the
                    // deepest single-session profile. (Was PcSamplingWithSass.)
};

inline const char* ProfilingEngineWireName(const ProfilingEngine engine) {
    switch (engine) {
        case ProfilingEngine::Monitor:       return "nvidia.none";
        case ProfilingEngine::Trace:         return "nvidia.trace";
        case ProfilingEngine::PcSampling:    return "nvidia.pc_sampling";
        case ProfilingEngine::SassMetrics:   return "nvidia.sass_metrics";
        case ProfilingEngine::PmSampling:    return "nvidia.pm_sampling";
        case ProfilingEngine::RangeProfiler: return "nvidia.range_profiler";
        case ProfilingEngine::RangeProfilerKernelReplay:
            return "nvidia.range_profiler_kernel_replay";
        case ProfilingEngine::Deep:          return "nvidia.pc_sampling_with_sass";
    }
    return "nvidia.unknown";
}

inline const char* ProfilingEngineSessionKind(const ProfilingEngine engine) {
    switch (engine) {
        case ProfilingEngine::Monitor:
        case ProfilingEngine::Trace:
            return "monitor";
        case ProfilingEngine::PcSampling:
        case ProfilingEngine::SassMetrics:
        case ProfilingEngine::PmSampling:
        case ProfilingEngine::RangeProfiler:
        case ProfilingEngine::RangeProfilerKernelReplay:
        case ProfilingEngine::Deep:
            return "trace";
    }
    return "monitor";
}

struct MonitorOptions {
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
    // log2 exponent: 2^N GPU cycles between samples (valid: 5-31).
    // Default 16 = 65536 cycles. Was 12 (4096 cycles), but on
    // many-kernel workloads (PyTorch / transformer training) the 4096-
    // cycle default floods drainData() with ~10⁶ samples per session,
    // contributing ~700% overhead on benchmark/run_benchmark.py's
    // MiniGPT case (vs ~1% on the GEMM case, which has narrow SM
    // occupancy and one PC range). 65536 cycles cuts sample volume
    // 16× with no meaningful loss for hotspot analysis — the
    // statistical signal for "which PCs dominate" needs maybe 10⁴
    // samples per hot range, not 10⁶. Users doing fine-grained stall
    // attribution can lower this back via InitOptions.
    uint32_t pc_sampling_period = 10;
    // PM sampling interval in microseconds when using the GPU time trigger.
    // PM Sampling is a hardware time-series engine, so this controls timeline
    // density rather than per-kernel attribution. Default 100us catches short scopes better.
    uint32_t pm_sampling_interval_us = 100;
    // Maximum decoded PM samples retained in the counter data image.
    uint32_t pm_sampling_max_samples = 4096;
    // PM sampling preset. "overview" is the default product view; engines may
    // map it to chip-available metrics and skip unavailable entries.
    std::string pm_sampling_preset = "overview";
    std::vector<std::string> pm_sampling_metrics;
    bool pm_sampling_scope_only = true;
    // Default Monitor: no CUPTI. The user-facing default lives on
    // InitOptions (gpufl.hpp); this internal default matches it so a
    // bare MonitorOptions (e.g. the system-monitor daemon, which only
    // wants telemetry) doesn't accidentally subscribe CUPTI.
    ProfilingEngine profiling_engine = ProfilingEngine::Monitor;
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

struct PmSampleInput {
    uint32_t sample_index = 0;
    int64_t ts_ns = 0;
    uint32_t device_id = 0;
    std::string metric_name;
    double value = 0.0;
};

// Session-level switch (set by the active backend at start): when true the
// collector DROPS orphaned launch metas at shutdown instead of emitting them as
// synthetic kernel rows. Used for engines where kernel-activity is intentionally
// off and the synthetic host-dispatch durations would mislead — SASS safe mode,
// where real kernel activity deadlocks (NVIDIA CUPTI/driver bug). The Execution
// Signature is accumulated separately, so a multi-pass merge is unaffected.
// Default false: normal / PC modes keep best-effort synthesis.
void SetSuppressOrphanSyntheticKernels(bool suppress);

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
     *  Push a raw activity record into the monitor ring buffer.
     * Used by low-level injection callbacks that already have a complete
     * ActivityRecord and should use the normal collector serialization path.
     */
    static void PushActivityRecord(const ActivityRecord& rec);

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

    /**
     * @brief Push decoded PM sampling time-series rows.
     */
    static void PushPmSamples(const std::vector<PmSampleInput>& samples);

    /**
     * @brief Emit PM sampling configuration metadata for readers/UI.
     */
    static void EmitPmSamplingConfig(uint32_t device_id,
                                     uint32_t interval_us,
                                     uint32_t max_samples,
                                     const std::string& preset,
                                     const std::vector<std::string>& metrics);

   private:
    Monitor() = delete;
};

}  // namespace gpufl

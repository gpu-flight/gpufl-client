#pragma once

#include <cuda_runtime.h>
#include <cupti.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "gpufl/backends/nvidia/cupti_common.hpp"
#include "gpufl/backends/nvidia/cupti_utils.hpp"
#include "gpufl/backends/nvidia/engine/profiling_engine.hpp"
#include "gpufl/backends/nvidia/profiling_plan.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/monitor.hpp"
#include "gpufl/core/monitor_backend.hpp"
#include "gpufl/gpufl.hpp"

namespace gpufl {

class ICuptiHandler;

/**
 * @brief CUPTI-based monitoring backend for NVIDIA GPUs.
 *
 * Owns exactly one IProfilingEngine at a time, selected by
 * MonitorOptions::profiling_engine at initialize() time.
 */
class CuptiBackend : public IMonitorBackend {
   public:
    CuptiBackend() = default;
    ~CuptiBackend() override = default;

    void initialize(const MonitorOptions& opts) override;
    void shutdown() override;

    static CUptiResult (*get_value())(CUpti_ActivityKind);

    void start() override;
    void stop()  override;

    bool IsMonitoringMode() override { return true; }
    bool IsProfilingMode()  override { return engine_ != nullptr; }

    bool IsSassProfilerMode() const { return resolved_plan_.is_sass_profiler; }
    bool SassMetricsOnlyMode() const { return resolved_plan_.sass_metrics_only; }
    bool UseSafeSassActivityDefaults() const {
        return resolved_plan_.safe_sass_activity_defaults;
    }
    bool AllowSassKernelActivity() const {
        return resolved_plan_.allow_sass_kernel_activity;
    }
    // True when an engine combo (GPUFL_ENGINE_COMBO) is active — the backend
    // runs a CompositeEngine over an arbitrary engine list instead of a single
    // engine. Used to measure the compatibility matrix and back the redefined
    // Deep (the maximal validated-compatible set).
    bool comboActive() const { return !combo_.empty(); }
    // Single source of truth for "collect CUPTI kernel ACTIVITY records"
    // (KERNEL + CONCURRENT_KERNEL). For a combo: true iff it includes a
    // kernel-collecting engine (Trace / PmSampling / RangeProfiler). For a
    // single engine: preserves prior behavior (Trace/PM/Range on; PcSampling
    // off; SASS off unless AllowSassKernelActivity). Drives both the handler's
    // requiredActivityKinds() and the capability report.
    bool collectsKernelEvents() const;
    // True when CUPTI kernel ACTIVITY records won't be collected, so every
    // launch must be reported from its callback as a synthetic kernel (PC
    // Sampling, or SASS profiler safe mode without GPUFL_SASS_ALLOW_KERNEL_
    // ACTIVITY). Mirrors KernelLaunchHandler::requiredActivityKinds() returning
    // {}. In these modes the launch callback precomputes the simplified kernel
    // occupancy (the activity record that would otherwise carry it never
    // arrives); see KernelLaunchHandler::handle + drainSyntheticKernels.
    bool WillEmitSyntheticKernels() const {
        return opts_.profiling_engine == ProfilingEngine::PcSampling ||
               !AllowSassKernelActivity();
    }
    bool AllowSassMarkerActivity() const {
        return resolved_plan_.allow_sass_marker_activity;
    }
    bool AllowSassMemTransferActivity() const {
        return resolved_plan_.allow_sass_mem_transfer_activity;
    }
    bool AllowSassMemory2Activity() const {
        return resolved_plan_.allow_sass_memory2_activity;
    }
    bool AllowSassSyncActivity() const {
        return resolved_plan_.allow_sass_sync_activity;
    }
    bool AllowSassGraphActivity() const {
        return resolved_plan_.allow_sass_graph_activity;
    }
    bool AllowSassExternalCorrelation() const {
        return resolved_plan_.allow_sass_external_correlation;
    }
    void FlushProfilingDataBeforeCudaTeardown(const char* reason);
    void NoteKernelLaunchForCleanupFlush() {
        last_cleanup_flush_ns_.store(0, std::memory_order_release);
    }
    void NoteMemoryActivityEmitted() {
        memory_activity_emitted_.fetch_add(1, std::memory_order_relaxed);
    }
    void NoteMemTransferActivityEmitted() {
        mem_transfer_activity_emitted_.fetch_add(1, std::memory_order_relaxed);
    }

    // Whether the active engine consumes cubin binaries. Cubin capture
    // feeds two consumers, and both want the binary for the SAME three
    // instruction-level engines:
    //   1. disassembly (nvdisasm → cubin_disassembly events for the
    //      Source/SASS dashboard view), and
    //   2. the engine's own per-PC cubin lookups (PcSampling and
    //      SassMetrics read cubin_by_crc_ to correlate samples).
    // Trace (activity records only) and RangeProfiler (scope-level HW
    // counters) need neither, so we skip cubin capture/disassembly
    // entirely for them — there's no per-instruction data to attach it
    // to. (ProfilingEngine::Monitor never constructs a CuptiBackend at
    // all, so it doesn't reach this method.)
    bool NeedsCubinCapture() const { return resolved_plan_.needs_cubin_capture; }

    void RegisterHandler(const std::shared_ptr<ICuptiHandler>& handler);

    bool IsActive() const { return active_.load(); }
    const MonitorOptions& GetOptions() const { return opts_; }

    // Drain CUPTI activity buffers from the CUPTI_CBID_RESOURCE_CONTEXT_
    // DESTROY_STARTING callback — i.e. while the context is still alive, just
    // before the driver tears it down. Complementary safety net for contexts
    // destroyed mid-process (an explicit cudaDeviceReset()/cuCtxDestroy, or
    // multi-context apps): our at-exit shutdown() skips cuptiActivityFlushAll
    // (it would deadlock against a dying context, see teardown_flag.hpp), so a
    // flush here captures that context's records first. NOTE: on Windows
    // process exit cudart does NOT proactively destroy the context (it's left
    // to driver DLL-detach), so this callback does NOT fire there — the final
    // kernel records on Windows-exit are instead recovered by Monitor::Shutdown's
    // post-join drain (see monitor.cpp). Invoked synchronously by ResourceHandler.
    void FlushOnContextDestroy();

    CUpti_SubscriberHandle GetSubscriber() const { return subscriber_; }

    void EmitCaptureCapabilities_() const;

    void OnScopeStart(const char* name) override {
        GFL_LOG_DEBUG("OnScopeStart");
        if (engine_) engine_->onScopeStart(name);
    }
    void DrainProfilingData() override {
        if (engine_) engine_->drainData();
    }
    void OnScopeStop(const char* name) override {
        GFL_LOG_DEBUG("OnScopeStop");
        if (engine_) engine_->onScopeStop(name);
        // cuptiActivityFlushAll(1) permanently kills the CUPTI subscriber
        // callback when the SamplingAPI is armed (enableStartStopControl=0,
        // driver 590+).  Skip per-scope flush; activity records accumulate
        // and are flushed at session stop() instead.  cudaDeviceSynchronize
        // ensures GPU work completes before the scope exits.
        if (opts_.profiling_engine == ProfilingEngine::PcSampling ||
            opts_.profiling_engine == ProfilingEngine::Deep) {
            cudaDeviceSynchronize();
        }
    }
    void OnPerfScopeStart(const char* name) override {
        if (engine_) engine_->onPerfScopeStart(name);
    }
    void OnPerfScopeStop(const char* name) override {
        if (engine_) engine_->onPerfScopeStop(name);
    }
    std::optional<PerfMetricEvent> TakeLastPerfEvent() override {
        if (engine_) return engine_->takeLastPerfEvent();
        return std::nullopt;
    }

   private:
    bool ShouldEnableNvtxMarkerActivityBeforeEngine_() const;
    bool ShouldEnableNvtxMarkerActivityForSelectedEngine_() const;
    static void EnableNvtxMarkerActivity_(const char* phase);
    static void LogNvtxMarkerActivityDisabled_(const char* phase);

    friend class ResourceHandler;
    friend class KernelLaunchHandler;
    friend class MemTransferHandler;
    friend class SynchronizationHandler;

    // CUPTI callback functions
    static void CUPTIAPI BufferRequested(uint8_t** buffer, size_t* size,
                                         size_t* maxNumRecords);
    static void CUPTIAPI BufferCompleted(CUcontext context, uint32_t streamId,
                                         uint8_t* buffer, size_t size,
                                         size_t validSize);
    static void CUPTIAPI GflCallback(void* userdata,
                                     CUpti_CallbackDomain domain,
                                     CUpti_CallbackId cbid, const void* cbdata);

    CUpti_SubscriberHandle subscriber_{};
    std::atomic<bool> active_{false};
    bool initialized_{false};
    MonitorOptions opts_;
    ProfilingRequest profiling_request_;
    DeviceFacts device_facts_;
    ResolvedProfilingPlan resolved_plan_;

    // Non-empty when GPUFL_ENGINE_COMBO selected a CompositeEngine: the ordered
    // list of sub-engines (teardown-safe order — PcSampling last). Empty =
    // single-engine mode via opts_.profiling_engine.
    std::vector<ProfilingEngine> combo_;

    // NOTE (Steps 4b-2 + 4c): the launch-meta AND sync-meta join maps + their
    // mutexes are GONE. All corr-keyed joins (kernel, memcpy/memset, and sync)
    // now run lock-free on the single collector thread (g_launchMetaByCorr /
    // g_syncStackByCorr in monitor.cpp); the callbacks push KERNEL_LAUNCH_META /
    // KERNEL_API_EXIT / SYNC_META records to the ring instead of taking a lock.
    // The CUPTI launch + sync callbacks are now zero-lock.

    // Per-session clock anchor mapping CUPTI activity timestamps
    // (cuptiGetTimestamp domain) to wall-clock ns. Captured fresh in start()
    // so re-init sessions re-anchor — the previous function-static anchor in
    // BufferCompleted persisted process-wide and skewed timestamps across
    // init/shutdown cycles. Written once in start() before any activity record
    // can arrive; read on the (serial) BufferCompleted thread.
    int64_t base_cpu_ns_ = 0;
    uint64_t base_cupti_ts_ = 0;
    int64_t toWallNs(uint64_t cuptiTs) const {
        return base_cpu_ns_ + static_cast<int64_t>(cuptiTs - base_cupti_ts_);
    }

    std::string cached_device_name_ = "Unknown Device";
    CUcontext ctx_ = nullptr;

    // Cubin map — written by ResourceHandler, read by engines.
    // seen_cubin_ptrs_ caches raw pointers so MODULE_PROFILED callbacks
    // (which fire on every kernel launch) are skipped after the first hit.
    std::mutex cubin_mu_;
    std::unordered_map<uint64_t, CubinInfo> cubin_by_crc_;
    std::unordered_set<const void*> seen_cubin_ptrs_;

    // Registered once in initialize() via RegisterHandler — BEFORE the CUPTI
    // subscriber + activity callbacks are enabled — and never modified for the
    // rest of the backend's lifetime (a fresh CuptiBackend is created per
    // session; see Monitor::Shutdown's adapter.reset() + Initialize). It is
    // therefore IMMUTABLE while any callback can run, so GflCallback /
    // BufferCompleted (and start()/stop()) read it lock-free — no handler_mu_,
    // and the per-callback vector copy is gone (zero-alloc dispatch). Do NOT
    // call RegisterHandler after initialize() has enabled callbacks.
    std::vector<std::shared_ptr<ICuptiHandler>> handlers_;

    std::atomic<uint64_t> last_kernel_end_ts_{0};
    std::atomic<uint64_t> kernel_activity_seen_{0};
    std::atomic<uint64_t> kernel_activity_emitted_{0};
    std::atomic<uint64_t> kernel_activity_throttled_{0};
    std::atomic<uint64_t> mem_transfer_activity_emitted_{0};
    std::atomic<uint64_t> sync_activity_emitted_{0};
    std::atomic<uint64_t> nvtx_marker_emitted_{0};
    std::atomic<uint64_t> graph_activity_emitted_{0};
    std::atomic<uint64_t> memory_activity_emitted_{0};
    std::atomic<uint64_t> external_correlation_seen_{0};
    std::atomic<uint64_t> source_locator_seen_{0};
    std::atomic<uint64_t> function_record_seen_{0};
    std::atomic<int64_t> last_cleanup_flush_ns_{0};
    // Re-entrancy guard for FlushOnContextDestroy(): cuptiActivityFlushAll
    // drives BufferCompleted on the calling thread, so this prevents a nested
    // context-destroy callback from recursing into another flush.
    std::atomic<bool> context_destroy_flushing_{false};
    mutable std::atomic<bool> capture_capabilities_emitted_{false};
    uint32_t device_id_ = 0;
    std::string chip_name_;

    // Active profiling engine — exactly one (or nullptr for monitoring only)
    std::unique_ptr<IProfilingEngine> engine_;
};

// External-correlation lookup. Defined in cupti_backend.cpp (the
// map lives in that TU's anonymous namespace alongside the BufferCompleted
// dispatch that populates it). Returns true iff a record was found and
// stamped into the output params; the entry is then erased so a stale
// kernel→corr mapping can't outlive the launch.
bool LookupAndPopExternalCorrelation(uint32_t corr_id,
                                     uint8_t* kind_out,
                                     uint64_t* id_out);

}  // namespace gpufl

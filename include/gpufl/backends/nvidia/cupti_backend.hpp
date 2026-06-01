#pragma once

#include <cuda_runtime.h>
#include <cupti.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
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

    bool IsSassProfilerMode() const {
        return opts_.profiling_engine == ProfilingEngine::SassMetrics ||
               opts_.profiling_engine == ProfilingEngine::Deep;
    }
    bool UseSafeSassActivityDefaults() const {
        if (!IsSassProfilerMode()) return false;
        if (EnvFlagEnabled_("GPUFL_SASS_FORCE_SAFE_ACTIVITY")) return true;
        if (EnvFlagEnabled_("GPUFL_SASS_FORCE_FULL_ACTIVITY")) return false;

        const ComputeCapability cc = GetComputeCapability(static_cast<int>(device_id_));
        const uint32_t cuptiVersion = GetCuptiVersion();
        if (cc.valid() && cc.atLeast(12, 0)) return false;
        if (cuptiVersion >= 130200) return false;
        return true;
    }
    bool AllowSassKernelActivity() const {
        return !UseSafeSassActivityDefaults() ||
               EnvFlagEnabled_("GPUFL_SASS_ALLOW_KERNEL_ACTIVITY");
    }
    bool AllowSassMarkerActivity() const {
        return !UseSafeSassActivityDefaults() ||
               EnvFlagEnabled_("GPUFL_SASS_ALLOW_MARKER_ACTIVITY");
    }
    bool AllowSassMemTransferActivity() const {
        if (!UseSafeSassActivityDefaults()) return true;
        return EnvFlagEnabled_("GPUFL_SASS_ALLOW_MEM_TRANSFER_ACTIVITY");
    }
    bool AllowSassMemory2Activity() const {
        if (!UseSafeSassActivityDefaults()) return true;
        const bool memTransferRequested =
            EnvFlagEnabled_("GPUFL_SASS_ALLOW_MEM_TRANSFER_ACTIVITY");
        const bool memory2Requested =
            EnvFlagEnabled_("GPUFL_SASS_ALLOW_MEMORY2_ACTIVITY") ||
            EnvFlagEnabled_("GPUFL_SASS_ALLOW_MEMORY_ACTIVITY");
        // Safe-mode default keeps MEMORY2 because it is tied to the explicit
        // enable_memory_tracking option.  If the user asks to test mem-transfer
        // activity, keep MEMORY2 off unless it is explicitly requested too;
        // enabling both is the confirmed deadlocking combination on sm_86 +
        // CUPTI 13.1.
        return memory2Requested || !memTransferRequested;
    }
    bool AllowSassSyncActivity() const {
        return !UseSafeSassActivityDefaults() ||
               EnvFlagEnabled_("GPUFL_SASS_ALLOW_SYNC_ACTIVITY");
    }
    bool AllowSassGraphActivity() const {
        return !UseSafeSassActivityDefaults() ||
               EnvFlagEnabled_("GPUFL_SASS_ALLOW_GRAPH_ACTIVITY");
    }
    bool AllowSassExternalCorrelation() const {
        return !UseSafeSassActivityDefaults() ||
               EnvFlagEnabled_("GPUFL_SASS_ALLOW_EXTERNAL_CORRELATION");
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
    bool NeedsCubinCapture() const {
        return opts_.profiling_engine == ProfilingEngine::PcSampling ||
               opts_.profiling_engine == ProfilingEngine::SassMetrics ||
               opts_.profiling_engine == ProfilingEngine::Deep;
    }

    void RegisterHandler(const std::shared_ptr<ICuptiHandler>& handler);

    bool IsActive() const { return active_.load(); }
    const MonitorOptions& GetOptions() const { return opts_; }

    // Flush any pending kernel metadata as synthetic activity records.
    // Called from scope stop after cudaDeviceSynchronize() so durations
    // reflect actual GPU execution time.
    void FlushPendingKernels();
    CUpti_SubscriberHandle GetSubscriber() const { return subscriber_; }

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
    friend class ResourceHandler;
    friend class KernelLaunchHandler;
    friend class MemTransferHandler;
    friend class SynchronizationHandler;

    static bool EnvFlagEnabled_(const char* name) {
        const char* v = std::getenv(name);
        return v && v[0] != '\0' && v[0] != '0' && std::strcmp(v, "false") != 0 &&
               std::strcmp(v, "FALSE") != 0 && std::strcmp(v, "off") != 0 &&
               std::strcmp(v, "OFF") != 0;
    }

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

    std::mutex meta_mu_;
    std::unordered_map<uint64_t, LaunchMeta> meta_by_corr_;

    // Sync metadata captured by SynchronizationHandler on API_ENTER and
    // joined to the SYNCHRONIZATION activity record by correlationId in
    // BufferCompleted. Separate mutex from meta_mu_ so sync API_ENTER
    // doesn't contend with kernel-launch metadata writes during bursty
    // workloads (PyTorch eager mode interleaves both heavily).
    struct SyncMeta {
        size_t stack_id = 0;
        int64_t api_enter_ns = 0;
    };
    std::mutex sync_meta_mu_;
    std::unordered_map<uint64_t, SyncMeta> sync_meta_by_corr_;

    std::string cached_device_name_ = "Unknown Device";
    CUcontext ctx_ = nullptr;

    // Cubin map — written by ResourceHandler, read by engines.
    // seen_cubin_ptrs_ caches raw pointers so MODULE_PROFILED callbacks
    // (which fire on every kernel launch) are skipped after the first hit.
    std::mutex cubin_mu_;
    std::unordered_map<uint64_t, CubinInfo> cubin_by_crc_;
    std::unordered_set<const void*> seen_cubin_ptrs_;

    std::vector<std::shared_ptr<ICuptiHandler>> handlers_;
    std::mutex handler_mu_;

    std::atomic<uint64_t> last_kernel_end_ts_{0};
    std::atomic<uint64_t> kernel_activity_seen_{0};
    std::atomic<uint64_t> kernel_activity_emitted_{0};
    std::atomic<uint64_t> kernel_activity_throttled_{0};
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

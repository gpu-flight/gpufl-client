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
    void OnScopeStop(const char* name) override {
        GFL_LOG_DEBUG("OnScopeStop");
        if (engine_) engine_->onScopeStop(name);
        // After engine scope stop (which does cudaDeviceSynchronize),
        // flush any pending kernel records with real durations.
        if (opts_.profiling_engine == ProfilingEngine::PcSampling ||
            opts_.profiling_engine == ProfilingEngine::PcSamplingWithSass) {
            FlushPendingKernels();
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

}  // namespace gpufl

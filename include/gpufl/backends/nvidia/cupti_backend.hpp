#pragma once

#include <cuda_runtime.h>
#include <cupti.h>
#include <cupti_pcsampling.h>
#include <cupti_sass_metrics.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "gpufl/backends/nvidia/cupti_common.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/monitor.hpp"
#include "gpufl/core/monitor_backend.hpp"
#include "gpufl/gpufl.hpp"

namespace gpufl {

class ICuptiHandler;

struct PCSamplingBuffers {
    CUpti_PCSamplingData* data;
    CUpti_PCSamplingPCData* pcRecords;
};

struct PCSamplingDeleter {
    void operator()(PCSamplingBuffers* b) const {
        if (!b) return;

        if (b->pcRecords && b->data) {
            const size_t maxPcs = b->data->collectNumPcs;
            for (size_t i = 0; i < maxPcs; ++i) {
                if (b->pcRecords[i].stallReason) {
                    std::free(b->pcRecords[i].stallReason);
                }
            }
            std::free(b->pcRecords);
        }

        if (b->data) std::free(b->data);
        delete b;
    }
};

/**
 * @brief CUPTI-based monitoring backend for NVIDIA GPUs.
 *
 * This backend uses NVIDIA's CUPTI (CUDA Profiling Tools Interface)
 * to intercept and monitor CUDA kernel launches and events.
 */
class CuptiBackend : public IMonitorBackend {
   public:
    CuptiBackend() = default;
    ~CuptiBackend() override = default;

    void initialize(const MonitorOptions& opts) override;
    void shutdown() override;

    static CUptiResult (*get_value())(CUpti_ActivityKind);

    void start() override;

    bool IsMonitoringMode() override;

    bool IsProfilingMode() override;

    void stop() override;

    void RegisterHandler(std::shared_ptr<ICuptiHandler> handler);

    bool IsActive() const { return active_.load(); }
    const MonitorOptions& GetOptions() const { return opts_; }
    CUpti_SubscriberHandle GetSubscriber() const { return subscriber_; }

    void OnScopeStart(const char* name) override {
        GFL_LOG_DEBUG("OnScopeStart");
        if (IsProfilingMode()) {
            StartPcSampling();
        }
    }

    void OnScopeStop(const char* name) override {
        GFL_LOG_DEBUG("OnScopeStop");
        if (IsProfilingMode()) {
            StopAndCollectPcSampling();
            StopAndCollectSassMetrics();
        }
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

    MonitorMode mode_ =
        MonitorMode::Monitoring;  // enable Monitoring by default.

    std::mutex meta_mu_;
    std::unordered_map<uint64_t, LaunchMeta> meta_by_corr_;

    std::string cached_device_name_ = "Unknown Device";
    CUcontext ctx_ = nullptr;  // context for the profiler.

    struct CubinInfo {
        std::vector<uint8_t> data;
        uint64_t crc;
    };
    std::mutex cubin_mu_;
    std::unordered_map<uint64_t, CubinInfo> cubin_by_crc_;

    // PC Sampling method tracking
    enum class PCSamplingMethod {
        None,         // PC Sampling not available or not initialized
        ActivityAPI,  // Using CUPTI Activity API (older GPUs)
        SamplingAPI   // Using PC Sampling API (newer GPUs, Windows skips
                      // GetData)
    };
    PCSamplingMethod pc_sampling_method_ = PCSamplingMethod::None;

    void EnableProfilingFeatures();
    void EnableSassMetrics();
    void StartPcSampling();
    void StopAndCollectPcSampling() const;
    void StopAndCollectSassMetrics() const;

    struct SassMetricsBuffers {
        CUpti_SassMetrics_Config* config;
        CUpti_SassMetrics_Data* data;
        size_t numMetrics;
    };
    SassMetricsBuffers* sass_metrics_buffers_ = nullptr;

    std::unique_ptr<PCSamplingBuffers, PCSamplingDeleter> pc_sampling_buffers_;

    std::atomic<int> pc_sampling_ref_count_{0};

    mutable std::mutex stall_reason_mu_;
    mutable std::unordered_map<uint32_t, std::string> stall_reason_map_;

    std::vector<std::shared_ptr<ICuptiHandler>> handlers_;
    std::mutex handler_mu_;

    std::atomic<uint64_t> last_kernel_end_ts_{0};
};

}  // namespace gpufl

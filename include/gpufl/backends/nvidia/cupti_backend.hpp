#pragma once

#include <cuda_runtime.h>
#include <cupti.h>
#include <cupti_pcsampling.h>
#include <cupti_sass_metrics.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#if GPUFL_HAS_PERFWORKS
#include <cupti_profiler_host.h>
#include <cupti_range_profiler.h>
#endif

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

    void RegisterHandler(const std::shared_ptr<ICuptiHandler>& handler);

    bool IsActive() const { return active_.load(); }
    const MonitorOptions& GetOptions() const { return opts_; }
    CUpti_SubscriberHandle GetSubscriber() const { return subscriber_; }

    void OnScopeStart(const char* name) override {
        GFL_LOG_DEBUG("OnScopeStart");
        // Keep deep metrics under perf-scope mode only.
        if (opts_.enable_perf_scope) return;
        if (IsProfilingMode()) {
            StartPcSampling();
        }
    }

    void OnScopeStop(const char* name) override {
        GFL_LOG_DEBUG("OnScopeStop");
        // Keep deep metrics under perf-scope mode only.
        if (opts_.enable_perf_scope) return;
        if (IsProfilingMode()) {
            StopAndCollectPcSampling();
            StopAndCollectSassMetrics();
        }
    }

    void OnPerfScopeStart(const char* name) override;
    void OnPerfScopeStop(const char* name) override;
    std::optional<PerfMetricEvent> TakeLastPerfEvent() override;

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
    uint32_t device_id_ = 0;
    std::string chip_name_;

#if GPUFL_HAS_PERFWORKS
    // Perfworks state
    mutable std::mutex perf_mu_;
    std::vector<uint8_t> perf_counter_data_image_;
    std::vector<uint8_t> perf_config_image_;
    std::vector<uint8_t> perf_scratch_buffer_;
    bool perf_session_active_ = false;
    CUpti_RangeProfiler_Object* range_profiler_object_ = nullptr;
    PerfMetricEvent perf_last_event_;
    bool perf_has_event_ = false;
    CUpti_Profiler_Host_Object* perf_host_object_ = nullptr;

    bool InitPerfworksSession();
    void EndPerfPassAndDecode();
#endif
};

}  // namespace gpufl

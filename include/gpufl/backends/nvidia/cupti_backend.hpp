#pragma once

#include "gpufl/backends/nvidia/cupti_common.hpp"
#include "gpufl/core/monitor_backend.hpp"
#include "gpufl/core/monitor.hpp"

#include <cuda_runtime.h>
#include <cupti.h>
#include <cupti_pcsampling.h>
#include <atomic>
#include <mutex>
#include <memory>
#include <unordered_map>
#include <vector>

#include "gpufl/gpufl.hpp"
#include "gpufl/core/debug_logger.hpp"

#include <cupti_sass_metrics.h>

namespace gpufl {

    class ICuptiHandler;

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

        bool isMonitoringMode() override;

        bool isProfilingMode() override;

        void stop() override;

        void registerHandler(std::shared_ptr<ICuptiHandler> handler);

        bool isActive() const { return active_.load(); }
        const MonitorOptions& getOptions() const { return opts_; }
        CUpti_SubscriberHandle getSubscriber() const { return subscriber_; }

        void onScopeStart(const char *name) override {
            GFL_LOG_DEBUG("onScopeStart");
            if (isProfilingMode()) {
                startPCSampling();
            }
        }

        void onScopeStop(const char *name) override {
            GFL_LOG_DEBUG("onScopeStop");
            if (isProfilingMode()) {
                stopAndCollectPCSampling();
                stopAndCollectSassMetrics();
            }
        }

    private:
        friend class ResourceHandler;
        friend class KernelLaunchHandler;
        friend class MemTransferHandler;

        // CUPTI callback functions
        static void CUPTIAPI BufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords);
        static void CUPTIAPI BufferCompleted(CUcontext context, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize);
        static void CUPTIAPI GflCallback(void *userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void *cbdata);

        CUpti_SubscriberHandle subscriber_{};
        std::atomic<bool> active_{false};
        bool initialized_{false};
        MonitorOptions opts_;

        MonitorMode mode_ = MonitorMode::Monitoring; // enable Monitoring by default.

        std::mutex metaMu_;
        std::unordered_map<uint64_t, LaunchMeta> metaByCorr_;

        std::string cachedDeviceName_ = "Unknown Device";
        CUcontext ctx_ = nullptr; // context for the profiler.

        struct CubinInfo {
            std::vector<uint8_t> data;
            uint64_t crc;
        };
        std::mutex cubinMu_;
        std::unordered_map<uint64_t, CubinInfo> cubinByCrc_;

        // PC Sampling method tracking
        enum class PCSamplingMethod {
            None,           // PC Sampling not available or not initialized
            ActivityAPI,    // Using CUPTI Activity API (older GPUs)
            SamplingAPI     // Using PC Sampling API (newer GPUs, Windows skips GetData)
        };
        PCSamplingMethod pcSamplingMethod_ = PCSamplingMethod::None;

        void enableProfilingFeatures();
        void startPCSampling();
        void stopAndCollectPCSampling() const;
        void stopAndCollectSassMetrics() const;

        struct SassMetricsBuffers {
            CUpti_SassMetrics_Config* config;
            CUpti_SassMetrics_Data* data;
            size_t numMetrics;
        };
        SassMetricsBuffers* sassMetricsBuffers_ = nullptr;

        struct PCSamplingBuffers {
            CUpti_PCSamplingData* data;
            CUpti_PCSamplingPCData* pcRecords;
        };
        PCSamplingBuffers* pcSamplingBuffers_ = nullptr;
        std::atomic<int> pcSamplingRefCount_{0};

        mutable std::mutex stallReasonMu_;
        mutable std::unordered_map<uint32_t, std::string> stallReasonMap_;

        std::vector<std::shared_ptr<ICuptiHandler>> handlers_;
        std::mutex handlerMu_;

        std::atomic<uint64_t> lastKernelEndTs_{0};
    };

} // namespace gpufl

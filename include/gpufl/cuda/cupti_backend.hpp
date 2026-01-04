#pragma once

#include "gpufl/core/monitor_backend.hpp"
#include "gpufl/core/monitor.hpp"

#include <cuda_runtime.h>
#include <cupti.h>
#include <cupti_pcsampling.h>
#include <atomic>
#include <mutex>
#include <unordered_map>

#include "gpufl/gpufl.hpp"
#include "gpufl/core/debug_logger.hpp"

namespace gpufl {

    struct ActivityRecord {
        int32_t deviceId;
        char name[128];
        TraceType type;
        cudaStream_t stream;
        cudaEvent_t startEvent;
        cudaEvent_t stopEvent;
        int64_t cpuStartNs;
        int64_t durationNs;

        // Detailed metrics (optional)
        bool hasDetails;
        int gridX, gridY, gridZ;
        int blockX, blockY, blockZ;
        int dynShared;
        int staticShared;
        int localBytes;
        int constBytes;
        int numRegs;
        float occupancy;

        int maxActiveBlocks;
        unsigned int corrId;

        uint32_t sourceLine;
        char sourceFile[256];
        uint32_t samplesCount;
        uint32_t stallReason;
        char deviceName[64];

        char userScope[256]{};
        int scopeDepth{};

        size_t stackId{};
    };

    struct LaunchMeta {
        int64_t apiEnterNs = 0;
        int64_t apiExitNs  = 0;
        bool hasDetails = false;
        int gridX=0, gridY=0, gridZ=0;
        int blockX=0, blockY=0, blockZ=0;
        int dynShared=0, staticShared=0, localBytes=0, constBytes=0, numRegs=0;
        float occupancy=0.0f;
        int maxActiveBlocks=0;
        char name[128]{};
        char userScope[256]{};
        int scopeDepth{};
        size_t stackId{};
    };

    struct SourceLocation {
        std::string fileName;
        uint32_t lineNumber;
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

        bool isMonitoringMode() override;

        bool isProfilingMode() override;

        void stop() override;

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
            }
        }

    private:
        // CUPTI callback functions
        static void CUPTIAPI BufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords);
        static void CUPTIAPI BufferCompleted(CUcontext context, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize);
        static void CUPTIAPI GflCallback(void *userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, CUpti_CallbackData *cbInfo);

        CUpti_SubscriberHandle subscriber_{};
        std::atomic<bool> active_{false};
        bool initialized_{false};
        MonitorOptions opts_;

        MonitorMode mode_ = MonitorMode::Monitoring; // enable Monitoring by default.

        std::mutex metaMu_;
        std::unordered_map<uint64_t, LaunchMeta> metaByCorr_;

        static std::mutex sourceMapMu_;
        static std::unordered_map<uint32_t, SourceLocation> sourceMap_;

        std::string cachedDeviceName_ = "Unknown Device";
        CUcontext ctx_ = nullptr; // context for the profiler.

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
    };

} // namespace gpufl

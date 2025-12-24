#pragma once

#include "gpufl/core/monitor_backend.hpp"
#include "gpufl/core/monitor.hpp"

#include <cuda_runtime.h>
#include <cupti.h>
#include <atomic>
#include <mutex>
#include <unordered_map>

namespace gpufl {

    struct ActivityRecord {
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

        void Initialize(const MonitorOptions& opts) override;
        void Shutdown() override;

        static CUptiResult (*get_value())(CUpti_ActivityKind);

        void Start() override;
        void Stop() override;

        bool IsActive() const { return active_.load(); }
        const MonitorOptions& GetOptions() const { return opts_; }
        CUpti_SubscriberHandle GetSubscriber() const { return subscriber_; }

    private:
        // CUPTI callback functions
        static void CUPTIAPI BufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords);
        static void CUPTIAPI BufferCompleted(CUcontext context, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize);
        static void CUPTIAPI GflCallback(void *userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, CUpti_CallbackData *cbInfo);

        CUpti_SubscriberHandle subscriber_{};
        std::atomic<bool> active_{false};
        bool initialized_{false};
        MonitorOptions opts_;

        std::mutex metaMu_;
        std::unordered_map<uint64_t, LaunchMeta> metaByCorr_;
    };

} // namespace gpufl

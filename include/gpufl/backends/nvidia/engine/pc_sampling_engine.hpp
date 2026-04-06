#pragma once

#include <cupti_pcsampling.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "gpufl/backends/nvidia/engine/profiling_engine.hpp"

namespace gpufl {

// ---- PC Sampling buffer management ----------------------------------------

struct PCSamplingBuffers {
    CUpti_PCSamplingData*   data      = nullptr;
    CUpti_PCSamplingPCData* pcRecords = nullptr;
};

struct PCSamplingDeleter {
    void operator()(PCSamplingBuffers* b) const;
};

// ---- Engine ----------------------------------------------------------------

class PcSamplingEngine final : public IProfilingEngine {
   public:
    PcSamplingEngine() = default;
    ~PcSamplingEngine() override = default;
    const char* name() const override { return "PcSamplingEngine"; }

    bool initialize(const MonitorOptions& opts,
                    const EngineContext& ctx) override;
    void start()    override;
    void stop()     override;
    void shutdown() override;

    void onScopeStart(const char* name) override;
    void onScopeStop(const char* name)  override;

    /// True when using the PC Sampling API (newer GPUs) rather than
    /// the legacy Activity API.  Used by the composite engine to decide
    /// whether SASS metrics can safely coexist.
    bool isSamplingAPI() const {
        return pc_sampling_method_ == Method::SamplingAPI;
    }


   private:
    enum class Method {
        None,        // PC Sampling not available / not initialized
        ActivityAPI, // CUPTI_ACTIVITY_KIND_PC_SAMPLING (older GPUs)
        SamplingAPI, // cuptiPCSampling* API (newer GPUs)
    };

    bool EnableSamplingFeatures_();
    void StartPcSampling_();
    void StopAndCollectPcSampling_();

    MonitorOptions opts_;
    EngineContext  ctx_;

    Method pc_sampling_method_ = Method::None;
    std::atomic<int> pc_sampling_ref_count_{0};
    std::atomic<bool> sampling_api_ready_{false};
    std::atomic<bool> sampling_api_started_{false};
    std::atomic<bool> sampling_api_blocked_{false};

    std::unique_ptr<PCSamplingBuffers, PCSamplingDeleter> pc_sampling_buffers_;

    mutable std::mutex                       stall_reason_mu_;
    mutable std::unordered_map<uint32_t, std::string> stall_reason_map_;
};

}  // namespace gpufl

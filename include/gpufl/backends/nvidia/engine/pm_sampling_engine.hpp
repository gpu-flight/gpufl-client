#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#include "gpufl/backends/nvidia/engine/profiling_engine.hpp"

#if GPUFL_HAS_PERFWORKS
#include <cupti_pmsampling.h>
#include <cupti_profiler_host.h>
#endif

namespace gpufl {

class PmSamplingEngine final : public IProfilingEngine {
   public:
    PmSamplingEngine() = default;
    ~PmSamplingEngine() override { shutdown(); }

    const char* name() const override { return "PmSamplingEngine"; }

    bool initialize(const MonitorOptions& opts,
                    const EngineContext& ctx) override;
    void start() override;
    void stop() override;
    void shutdown() override;

    void onScopeStart(const char* name) override;
    void onScopeStop(const char* name) override;

    bool hasInsufficientPrivileges() const override {
        return insufficient_privileges_.load(std::memory_order_relaxed);
    }
    bool isOperational() const override {
        return operational_.load(std::memory_order_relaxed) ||
               attempted_.load(std::memory_order_relaxed);
    }
    bool producedData() const override {
        return produced_data_.load(std::memory_order_relaxed);
    }

   private:
    std::vector<std::string> ResolveMetrics_() const;
    void EmitConfig_() const;

#if GPUFL_HAS_PERFWORKS
    bool InitializePmSampling_();
    bool BuildConfigImage_();
    bool CreateCounterDataImage_();
    void DecodeAndEmit_();
    void DisablePmSampling_();
    void StartPmSampling_();
    void StopPmSampling_();

    mutable std::mutex pm_mu_;
    CUpti_PmSampling_Object* pm_object_ = nullptr;
    CUpti_Profiler_Host_Object* host_object_ = nullptr;
    std::vector<const char*> metric_name_ptrs_;
    std::vector<uint8_t> counter_availability_image_;
    std::vector<uint8_t> config_image_;
    std::vector<uint8_t> counter_data_image_;
    bool profiler_initialized_ = false;
    bool profiler_init_owned_ = false;
    bool pm_initialized_ = false;
    bool config_emitted_ = false;
#endif

    MonitorOptions opts_;
    EngineContext ctx_;
    std::vector<std::string> metrics_;
    bool running_ = false;
    std::atomic<bool> operational_{false};
    std::atomic<bool> attempted_{false};
    std::atomic<bool> produced_data_{false};
    std::atomic<bool> insufficient_privileges_{false};
};

}  // namespace gpufl

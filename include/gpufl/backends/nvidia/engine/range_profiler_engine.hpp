#pragma once

#include <atomic>
#include <mutex>
#include <optional>
#include <vector>

#include "gpufl/backends/nvidia/engine/profiling_engine.hpp"

#if GPUFL_HAS_PERFWORKS
#include <cupti_profiler_host.h>
#include <cupti_range_profiler.h>
#endif

namespace gpufl {

class RangeProfilerEngine final : public IProfilingEngine {
   public:
    enum class Mode {
        Scope,
        KernelReplay,
    };

    explicit RangeProfilerEngine(Mode mode = Mode::Scope) : mode_(mode) {}
    ~RangeProfilerEngine() override { shutdown(); }
    const char* name() const override {
        return mode_ == Mode::KernelReplay
            ? "RangeProfilerKernelReplayEngine"
            : "RangeProfilerEngine";
    }

    bool initialize(const MonitorOptions& opts,
                    const EngineContext& ctx) override;
    void start()    override;
    void stop()     override;
    void shutdown() override;

    void onPerfScopeStart(const char* name) override;
    void onPerfScopeStop(const char* name)  override;

    std::optional<PerfMetricEvent> takeLastPerfEvent() override;
    std::vector<KernelPerfMetricEvent> takeKernelPerfEvents() override;
    bool isOperational() const override {
        return operational_.load(std::memory_order_relaxed) ||
               attempted_.load(std::memory_order_relaxed);
    }
    bool producedData() const override {
        return produced_data_.load(std::memory_order_relaxed);
    }
    bool kernelReplayMode() const { return mode_ == Mode::KernelReplay; }

   private:
#if GPUFL_HAS_PERFWORKS
    bool InitPerfworksSession_();
    bool InitPerfworksSession_(bool require_single_pass);
    void EndPerfPassAndDecode_();
    void DecodeKernelReplayEvents_();

    bool                       perf_session_active_  = false;
    mutable std::mutex         perf_mu_;
    std::vector<uint8_t>       perf_counter_data_image_;
    std::vector<uint8_t>       perf_config_image_;
    std::vector<uint8_t>       perf_scratch_buffer_;
    std::vector<const char*>    active_metric_names_;
    CUpti_RangeProfiler_Object* range_profiler_object_ = nullptr;
    CUpti_Profiler_Host_Object* perf_host_object_      = nullptr;
    PerfMetricEvent             perf_last_event_;
    bool                        perf_has_event_       = false;
    std::vector<KernelPerfMetricEvent> kernel_perf_events_;
    bool                        kernel_replay_running_ = false;
    bool                        kernel_replay_decoded_ = false;
#endif

    Mode mode_ = Mode::Scope;
    MonitorOptions opts_;
    EngineContext  ctx_;
    std::atomic<bool> operational_{false};
    std::atomic<bool> attempted_{false};
    std::atomic<bool> produced_data_{false};
};

}  // namespace gpufl

#pragma once

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
    RangeProfilerEngine() = default;
    ~RangeProfilerEngine() override { shutdown(); }

    bool initialize(const MonitorOptions& opts,
                    const EngineContext& ctx) override;
    void start()    override;
    void stop()     override;
    void shutdown() override;

    void onPerfScopeStart(const char* name) override;
    void onPerfScopeStop(const char* name)  override;

    std::optional<PerfMetricEvent> takeLastPerfEvent() override;

   private:
#if GPUFL_HAS_PERFWORKS
    bool InitPerfworksSession_();
    void EndPerfPassAndDecode_();

    bool                       perf_session_active_  = false;
    mutable std::mutex         perf_mu_;
    std::vector<uint8_t>       perf_counter_data_image_;
    std::vector<uint8_t>       perf_config_image_;
    std::vector<uint8_t>       perf_scratch_buffer_;
    CUpti_RangeProfiler_Object* range_profiler_object_ = nullptr;
    CUpti_Profiler_Host_Object* perf_host_object_      = nullptr;
    PerfMetricEvent             perf_last_event_;
    bool                        perf_has_event_       = false;
#endif

    MonitorOptions opts_;
    EngineContext  ctx_;
};

}  // namespace gpufl

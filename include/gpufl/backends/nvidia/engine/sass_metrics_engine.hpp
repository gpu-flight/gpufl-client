#pragma once

#include <cupti_sass_metrics.h>

#include "gpufl/backends/nvidia/engine/profiling_engine.hpp"

namespace gpufl {

class SassMetricsEngine final : public IProfilingEngine {
   public:
    SassMetricsEngine() = default;
    ~SassMetricsEngine() override { shutdown(); }

    bool initialize(const MonitorOptions& opts,
                    const EngineContext& ctx) override;
    void start()    override;
    void stop()     override;
    void shutdown() override;

    // SASS metrics are collected at scope end.
    void onScopeStop(const char* name) override;

   private:
    void EnableSassMetrics_();
    void StopAndCollectSassMetrics_();

    struct SassMetricsBuffers {
        CUpti_SassMetrics_Config* config    = nullptr;
        CUpti_SassMetrics_Data*   data      = nullptr;
        size_t                    numMetrics = 0;
    };

    MonitorOptions opts_;
    EngineContext  ctx_;

    SassMetricsBuffers* sass_metrics_buffers_ = nullptr;
};

}  // namespace gpufl

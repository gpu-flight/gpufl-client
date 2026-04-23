#pragma once

#include <cupti_sass_metrics.h>

#include <string>
#include <unordered_map>

#include "gpufl/backends/nvidia/engine/profiling_engine.hpp"

namespace gpufl {

class SassMetricsEngine final : public IProfilingEngine {
   public:
    SassMetricsEngine() = default;
    ~SassMetricsEngine() override { shutdown(); }
    const char* name() const override { return "SassMetricsEngine"; }

    bool initialize(const MonitorOptions& opts,
                    const EngineContext& ctx) override;
    void start()    override;
    void stop()     override;
    void shutdown() override;

    // SASS metrics are collected at scope end.
    void onScopeStop(const char* name) override;

    bool isEnabled() const { return enabled_; }

    /**
     * True when cuptiProfilerInitialize / cuptiSassMetricsEnable returned
     * CUPTI_ERROR_INSUFFICIENT_PRIVILEGES during start(). Surfaces up to
     * gpufl::init() so the user sees a clear error + recovery steps.
     */
    bool hasInsufficientPrivileges() const override {
        return insufficient_privileges_;
    }

    bool isOperational() const override { return enabled_; }

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
    std::unordered_map<uint64_t, std::string> metric_id_to_name_;
    std::vector<std::string> skipped_metrics_;
    bool enabled_ = false;
    bool config_set_ = false;
    bool insufficient_privileges_ = false;
};

}  // namespace gpufl

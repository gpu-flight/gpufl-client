#pragma once

#include <cupti_sass_metrics.h>

#include <atomic>
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

    // SASS metrics are configured at engine start, armed at scope start, and
    // collected at scope end.
    void onScopeStart(const char* name) override;
    void onScopeStop(const char* name) override;
    void flushBeforeCudaTeardown(const char* reason) override;

    bool isEnabled() const { return profiler_initialized_ && config_set_ && !insufficient_privileges_; }

    /**
     * True when cuptiProfilerInitialize / cuptiSassMetricsEnable returned
     * CUPTI_ERROR_INSUFFICIENT_PRIVILEGES during start(). Surfaces up to
     * gpufl::init() so the user sees a clear error + recovery steps.
     */
    bool hasInsufficientPrivileges() const override {
        return insufficient_privileges_;
    }

    bool isOperational() const override { return isEnabled(); }

    /** True once at least one SASS metric sample was pushed this session. */
    bool producedData() const override {
        return produced_data_.load(std::memory_order_relaxed);
    }

   private:
    void ConfigureSassMetrics_();
    void ArmSassMetrics_();
    void StopAndCollectSassMetrics_();
    /**
     * Undo cuptiProfilerInitialize if start() got that far before
     * SASS metric setup failed (e.g. cuptiSassMetricsGetProperties
     * returning INVALID_PARAMETER on a Blackwell laptop). Leaving the
     * profiler in the "initialized" state after a partial SASS setup
     * permanently disables CUPTI's PC Sampling API for the rest of the
     * session, causing PcSamplingWithSass to hang on the next sample
     * drain. Idempotent — safe to call when never initialized or after
     * already deinited.
     */
    void DeInitProfilerIfNeeded_();

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
    std::atomic<bool> produced_data_{false};
    bool config_set_ = false;
    bool insufficient_privileges_ = false;
    // True after cuptiProfilerInitialize() returned CUPTI_SUCCESS in
    // start(). Drives DeInitProfilerIfNeeded_() so partial-setup
    // failures don't leave CUPTI in a state that hangs PC Sampling
    // (see DeInitProfilerIfNeeded_ doc).
    bool profiler_initialized_ = false;
};

}  // namespace gpufl

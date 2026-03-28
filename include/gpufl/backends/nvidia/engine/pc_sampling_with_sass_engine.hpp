#pragma once

#include <memory>

#include "gpufl/backends/nvidia/engine/pc_sampling_engine.hpp"
#include "gpufl/backends/nvidia/engine/profiling_engine.hpp"
#include "gpufl/backends/nvidia/engine/sass_metrics_engine.hpp"

namespace gpufl {

/**
 * @brief Composite engine that runs PcSampling and SassMetrics simultaneously.
 *
 * PcSampling uses hardware stall-reason counters; SassMetrics uses software
 * lazy SASS patching (cuptiSassMetricsEnable with enableLazyPatching=1) and
 * does not lock hardware perf counters.  The two subsystems are independent
 * and can coexist on the same CUDA context.
 *
 * Scope lifecycle:
 *   start  → pc start, then sass start
 *   onScopeStart → pc only (sass has no scope-start work)
 *   onScopeStop  → pc first (includes cudaDeviceSynchronize), then sass
 *                  (GPU is idle so sass flush is safe)
 *   shutdown → pc shutdown, then sass shutdown (if sass started ok)
 *
 * If SASS fails to initialise (e.g. unsupported GPU, profiler conflict), the
 * engine silently falls back to PC-sampling-only.
 */
class PcSamplingWithSassEngine final : public IProfilingEngine {
   public:
    PcSamplingWithSassEngine() = default;
    ~PcSamplingWithSassEngine() override = default;

    const char* name() const override { return "PcSamplingWithSass"; }

    bool initialize(const MonitorOptions& opts,
                    const EngineContext& ctx) override;
    void start()    override;
    void stop()     override;
    void shutdown() override;

    void onScopeStart(const char* name) override;
    void onScopeStop(const char* name)  override;

   private:
    std::unique_ptr<PcSamplingEngine>  pc_;
    std::unique_ptr<SassMetricsEngine> sass_;
    bool sass_ok_ = false;  // true only if SASS engine started successfully
};

}  // namespace gpufl

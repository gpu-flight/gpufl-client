#pragma once

#include <memory>

#include "gpufl/backends/nvidia/engine/pc_sampling_engine.hpp"
#include "gpufl/backends/nvidia/engine/pm_sampling_engine.hpp"
#include "gpufl/backends/nvidia/engine/profiling_engine.hpp"
#include "gpufl/backends/nvidia/engine/sass_metrics_engine.hpp"

namespace gpufl {

/**
 * @brief Deep-mode engine: the deepest analysis the GPU supports.
 *
 * "Deep" combines two complementary CUPTI capabilities — PC sampling
 * (hardware stall-reason sampling: where/why warps stall) and SASS metrics
 * (per-instruction execution counts + coalescing efficiency). Current NVIDIA
 * drivers make the two MUTUALLY EXCLUSIVE (the Profiler API blocks the PC
 * Sampling API and vice versa), so Deep collects ONE per session, not both.
 *
 * Deep attempts SASS metrics by default and falls back to PC sampling if SASS
 * can't arm (see ShouldAttemptSassInDeep in the .cpp). The lazy-patching
 * deadlock is guarded by the per-architecture SASS exclusion gate, not by
 * forcing eager module loading:
 *   - Default: attempt SASS; fall back to PC sampling if SASS fails to arm.
 *   - Architecture in GPUFL_SASS_EXCLUDE_ARCHS (e.g. RTX 3090 / sm_86):
 *     SassMetricsEngine::start() declines, so Deep runs PC sampling only.
 *   - GPUFL_EAGER_MODULE_LOADING=1: opt-in EAGER loading as an alternative
 *     deadlock workaround (no longer required / forced).
 *   - GPUFL_DEEP_PC_ONLY=1: force PC-sampling-only regardless (escape hatch).
 *
 * This is what distinguishes Deep from ProfilingEngine::PcSampling (always
 * pure PC sampling) and ProfilingEngine::SassMetrics (always SASS — caller
 * accepts the risk). When SASS isn't attempted, `sass_` is never constructed
 * and every `sass_` path is guarded by `sass_ok_` (false), so the engine
 * behaves as a thin wrapper over PcSamplingEngine.
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

    /** Forward the collector thread's periodic, non-blocking drain to the
     *  active PC sampling sub-engine. Without this override Deep mode
     *  inherited the base-class no-op drainData(), so PC sampling's
     *  CONTINUOUS-mode buffer was never emptied mid-run — it filled, the
     *  driver back-pressured sample collection, and a CUPTI helper thread
     *  deadlocked on a driver lock. No-op when SASS won the session. */
    void drainData() override;
    void flushBeforeCudaTeardown(const char* reason) override;

    /** Insufficient if EITHER sub-engine was blocked — the composite
     *  requires both to be fully operational. */
    bool hasInsufficientPrivileges() const override {
        return (pc_   && pc_->hasInsufficientPrivileges())
            || (sass_ && sass_->hasInsufficientPrivileges());
    }

    /** Operational if EITHER path armed: PC sampling running, or SASS
     *  enabled. On Blackwell, start() resets pc_ once SASS wins, so we must
     *  also accept sass_ok_ — otherwise an active SASS session would
     *  wrongly report not-operational. */
    bool isOperational() const override {
        return (pc_ && pc_->isOperational()) || sass_ok_;
    }

    bool sassActive() const { return sass_ok_; }
    bool pcSamplingActive() const { return pc_ && pc_->isOperational(); }

    /** True once the corresponding sub-engine actually emitted rows (not merely
     *  armed) — drives the "enabled but 0 data" capability state. */
    bool sassProducedData() const { return sass_ && sass_->producedData(); }
    bool pcProducedData() const { return pc_ && pc_->producedData(); }
    bool pmSamplingActive() const { return pm_ && pm_->isOperational(); }
    bool pmProducedData() const { return pm_ && pm_->producedData(); }

   private:
    std::unique_ptr<PcSamplingEngine>  pc_;
    std::unique_ptr<SassMetricsEngine> sass_;
    std::unique_ptr<PmSamplingEngine>  pm_;
    bool sass_ok_ = false;        // true only if SASS engine started successfully
    bool sass_gate_open_ = false; // GPU is one where Deep may attempt SASS (Blackwell+)
    bool skip_pc_scope_ = false;  // SamplingAPI + SASS conflict → skip PC scope work
};

}  // namespace gpufl

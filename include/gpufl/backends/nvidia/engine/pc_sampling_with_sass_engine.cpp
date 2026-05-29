#include "gpufl/backends/nvidia/engine/pc_sampling_with_sass_engine.hpp"

#include <cupti.h>

#include "gpufl/core/debug_logger.hpp"

namespace gpufl {

bool PcSamplingWithSassEngine::initialize(const MonitorOptions& opts,
                                          const EngineContext& ctx) {
    pc_   = std::make_unique<PcSamplingEngine>();
    sass_ = std::make_unique<SassMetricsEngine>();
    pc_->initialize(opts, ctx);
    sass_->initialize(opts, ctx);
    // sass_ok_ is set after start() once we know whether cuptiSassMetricsEnable
    // succeeded.  initialize() itself is lightweight (stores opts/ctx).
    return true;
}

void PcSamplingWithSassEngine::start() {
    // SASS metrics (Profiler API) and PC Sampling API are mutually
    // exclusive on current NVIDIA drivers (confirmed on Ampere sm_86
    // and Blackwell sm_120; almost certainly the same on Hopper). The
    // old design armed both, then tried to tear PC sampling down if
    // SASS succeeded — but `cuptiPCSamplingDisable` itself is unsafe
    // while another CUPTI API is initialized (see pc_sampling_engine
    // comment around stop()), and on Ampere this teardown hangs
    // indefinitely at the next flush (observed in
    // example/cuda/manykernel_benchmark Deep-mode run, where the
    // process froze after `[flushDisassembly] parsed 40 functions`).
    //
    // The safer pattern: try SASS first, and only arm PC sampling at
    // all if SASS didn't succeed. No risky disable path because PC
    // sampling was never armed alongside the Profiler API.
    //
    // Trade-off: Deep mode now provides SASS-only data when SASS works,
    // and PC-sampling-only data when SASS doesn't. Users wanting both
    // streams need to run two separate sessions until NVIDIA exposes
    // a way to multiplex the underlying hardware counters.
    sass_->start();
    sass_ok_ = sass_->isEnabled();

    if (sass_ok_) {
        // PcSamplingEngine::initialize() is CUPTI-free (just sets
        // fields), so pc_ has touched no CUPTI state yet. Dropping it
        // here is clean — no disable / unsubscribe needed.
        GFL_LOG_DEBUG(
            "[PcSamplingWithSass] SASS active — skipping PC sampling "
            "(mutually exclusive with Profiler API). Deep mode provides "
            "SASS metrics only for this session.");
        pc_.reset();
    } else {
        pc_->start();
        GFL_LOG_DEBUG(
            "[PcSamplingWithSass] SASS unavailable — running PC sampling only.");
    }
}

void PcSamplingWithSassEngine::stop() {
    if (pc_) pc_->stop();
    // SASS needs a final drain at stop — onScopeStop is the normal
    // per-scope drain trigger, but workloads that don't wrap kernels
    // in scopes (e.g. PyTorch training loops without a surrounding
    // gpufl.Scope) would otherwise lose every sample because nothing
    // pulls CUPTI's pending data into g_profileBatch before shutdown
    // tears the buffers down. SassMetricsEngine::stop() now performs
    // that drain — calling it here so the per-scope and per-session
    // paths both flush.
    if (sass_ok_) sass_->stop();
}

void PcSamplingWithSassEngine::shutdown() {
    if (pc_) pc_->shutdown();
    // Belt-and-suspenders: SassMetricsEngine::shutdown() also drains
    // before disabling CUPTI, so a teardown path that skips stop()
    // (or one where stop()'s drain was a no-op because the engine
    // wasn't enabled yet) still gets a final flush.
    if (sass_ok_) sass_->shutdown();
}

void PcSamplingWithSassEngine::onScopeStart(const char* name) {
    if (pc_ && !skip_pc_scope_) pc_->onScopeStart(name);
}

void PcSamplingWithSassEngine::onScopeStop(const char* name) {
    if (pc_ && !skip_pc_scope_) pc_->onScopeStop(name);
    if (sass_ok_) sass_->onScopeStop(name);
}

}  // namespace gpufl

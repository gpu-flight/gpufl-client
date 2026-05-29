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
    sass_->start();
    sass_ok_ = sass_->isEnabled();
    pc_->start();

    if (sass_ok_ && pc_->isSamplingAPI()) {
        // SASS metrics (Profiler API) and PC Sampling API are mutually
        // exclusive on current drivers (Hopper/Blackwell). Keeping both
        // armed produces duplicate subscriber callbacks per kernel and
        // deadlocks at shutdown / first flush. Prefer SASS metrics
        // (higher-value data) and tear PC sampling down cleanly.
        GFL_LOG_DEBUG(
            "[PcSamplingWithSass] SamplingAPI + SASS incompatible — "
            "disabling PC sampling for this session.");
        pc_->stop();
        pc_->shutdown();
        pc_.reset();
    } else if (!sass_ok_ && pc_->isSamplingAPI()) {
        GFL_LOG_DEBUG(
            "[PcSamplingWithSass] SASS failed and PC sampling on SamplingAPI — "
            "running with PC sampling only.");
    } else if (!sass_ok_) {
        GFL_LOG_DEBUG(
            "[PcSamplingWithSass] SASS failed — running PC sampling only.");
    } else {
        GFL_LOG_DEBUG("[PcSamplingWithSass] Both PC sampling and SASS metrics active.");
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

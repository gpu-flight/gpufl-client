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
    // Start SASS first — cuptiSassMetricsEnable() patches loaded modules.
    // Then start PC sampling — it must see already-patched cubins,
    // otherwise both subsystems racing to patch on first kernel launch
    // can deadlock.  The original deadlock was caused by periodic
    // cuptiActivityFlushAll(0) from the collector thread (now removed),
    // not by SASS + PC Sampling coexistence.
    sass_->start();
    sass_ok_ = sass_->isEnabled();
    pc_->start();
    if (!sass_ok_) {
        GFL_LOG_DEBUG(
            "[PcSamplingWithSass] SASS metrics failed to enable — "
            "falling back to PC sampling only.");
    } else if (pc_->isSamplingAPI()) {
        // SamplingAPI PC sampling is mutually exclusive with the Profiler
        // API (SASS metrics) on many driver versions.  Any CUPTI call
        // after cuptiPCSamplingEnable (getData, Disable, FlushAll) can
        // permanently kill the subscriber callback, silently dropping all
        // kernel launch events for subsequent scopes.  Skip PC sampling
        // scope handling when SASS metrics are active — SASS metrics are
        // the higher-value data source in PcSamplingWithSass mode.
        skip_pc_scope_ = true;
        GFL_LOG_DEBUG(
            "[PcSamplingWithSass] SASS active + SamplingAPI — "
            "disabling per-scope PC sampling (Profiler API conflict).");
    } else {
        GFL_LOG_DEBUG("[PcSamplingWithSass] Both PC sampling and SASS metrics active.");
    }
}

void PcSamplingWithSassEngine::stop() {
    pc_->stop();
    // sass has no stop work
}

void PcSamplingWithSassEngine::shutdown() {
    pc_->shutdown();
    if (sass_ok_) sass_->shutdown();
}

void PcSamplingWithSassEngine::onScopeStart(const char* name) {
    if (!skip_pc_scope_) pc_->onScopeStart(name);
}

void PcSamplingWithSassEngine::onScopeStop(const char* name) {
    if (!skip_pc_scope_) pc_->onScopeStop(name);
    if (sass_ok_) sass_->onScopeStop(name);
}

}  // namespace gpufl

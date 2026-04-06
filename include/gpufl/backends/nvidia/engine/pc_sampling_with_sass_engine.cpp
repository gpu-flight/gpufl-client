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
    // Start PC sampling first to determine the method.
    pc_->start();

    // SASS metrics use cuptiSassMetricsEnable() which lazily patches
    // cubins on first kernel launch.  When the PC Sampling API is active
    // (SamplingAPI/CONTINUOUS mode), the lazy patching races with PC
    // sampling hardware setup for CUPTI internal locks, causing an
    // intermittent deadlock on the first kernel launch.  Skip SASS in
    // this case — PC sampling provides stall-reason data on its own.
    if (pc_->isSamplingAPI()) {
        sass_ok_ = false;
        GFL_LOG_DEBUG(
            "[PcSamplingWithSass] PC Sampling API active — skipping SASS "
            "metrics to avoid deadlock with lazy cubin patching.");
    } else {
        sass_->start();
        sass_ok_ = sass_->isEnabled();
    }
    if (!sass_ok_) {
        GFL_LOG_DEBUG(
            "[PcSamplingWithSass] SASS metrics failed to enable — "
            "falling back to PC sampling only.");
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
    pc_->onScopeStart(name);
    // SassMetricsEngine has no scope-start work — lazy patching is persistent.
}

void PcSamplingWithSassEngine::onScopeStop(const char* name) {
    pc_->onScopeStop(name);
    if (sass_ok_) sass_->onScopeStop(name);
}

}  // namespace gpufl

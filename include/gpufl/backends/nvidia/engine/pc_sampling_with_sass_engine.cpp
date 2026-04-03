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
    pc_->start();
    sass_->start();
    sass_ok_ = sass_->isEnabled();
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
    // PC stop first: performs cudaDeviceSynchronize() so GPU is fully idle.
    pc_->onScopeStop(name);

    // Flush CUPTI activity buffers now, while the ring buffer still has space.
    // pc_->onScopeStop() synchronizes the GPU and stops PC sampling, making
    // kernel activity records available in CUPTI's internal buffers.  Flushing
    // here (non-blocking) pushes those kernel records into g_monitorBuffer
    // before sass_->onScopeStop() pushes its own records, which could
    // otherwise fill the ring buffer and cause kernel records to be dropped.
    cuptiActivityFlushAll(0);

    if (sass_ok_) sass_->onScopeStop(name);
}

}  // namespace gpufl

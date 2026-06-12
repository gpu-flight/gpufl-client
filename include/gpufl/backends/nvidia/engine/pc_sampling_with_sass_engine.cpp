#include "gpufl/backends/nvidia/engine/pc_sampling_with_sass_engine.hpp"

#include "gpufl/core/env_vars.hpp"

#include <cupti.h>

#include <cstdlib>
#include <memory>

#include "gpufl/core/debug_logger.hpp"

namespace gpufl {

namespace {
// ── When does Deep attempt SASS metrics? ────────────────────────────────
// Deep attempts SASS by default and falls back to PC sampling if SASS can't
// arm. The two are mutually exclusive on current NVIDIA drivers (the Profiler
// API that SASS uses blocks the PC Sampling API and vice versa), so Deep
// collects ONE per session -- never both (unless GPUFL_DEEP_TRY_BOTH is set
// to experiment with coexistence; see start()).
//
// SASS's CUPTI instrumentation can DEADLOCK against concurrent kernel launches
// under CUDA's default LAZY module loading (each kernel is finalized +
// SASS-patched on first launch; many such first-launches racing across
// PyTorch's threads invert CUPTI/driver locks -- froze an RTX 3090 run). That
// hazard is handled WITHOUT gating here:
//   * the per-architecture exclusion gate in SassMetricsEngine::start()
//     (GPUFL_SASS_EXCLUDE_ARCHS) skips SASS on confirmed-bad architectures, and
//   * SassMetricsEngine's failure paths return not-enabled, so Deep degrades
//     to PC sampling cleanly.
// EAGER module loading is now an OPT-IN alternative workaround
// (GPUFL_EAGER_MODULE_LOADING=1), no longer required for Deep to try SASS.
//   * Normal Deep run        -> attempt SASS, PC fallback if it can't arm.
//   * Arch in exclusion list -> SassMetricsEngine declines -> Deep is PC-only.
//   * GPUFL_DEEP_PC_ONLY=1   -> always PC-only (manual override).
// PC sampling is always the fallback, so a Deep session is never unsafe.
bool ShouldAttemptSassInDeep() {
    // Manual escape hatch: force PC-sampling-only regardless of hardware.
    if (const char* e = std::getenv(env::kDeepPcOnly);
        e && e[0] != '\0' && e[0] != '0') {
        return false;
    }
    // Otherwise attempt SASS. The per-architecture exclusion gate and the
    // failure-handling paths in SassMetricsEngine::start() decide whether it
    // actually arms; if not, start() leaves it not-enabled and Deep runs PC
    // sampling only.
    return true;
}
}  // namespace

bool PcSamplingWithSassEngine::initialize(const MonitorOptions& opts,
                                          const EngineContext& ctx) {
    pc_ = std::make_unique<PcSamplingEngine>();
    pc_->initialize(opts, ctx);

    pm_ = std::make_unique<PmSamplingEngine>();
    pm_->initialize(opts, ctx);

    // Decide once whether Deep should try SASS this session. Only construct
    // the SASS sub-engine when it will - otherwise Deep touches no
    // Profiler-API state and behaves as a PC sampling engine.
    sass_gate_open_ = ShouldAttemptSassInDeep();
    if (sass_gate_open_) {
        sass_ = std::make_unique<SassMetricsEngine>();
        sass_->initialize(opts, ctx);
        GFL_LOG_DEBUG(
            "[PcSamplingWithSass] eager module loading active - Deep will "
            "attempt SASS metrics (PC sampling is the fallback; the two are "
            "mutually exclusive, so Deep collects one or the other).");
    } else {
        GFL_LOG_DEBUG(
            "[PcSamplingWithSass] SASS not attempted (CUDA_MODULE_LOADING is "
            "not EAGER, or GPUFL_DEEP_PC_ONLY set) - Deep runs PC sampling "
            "only, which avoids the lazy-patching deadlock.");
    }
    // sass_ok_ is finalized in start() once we know whether
    // cuptiSassMetricsEnable actually succeeded.
    return true;
}

void PcSamplingWithSassEngine::start() {
    if (pm_) pm_->start();

    // Deep attempts SASS only where ShouldAttemptSassInDeep() allowed it
    // (eager module loading in effect, unless GPUFL_DEEP_PC_ONLY is set).
    // Otherwise sass_ was never constructed and we go straight to PC
    // sampling - no Profiler API, no lazy patching, no deadlock.
    if (sass_gate_open_ && sass_) {
        sass_->start();
        sass_ok_ = sass_->isEnabled();
    } else {
        sass_ok_ = false;
    }

    if (sass_ok_) {
        // SASS armed. PC sampling and the Profiler API have been mutually
        // exclusive on every driver we've tested - BUT that was only ever
        // observed under lazy module loading, tangled up with the now-fixed
        // deadlock. GPUFL_DEEP_TRY_BOTH=1 tests whether they can COEXIST
        // under eager loading: try arming PC sampling alongside SASS and keep
        // both if it actually arms; otherwise (the expected outcome) fall
        // back to SASS only. If the enable is rejected, nothing armed, so
        // there's nothing to disable (avoids the unsafe disable-PC-while-
        // Profiler-API-active teardown).
        bool tryBoth = false;
        if (const char* e = std::getenv(gpufl::env::kDeepTryBoth))
            tryBoth = (e[0] != '\0' && e[0] != '0');

        if (tryBoth && pc_) {
            pc_->start();
            if (pc_->isOperational()) {
                // Coexistence - keep BOTH sub-engines. Logged at error level
                // (always visible) because it would overturn the long-standing
                // mutual-exclusion assumption and warrants verification.
                GFL_LOG_ERROR(
                    "[PcSamplingWithSass] EXPERIMENTAL: SASS + PC sampling are "
                    "BOTH armed (GPUFL_DEEP_TRY_BOTH). Confirm BOTH data streams "
                    "populate (sass_metric rows AND pc_sampling rows) and the "
                    "run stays stable before trusting this.");
            } else {
                GFL_LOG_DEBUG(
                    "[PcSamplingWithSass] GPUFL_DEEP_TRY_BOTH: PC sampling did "
                    "not arm alongside SASS (still mutually exclusive) - "
                    "keeping SASS only.");
                pc_.reset();  // enable rejected; nothing armed to disable
            }
        } else {
            GFL_LOG_DEBUG(
                "[PcSamplingWithSass] SASS active - skipping PC sampling "
                "(mutually exclusive with Profiler API). Deep = SASS metrics "
                "this session.");
            pc_.reset();
        }
    } else {
        pc_->start();
        // Expected Deep behavior when SASS isn't attempted (eager loading
        // not in effect) or SASS declined to arm - not an error.
        GFL_LOG_DEBUG(
            "[PcSamplingWithSass] Deep mode: PC sampling active (stall-reason "
            "data, drains on the collector thread).");
    }
}

void PcSamplingWithSassEngine::stop() {
    // Tear SASS down BEFORE PC sampling. cuptiPCSamplingDisable is unsafe
    // while another CUPTI API (the Profiler API, which SASS uses) is still
    // initialized - it can hang. When only one sub-engine is active the
    // other call is a no-op, so this order is safe in every case, and it's
    // required for the GPUFL_DEEP_TRY_BOTH path where both are armed.
    //
    // SassMetricsEngine::stop() also performs a final drain - important for
    // workloads that don't wrap kernels in a gpufl.Scope (e.g. a PyTorch
    // training loop), which would otherwise lose all pending SASS data
    // because nothing pulls CUPTI's buffer into g_profileBatch before
    // shutdown tears it down.
    if (pm_) pm_->stop();
    if (sass_ok_) sass_->stop();
    if (pc_) pc_->stop();
}

void PcSamplingWithSassEngine::shutdown() {
    // Same ordering as stop(): SASS (Profiler API) down before PC sampling.
    // SassMetricsEngine::shutdown() also drains before disabling CUPTI, so a
    // teardown path that skipped stop() (or where stop()'s drain was a no-op)
    // still gets a final flush.
    if (pm_) pm_->shutdown();
    if (sass_ok_) sass_->shutdown();
    if (pc_) pc_->shutdown();
}

void PcSamplingWithSassEngine::onScopeStart(const char* name) {
    if (pm_) pm_->onScopeStart(name);
    if (pc_ && !skip_pc_scope_) pc_->onScopeStart(name);
    if (sass_ok_) sass_->onScopeStart(name);
}

void PcSamplingWithSassEngine::onScopeStop(const char* name) {
    if (pc_ && !skip_pc_scope_) pc_->onScopeStop(name);
    if (sass_ok_) sass_->onScopeStop(name);
    if (pm_) pm_->onScopeStop(name);
}

void PcSamplingWithSassEngine::drainData() {
    // CRITICAL: forward the collector thread's periodic (250 ms),
    // non-blocking drain to the PC sampling sub-engine. PC sampling runs
    // in CONTINUOUS mode - CUPTI fills a sampling buffer as kernels execute
    // and we MUST call cuptiPCSamplingGetData regularly to empty it. The
    // base IProfilingEngine::drainData() is a no-op, so before this
    // override Deep mode never drained mid-run: on a long scope (e.g. a
    // whole training epoch wrapped in one gpufl.Scope) the buffer filled,
    // the driver back-pressured sample collection, and a CUPTI helper
    // thread blocked on a driver rwlock - freezing the whole process.
    // (Plain PcSampling mode was unaffected because PcSamplingEngine
    // exposes the real drainData() directly.)
    //
    // pc_ is null when SASS won the session (the two are mutually
    // exclusive on current drivers - see start()); on that path SASS
    // flushes at scope stop, so this is correctly a no-op.
    if (pc_) pc_->drainData();
}

void PcSamplingWithSassEngine::flushBeforeCudaTeardown(const char* reason) {
    if (sass_ok_ && sass_) sass_->flushBeforeCudaTeardown(reason);
}

}  // namespace gpufl

#include "gpufl/backends/nvidia/engine/sass_metrics_engine.hpp"

#include <cupti.h>
#include <cupti_pcsampling.h>
#include <cupti_profiler_target.h>
#include <cupti_sass_metrics.h>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gpufl/backends/nvidia/cupti_utils.hpp"
#include "gpufl/core/activity_record.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/logger/logger.hpp"
#include "gpufl/core/model/lifecycle_model.hpp"
#include "gpufl/core/monitor.hpp"
#include "gpufl/core/ring_buffer.hpp"
#include "gpufl/core/runtime.hpp"

namespace gpufl {

namespace {
std::vector<const char*> kSassMetricNames = {
    "smsp__sass_inst_executed",
    "smsp__sass_thread_inst_executed",
    "smsp__sass_sectors_mem_global",
    "smsp__sass_sectors_mem_global_ideal",        // sm_120+: aggregate ideal
    "smsp__sass_sectors_mem_global_op_ld_ideal",  // fallback: load ideal
                                                  // (sm_86)
    "smsp__sass_sectors_mem_global_op_st_ideal",  // fallback: store ideal
                                                  // (sm_86)
};

/** Metrics that CUPTI accepts on pre-sm_120 GPUs but incur massive per-launch
 *  overhead due to separate kernel replay passes (one per metric class).
 *  Measured ~120x slowdown on RTX 3090 (sm_86) for tiny-kernel workloads.
 *  We proactively skip these on older hardware so users keep functional
 *  profiling (instruction + thread + actual-sectors counts) at reasonable cost.
 *  The frontend surfaces the skip via sass_config.skipped_metrics. */
bool IsExpensiveOnPreSm120(const char* name) {
    if (!name) return false;
    return std::strcmp(name, "smsp__sass_sectors_mem_global_op_ld_ideal") ==
               0 ||
           std::strcmp(name, "smsp__sass_sectors_mem_global_op_st_ideal") == 0;
}

bool IsInsufficientPrivilege(CUptiResult res) {
    if (res == CUPTI_ERROR_INSUFFICIENT_PRIVILEGES) return true;
#ifdef CUPTI_ERROR_VIRTUALIZED_DEVICE_INSUFFICIENT_PRIVILEGES
    if (res == CUPTI_ERROR_VIRTUALIZED_DEVICE_INSUFFICIENT_PRIVILEGES)
        return true;
#endif
    return false;
}

bool IsExpectedTeardownError(CUptiResult res) {
    return res == CUPTI_SUCCESS || res == CUPTI_ERROR_NOT_INITIALIZED ||
           IsInsufficientPrivilege(res);
}

void FreeCuptiCorrelationString(char* s) {
    if (!s) return;
#if defined(_WIN32) && defined(_DEBUG)
    // CUPTI may allocate these with a different CRT heap than the debug app.
    // Avoid invalid heap-pointer assertions in Debug on Windows.
    (void)s;
#else
    std::free(s);
#endif
}

// ── Device-architecture exclusion for SASS ───────────────────────────────
// SASS metrics intermittently hang or fail on specific GPU architectures
// (notably RTX 3090 / Ampere sm_86: cuptiSassMetricsSetConfig OUT_OF_MEMORY,
// plus lazy-patch launch hangs that EAGER module loading does not always
// cure). GPUFL_SASS_EXCLUDE_ARCHS turns SASS off for confirmed-bad
// architectures WITHOUT a rebuild, so we can compare e.g. sm_86 vs sm_120
// empirically and then exclude only what is actually broken.
//
// Syntax: comma-separated compute capabilities. Each entry is one of:
//   "86"  / "8.6"  → exact match (major 8, minor 6)
//   "120" / "12.0" → exact match (major 12, minor 0)
//   "8"   / "8.x"  → whole generation (any sm_8x)
// Default (env unset): use kDefaultSassExcludeArchs (empty = attempt on every
// architecture). Setting the env var — even to "" — overrides the compiled
// default, so an end user can clear a baked-in exclusion at runtime.
constexpr const char* kDefaultSassExcludeArchs = "";

// True if the trimmed token [begin, end) names the given compute capability.
bool CcMatchesToken(const ComputeCapability& cc, const char* begin,
                    const char* end) {
    while (begin < end && std::isspace(static_cast<unsigned char>(*begin)))
        ++begin;
    while (end > begin && std::isspace(static_cast<unsigned char>(end[-1])))
        --end;
    if (begin >= end) return false;

    const std::string tok(begin, end);
    int wantMajor = -1;
    int wantMinor = -1;
    bool wholeGen = false;

    const auto dot = tok.find('.');
    if (dot != std::string::npos) {
        wantMajor = std::atoi(tok.substr(0, dot).c_str());
        const std::string m = tok.substr(dot + 1);
        if (m.empty() || m == "x" || m == "X") {
            wholeGen = true;
        } else {
            wantMinor = std::atoi(m.c_str());
        }
    } else {
        // All-digit form. A single digit means the whole generation;
        // otherwise the last digit is the minor and the rest is the major
        // (so "86" → 8.6, "120" → 12.0).
        for (const char c : tok)
            if (c < '0' || c > '9') return false;
        if (tok.size() == 1) {
            wantMajor = tok[0] - '0';
            wholeGen = true;
        } else {
            wantMinor = tok.back() - '0';
            wantMajor = std::atoi(tok.substr(0, tok.size() - 1).c_str());
        }
    }

    if (wantMajor < 0 || cc.major != wantMajor) return false;
    return wholeGen || cc.minor == wantMinor;
}

// True if SASS is excluded on this GPU's architecture via the configured
// exclusion list. Never excludes on an unknown architecture (don't
// over-exclude when the capability query failed).
bool ArchExcludedForSass(const ComputeCapability& cc) {
    if (!cc.valid()) return false;
    const char* env = std::getenv("GPUFL_SASS_EXCLUDE_ARCHS");
    const char* list = env ? env : kDefaultSassExcludeArchs;
    if (!list || !*list) return false;

    for (const char* p = list; *p;) {
        const char* comma = std::strchr(p, ',');
        const char* end = comma ? comma : (p + std::strlen(p));
        if (CcMatchesToken(cc, p, end)) return true;
        if (!comma) break;
        p = comma + 1;
    }
    return false;
}

bool EnvFlagEnabled(const char* name) {
    const char* v = std::getenv(name);
    return v && v[0] != '\0' && v[0] != '0';
}

bool ShouldUseLazyPatching() {
    // Lazy SASS patching instruments each cubin on its first kernel launch.
    // That is cheap for small standalone CUDA programs, but framework stacks
    // such as PyTorch often issue many first launches from multiple runtime
    // paths while CUDA/CUPTI helper threads are also active.  The observed
    // deadlock has the app thread in cuLaunchKernel -> libcupti and a CUPTI
    // helper blocked on a libcuda rwlock.  Patch modules at load/profile time
    // by default so SASS remains enabled without entering the launch-time
    // patching path.  Keep an env override for experiments and tiny samples.
    return EnvFlagEnabled("GPUFL_SASS_LAZY_PATCHING");
}
}  // namespace

bool SassMetricsEngine::initialize(const MonitorOptions& opts,
                                   const EngineContext& ctx) {
    opts_ = opts;
    ctx_ = ctx;
    enabled_ = false;
    config_set_ = false;
    GFL_LOG_DEBUG("[SassMetricsEngine] initialized");
    return true;
}

void SassMetricsEngine::start() {
    if (!ctx_.cuda_ctx) {
        GFL_LOG_ERROR("[SassMetricsEngine] No CUDA context available");
        return;
    }

    // Resolve the device id + chip name up front. cuptiGetDeviceId is a
    // standalone query (it does NOT require the Profiler API), so it's safe
    // to call before cuptiProfilerInitialize — and it lets the architecture
    // gate below run BEFORE we touch any Profiler/SASS state on an excluded
    // GPU. EnableSassMetrics_ repeats this guarded by chip_name.empty(), so
    // it becomes a no-op there.
    if (ctx_.chip_name.empty()) {
        if (LogCuptiErrorIfFailed(
                this->name(), "cuptiGetDeviceId",
                cuptiGetDeviceId(ctx_.cuda_ctx, &ctx_.device_id))) {
            return;
        }
        ctx_.chip_name = getChipName(ctx_.device_id);
    }

    // ── Device-architecture exclusion gate ───────────────────────────────
    // Turn SASS off for architectures confirmed to hang/fail — e.g. run with
    // GPUFL_SASS_EXCLUDE_ARCHS=86 for RTX 3090 / Ampere sm_86. This runs
    // before cuptiProfilerInitialize, so an excluded GPU never enters the
    // Profiler API / lazy-patching path at all. enabled_ stays false →
    // standalone SassMetrics produces no data (use PcSampling instead) and
    // Deep mode degrades to PC sampling (sass_->isEnabled() == false).
    const ComputeCapability cc =
        GetComputeCapability(static_cast<int>(ctx_.device_id));
    if (ArchExcludedForSass(cc)) {
        GFL_LOG_ERROR(
            "[SassMetricsEngine] SASS metrics DISABLED on sm_", cc.major,
            cc.minor,
            " via GPUFL_SASS_EXCLUDE_ARCHS (this architecture is configured as "
            "unsupported for SASS). Standalone SassMetrics produces no data — "
            "use ProfilingEngine.PcSampling instead; Deep mode degrades to PC "
            "sampling. Monitor / Trace / PcSampling are unaffected.");
        return;
    }

    // Initialize profiler device — required before SASS Metrics APIs.
    // Mark profiler_initialized_ on success so EnableSassMetrics_'s
    // failure paths can undo this via cuptiProfilerDeInitialize.
    // Leaving the profiler initialized after a partial SASS setup
    // permanently disables CUPTI's PC Sampling API, which is what
    // caused the PcSamplingWithSass-mode hang on Blackwell.
    CUpti_Profiler_Initialize_Params p = {
        CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    CUptiResult initRes = cuptiProfilerInitialize(&p);
    if (LogCuptiErrorIfFailed(this->name(), "cuptiProfilerInitialize",
                              initRes)) {
        if (IsInsufficientPrivilege(initRes)) {
            insufficient_privileges_ = true;
            GFL_LOG_ERROR(
                "[SassMetricsEngine] Insufficient privileges for CUPTI "
                "SASS metrics. Enable GPU performance counter access or "
                "run with elevated privileges. Skipping SASS; the session "
                "continues with kernel-trace data only.");
        } else {
            // The Profiler API wants to initialize against a context that
            // hasn't already loaded modules / run kernels. The #1 cause
            // of this failure in framework apps (PyTorch, etc.) is calling
            // gpufl.init() AFTER GPU work has started. Name that explicitly
            // so users get an actionable fix instead of a raw CUPTI code.
            GFL_LOG_ERROR(
                "[SassMetricsEngine] cuptiProfilerInitialize failed (CUptiResult ",
                static_cast<int>(initRes),
                "). SASS / Deep mode cannot start. Two common causes: "
                "(1) gpufl.init() was called AFTER CUDA work already "
                "began — the CUPTI Profiler API must initialize against a "
                "clean context, so call gpufl.init() BEFORE the first CUDA "
                "kernel (right after `import torch`); (2) this GPU + driver "
                "does not support CUPTI SASS metrics. Skipping SASS; "
                "Monitor / Trace / PcSampling are unaffected.");
        }
        return;
    }
    profiler_initialized_ = true;

    EnableSassMetrics_();
}

void SassMetricsEngine::DeInitProfilerIfNeeded_() {
    if (!profiler_initialized_) return;
    CUpti_Profiler_DeInitialize_Params dp = {
        CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
    CUptiResult res = cuptiProfilerDeInitialize(&dp);
    if (!IsExpectedTeardownError(res)) {
        LogCuptiErrorIfFailed(this->name(), "cuptiProfilerDeInitialize", res);
    }
    profiler_initialized_ = false;
}

void SassMetricsEngine::stop() {
    // Do not drain SASS data at ordinary scope/session stop.  On CUDA 13.1
    // + PyTorch + sm_86, calling cuptiSassMetricsFlushData while the process
    // continues launching kernels leaves CUPTI's launch-path mutex wedged; the
    // next cudaLaunchKernel blocks in libcupti.  Keep SASS armed and defer the
    // one required drain until shutdown(), immediately before disable/teardown.
}

void SassMetricsEngine::shutdown() {
    if (enabled_) {
        // Final drain pass — Monitor::Shutdown() may call shutdown()
        // directly without going through stop() in some teardown paths,
        // so we belt-and-suspender here too. StopAndCollectSassMetrics_
        // is idempotent w.r.t. CUPTI state (just reads pending samples).
        StopAndCollectSassMetrics_();
        CUpti_SassMetricsDisable_Params disableParams = {
            CUpti_SassMetricsDisable_Params_STRUCT_SIZE};
        disableParams.ctx = ctx_.cuda_ctx;
        CUptiResult disableRes = cuptiSassMetricsDisable(&disableParams);
        if (!IsExpectedTeardownError(disableRes)) {
            LogCuptiErrorIfFailed(this->name(), "cuptiSassMetricsDisable",
                                  disableRes);
        }
    }

    if (config_set_) {
        CUpti_SassMetricsUnsetConfig_Params unsetParams = {
            CUpti_SassMetricsUnsetConfig_Params_STRUCT_SIZE};
        unsetParams.deviceIndex = ctx_.device_id;
        CUptiResult unsetRes = cuptiSassMetricsUnsetConfig(&unsetParams);
        if (!IsExpectedTeardownError(unsetRes)) {
            LogCuptiErrorIfFailed(this->name(), "cuptiSassMetricsUnsetConfig",
                                  unsetRes);
        }
    }

    if (sass_metrics_buffers_) {
        if (sass_metrics_buffers_->config)
            std::free(sass_metrics_buffers_->config);
        if (sass_metrics_buffers_->data) std::free(sass_metrics_buffers_->data);
        delete sass_metrics_buffers_;
        sass_metrics_buffers_ = nullptr;
    }
    // Symmetric teardown of cuptiProfilerInitialize from start(). Skipped
    // if the partial-failure paths in EnableSassMetrics_ already called
    // it — DeInitProfilerIfNeeded_ is idempotent via the profiler_initialized_ flag.
    DeInitProfilerIfNeeded_();
    enabled_ = false;
    config_set_ = false;
}

void SassMetricsEngine::onScopeStop(const char* /*name*/) {
    // See stop(): mid-run SASS flush can deadlock the following PyTorch kernel
    // launch.  SASS remains enabled; samples are drained once during shutdown.
}

// ---- Private helpers -------------------------------------------------------

void SassMetricsEngine::EnableSassMetrics_() {
    if (ctx_.chip_name.empty()) {
        if (LogCuptiErrorIfFailed(
                this->name(), "cuptiGetDeviceId",
                cuptiGetDeviceId(ctx_.cuda_ctx, &ctx_.device_id))) {
            return;
        }
        ctx_.chip_name = getChipName(ctx_.device_id);
    }

    // (Note: an earlier blanket sm_8x blocklist was removed. SASS works
    // on Ampere/Ada when it's not racing against PC Sampling — and the
    // PcSamplingWithSassEngine start() change that arms SASS first and
    // skips PC sampling when SASS succeeds eliminates that race. The
    // remaining real-failure paths — cuptiProfilerInitialize failure,
    // cuptiSassMetricsSetConfig OOM, the partial-failure bailout for
    // Blackwell below — all call DeInitProfilerIfNeeded_ and return
    // with isEnabled() == false, which the composite engine handles
    // cleanly. No reason to blocklist Ampere preemptively. An opt-in,
    // per-architecture gate now lives at the top of start()
    // (GPUFL_SASS_EXCLUDE_ARCHS) for architectures CONFIRMED broken; it
    // defaults to empty, so nothing is excluded unless explicitly set.)

    metric_id_to_name_.clear();
    if (!sass_metrics_buffers_) {
        sass_metrics_buffers_ = new SassMetricsBuffers();
        sass_metrics_buffers_->numMetrics = kSassMetricNames.size();
        sass_metrics_buffers_->config = static_cast<CUpti_SassMetrics_Config*>(
            std::calloc(sass_metrics_buffers_->numMetrics,
                        sizeof(CUpti_SassMetrics_Config)));
    }

    // Proactively skip fallback sector-ideal metrics on pre-sm_120 GPUs.
    // CUPTI accepts them but each requires a separate kernel replay pass,
    // causing ~120x per-launch slowdown (measured on RTX 3090 sm_86 with a
    // 2000-launch tiny-kernel workload: 120s vs ~1s on RTX 5060 sm_120).
    // Users keep instruction/thread/actual-sector metrics; coalescing
    // efficiency is unavailable and surfaced via sass_config.skipped_metrics
    // (the frontend shows a hardware-limitation banner in the Memory tab).
    ComputeCapability cc =
        GetComputeCapability(static_cast<int>(ctx_.device_id));
    const bool skipExpensive = cc.valid() && !cc.atLeast(12, 0);
    if (skipExpensive) {
        GFL_LOG_DEBUG(
            "[SassMetricsEngine] GPU compute capability sm_", cc.major,
            cc.minor,
            " (< sm_120) — skipping sector-ideal fallback metrics to avoid "
            "kernel-replay overhead. Coalescing efficiency will be "
            "unavailable.");
    }

    size_t validConfigs = 0;
    size_t failedQueries = 0;
    for (size_t i = 0; i < kSassMetricNames.size(); ++i) {
        // Proactive skip for known-expensive metrics on older GPUs.
        if (skipExpensive && IsExpensiveOnPreSm120(kSassMetricNames[i])) {
            skipped_metrics_.push_back(kSassMetricNames[i]);
            continue;
        }
        CUpti_SassMetrics_GetProperties_Params propParams = {
            CUpti_SassMetrics_GetProperties_Params_STRUCT_SIZE};
        propParams.pChipName = ctx_.chip_name.c_str();
        propParams.pMetricName = kSassMetricNames[i];
        if (LogCuptiErrorIfFailed(this->name(), "cuptiSassMetricsGetProperties",
                                  cuptiSassMetricsGetProperties(&propParams))) {
            GFL_LOG_DEBUG(
                "[SassMetricsEngine] Metric not available on this GPU, "
                "skipping: ",
                kSassMetricNames[i]);
            skipped_metrics_.push_back(kSassMetricNames[i]);
            ++failedQueries;
            continue;
        }
        sass_metrics_buffers_->config[validConfigs].metricId =
            propParams.metric.metricId;
        sass_metrics_buffers_->config[validConfigs].outputGranularity =
            CUPTI_SASS_METRICS_OUTPUT_GRANULARITY_GPU;
        metric_id_to_name_[propParams.metric.metricId] = kSassMetricNames[i];
        ++validConfigs;
    }

    if (validConfigs == 0) {
        GFL_LOG_ERROR(
            "[SassMetricsEngine] No valid SASS metrics for this GPU "
            "(all cuptiSassMetricsGetProperties calls failed). "
            "Tearing down CUPTI profiler so PC Sampling can continue; "
            "Deep mode will degrade to PC Sampling only this session.");
        DeInitProfilerIfNeeded_();
        return;
    }

    // Conservative bail on sm_120+ (Blackwell): any metric-query failure
    // here historically indicated a deeper compatibility issue with the
    // SASS API on this hardware path (Blackwell laptop + Windows WDDM),
    // not just an isolated unsupported metric. Proceeding to
    // cuptiSassMetricsSetConfig + cuptiSassMetricsEnable in that
    // partially-failed state hung the next cuLaunchKernel in lazy-
    // patching interception, because CUPTI's profiler state was
    // already corrupted by the partial init. Bail cleanly instead so
    // Deep mode degrades to PC Sampling only — same UX as the all-
    // failed case above, just triggered by one-or-more failures
    // instead of strictly all.
    if (failedQueries > 0 && cc.valid() && cc.atLeast(12, 0)) {
        GFL_LOG_ERROR(
            "[SassMetricsEngine] ", failedQueries,
            " of ", kSassMetricNames.size(),
            " SASS metric queries failed on sm_", cc.major, cc.minor,
            ". On this hardware family a partial-failure state has been "
            "observed to hang subsequent kernel launches via CUPTI's "
            "lazy-patching path. Skipping SASS entirely; Deep mode will "
            "degrade to PC Sampling only this session.");
        DeInitProfilerIfNeeded_();
        return;
    }

    CUpti_SassMetricsSetConfig_Params setConfigParams = {
        CUpti_SassMetricsSetConfig_Params_STRUCT_SIZE};
    setConfigParams.deviceIndex = ctx_.device_id;
    setConfigParams.numOfMetricConfig = validConfigs;
    setConfigParams.pConfigs = sass_metrics_buffers_->config;
    CUptiResult res = cuptiSassMetricsSetConfig(&setConfigParams);
    if (res == CUPTI_SUCCESS || res == CUPTI_ERROR_INVALID_OPERATION) {
        if (res == CUPTI_SUCCESS) config_set_ = true;

        CUpti_SassMetricsEnable_Params enableParams = {
            CUpti_SassMetricsEnable_Params_STRUCT_SIZE};
        enableParams.ctx = ctx_.cuda_ctx;
        // Default to non-lazy patching.  Lazy patching hooks the first
        // cuLaunchKernel for each cubin, which is exactly where PyTorch
        // deadlocks on CUDA 13/CUPTI 13 in the captured gdb dump.  Non-lazy
        // patching keeps SASS metrics enabled but moves CUPTI's work out of
        // the hot launch path.  Set GPUFL_SASS_LAZY_PATCHING=1 to restore the
        // old behavior for experiments.
        const bool lazyPatching = ShouldUseLazyPatching();
        enableParams.enableLazyPatching = lazyPatching ? 1 : 0;
        CUptiResult enableRes = cuptiSassMetricsEnable(&enableParams);
        if (LogCuptiErrorIfFailed(this->name(), "cuptiSassMetricsEnable",
                                  enableRes)) {
            if (config_set_) {
                CUpti_SassMetricsUnsetConfig_Params unsetParams = {
                    CUpti_SassMetricsUnsetConfig_Params_STRUCT_SIZE};
                unsetParams.deviceIndex = ctx_.device_id;
                CUptiResult unsetRes =
                    cuptiSassMetricsUnsetConfig(&unsetParams);
                if (!IsExpectedTeardownError(unsetRes)) {
                    LogCuptiErrorIfFailed(
                        this->name(), "cuptiSassMetricsUnsetConfig", unsetRes);
                }
                config_set_ = false;
            }
            if (IsInsufficientPrivilege(enableRes)) {
                insufficient_privileges_ = true;
                GFL_LOG_ERROR(
                    "[SassMetricsEngine] Insufficient privileges for CUPTI "
                    "SASS metrics. Skipping SASS instrumentation.");
            }
            DeInitProfilerIfNeeded_();
            return;
        }
        enabled_ = true;
    } else {
        LogCuptiErrorIfFailed(this->name(), "cuptiSassMetricsSetConfig", res);
        // OUT_OF_MEMORY here is the signature of a GPU/driver that can't
        // allocate the SASS metric configuration — observed on Ampere
        // (RTX 3090, sm_86). It is a hardware/driver limitation, NOT a
        // late-init problem (cuptiProfilerInitialize already succeeded by
        // this point). SASS metrics simply aren't available on this
        // setup; Deep degrades to PC sampling and the other engines are
        // unaffected. Spell that out so the raw CUPTI code isn't the only
        // signal the user gets.
        if (res == CUPTI_ERROR_OUT_OF_MEMORY) {
            GFL_LOG_ERROR(
                "[SassMetricsEngine] cuptiSassMetricsSetConfig returned "
                "OUT_OF_MEMORY — this GPU/driver can't allocate SASS metric "
                "counters (seen on Ampere / sm_86). SASS metrics are "
                "unavailable on this setup; this is a hardware/driver limit, "
                "not a configuration error. Deep mode runs as PC Sampling "
                "here; use ProfilingEngine.PcSampling directly to skip this "
                "message. Monitor / Trace / PcSampling are unaffected.");
        } else {
            GFL_LOG_ERROR(
                "[SassMetricsEngine] Failed to enable SASS Metrics "
                "(cuptiSassMetricsSetConfig). Deep degrades to PC sampling; "
                "Monitor / Trace / PcSampling are unaffected.");
        }
        DeInitProfilerIfNeeded_();
        return;
    }
    GFL_LOG_DEBUG("[SassMetricsEngine] SASS Metrics Enabled (lazy_patching=",
                  ShouldUseLazyPatching() ? "true" : "false", ")");

    // Emit sass_config event so the backend can distinguish "metric not
    // supported on this GPU" from "metric produced no data for this kernel".
    if (Runtime* rt = runtime(); rt && rt->logger) {
        SassConfigEvent evt;
        evt.session_id = rt->session_id;
        evt.ts_ns = static_cast<int64_t>(detail::GetTimestampNs());
        evt.device_id = ctx_.device_id;
        for (const auto& [id, name] : metric_id_to_name_)
            evt.configured_metrics.push_back(name);
        evt.skipped_metrics = skipped_metrics_;
        rt->logger->write(model::SassConfigModel(evt));
    }
}

void SassMetricsEngine::StopAndCollectSassMetrics_() {
    if (!enabled_ || !ctx_.cuda_ctx) return;

    CUpti_SassMetricsGetDataProperties_Params props = {
        CUpti_SassMetricsGetDataProperties_Params_STRUCT_SIZE};
    props.ctx = ctx_.cuda_ctx;
    if (cuptiSassMetricsGetDataProperties(&props) != CUPTI_SUCCESS ||
        props.numOfPatchedInstructionRecords == 0) {
        return;
    }

    size_t nRecords = props.numOfPatchedInstructionRecords;
    size_t nInstances = props.numOfInstances;
    auto* data = static_cast<CUpti_SassMetrics_Data*>(
        std::calloc(nRecords, sizeof(CUpti_SassMetrics_Data)));
    auto* instances = static_cast<CUpti_SassMetrics_InstanceValue*>(std::calloc(
        nRecords * nInstances, sizeof(CUpti_SassMetrics_InstanceValue)));
    if (!data || !instances) {
        std::free(instances);
        std::free(data);
        GFL_LOG_ERROR("[SassMetricsEngine] Failed to allocate metric buffers");
        return;
    }

    for (size_t i = 0; i < nRecords; ++i) {
        data[i].structSize = sizeof(CUpti_SassMetrics_Data);
        data[i].pInstanceValues = &instances[i * nInstances];
    }

    CUpti_SassMetricsFlushData_Params flushParams = {
        CUpti_SassMetricsFlushData_Params_STRUCT_SIZE};
    flushParams.ctx = ctx_.cuda_ctx;
    flushParams.numOfPatchedInstructionRecords = nRecords;
    flushParams.numOfInstances = nInstances;
    flushParams.pMetricsData = data;

    CUptiResult flushRes = cuptiSassMetricsFlushData(&flushParams);
    if (flushRes == CUPTI_SUCCESS) {
        // Build the entire burst locally, then push in a single bulk call.
        // Bypassing g_monitorBuffer (which the per-sample loop used to use)
        // avoids ring-buffer overrun: a multi-thousand-sample SASS drain
        // saturated the 8K ring faster than the collector could drain it,
        // silently dropping the kernel activity records that arrive last
        // at cuptiActivityFlushAll().  Routing this drain straight into
        // g_profileBatch under g_scopeBatchMu eliminates the contention.
        std::vector<ProfileSampleInput> samples;
        samples.reserve(nRecords * nInstances);
        const int64_t now_ns = detail::GetTimestampNs();
        for (size_t i = 0; i < nRecords; ++i) {
            char srcFile[256]{};
            uint32_t srcLine = 0;
            bool hasSource = false;
            // Grab data pointer under lock, then call CUPTI outside to avoid
            // deadlock if CUPTI triggers a module-load callback.
            const uint8_t* cubinData = nullptr;
            size_t cubinSize = 0;
            if (ctx_.cubin_mu && ctx_.cubin_by_crc) {
                std::lock_guard lk(*ctx_.cubin_mu);
                auto it = ctx_.cubin_by_crc->find(data[i].cubinCrc);
                if (it != ctx_.cubin_by_crc->end()) {
                    cubinData = it->second.data.data();
                    cubinSize = it->second.data.size();
                }
            }
            if (cubinData) {
                CUpti_GetSassToSourceCorrelationParams corrParams = {
                    sizeof(CUpti_GetSassToSourceCorrelationParams)};
                corrParams.cubin = cubinData;
                corrParams.cubinSize = cubinSize;
                corrParams.functionName = data[i].functionName;
                corrParams.pcOffset = data[i].pcOffset;
                CUptiResult res = cuptiGetSassToSourceCorrelation(&corrParams);
                if (res == CUPTI_SUCCESS) {
                    if (corrParams.fileName) {
                        if (corrParams.dirName &&
                            corrParams.dirName[0] != '\0') {
                            std::snprintf(srcFile, sizeof(srcFile), "%s/%s",
                                          corrParams.dirName,
                                          corrParams.fileName);
                        } else {
                            std::snprintf(srcFile, sizeof(srcFile), "%s",
                                          corrParams.fileName);
                        }
                        hasSource = true;
                    }
                    srcLine = corrParams.lineNumber;
                    FreeCuptiCorrelationString(corrParams.fileName);
                    FreeCuptiCorrelationString(corrParams.dirName);
                } else {
                    LogCuptiErrorIfFailed(
                        this->name(), "cuptiGetSassToSourceCorrelation", res);
                }
            }

            const std::string functionName =
                data[i].functionName ? data[i].functionName : "unknown";
            const std::string sourceFile = hasSource ? srcFile : std::string();
            // function_key matches the format CollectorLoop's PC_SAMPLE
            // branch used (function_name + "@" + source_file) — keeps
            // function_id values stable across the refactor.
            const std::string functionKey = functionName + "@" + sourceFile;
            for (size_t j = 0; j < nInstances; ++j) {
                const auto& [metricId, value] = data[i].pInstanceValues[j];
                ProfileSampleInput s;
                s.ts_ns = now_ns;
                s.device_id = ctx_.device_id;
                s.pc_offset = data[i].pcOffset;
                s.sample_kind = 1;  // sass_metric
                s.function_key = functionKey;
                auto itName = metric_id_to_name_.find(metricId);
                s.metric_name = itName != metric_id_to_name_.end()
                                    ? itName->second
                                    : "metric_" + std::to_string(metricId);
                s.metric_value = value;
                s.source_file = sourceFile;
                s.source_line = hasSource ? srcLine : 0;
                samples.push_back(std::move(s));
            }
        }
        Monitor::PushProfileSamples(samples);
    } else {
        LogCuptiErrorIfFailed(this->name(), "cuptiSassMetricsFlushData",
                              flushRes);
    }
    std::free(instances);
    std::free(data);
}

}  // namespace gpufl

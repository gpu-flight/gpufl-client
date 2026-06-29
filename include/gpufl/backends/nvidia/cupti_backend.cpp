#include "gpufl/backends/nvidia/cupti_backend.hpp"
#include "gpufl/backends/nvidia/cuda_cleanup_handler.hpp"

#include <cupti_profiler_target.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <set>
#include <string>
#include <thread>

#include "gpufl/backends/nvidia/cupti_activity_state.hpp"
#include "gpufl/backends/nvidia/cupti_engine_selection.hpp"
#include "gpufl/backends/nvidia/cupti_runtime_support.hpp"
#include "gpufl/backends/nvidia/cupti_utils.hpp"
#include "gpufl/backends/nvidia/engine/pc_sampling_with_sass_engine.hpp"
#include "gpufl/backends/nvidia/kernel_launch_handler.hpp"
#include "gpufl/backends/nvidia/mem_transfer_handler.hpp"
#include "gpufl/backends/nvidia/resource_handler.hpp"
#include "gpufl/backends/nvidia/synchronization_handler.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/logger/logger.hpp"
#include "gpufl/core/monitor.hpp"  // Monitor::RequestSyntheticDrainAndWait
#include "gpufl/core/model/perf_metric_model.hpp"
#include "gpufl/core/runtime.hpp"
#include "gpufl/core/teardown_flag.hpp"

namespace gpufl {

bool CuptiBackend::IsWindowsInjectedPcSampling() const {
    // Single-engine PC sampling under Windows DLL injection. The cubin worker
    // keys captured cubins for disassembly + PC source correlation; doing that
    // with cuptiGetCubinCrc() takes CUPTI's internal global lock, which while
    // PC sampling is armed under Windows injection disengages the GPU PC sampler
    // -> zero samples (proven root cause). So the worker uses zlib crc32 here
    // and disassembles during the run (PC samples join the disassembly by
    // function name, not by CRC, so the CRC choice is immaterial).
    return WindowsInjectedProcess() && combo_.empty() &&
           opts_.profiling_engine == ProfilingEngine::PcSampling;
}

bool CuptiBackend::ShouldEnableNvtxMarkerActivityBeforeEngine_() const {
    if (!collectsKernelEvents()) return false;
    if (!IsSassProfilerMode()) return true;
    return AllowSassMarkerActivity();
}

bool CuptiBackend::ShouldEnableNvtxMarkerActivityForSelectedEngine_() const {
    if (!collectsKernelEvents()) return false;
    if (!IsSassProfilerMode()) return true;
    if (AllowSassMarkerActivity()) return true;

    // Deep is requested as a SASS-capable mode, but its selected engine is
    // known only after PcSamplingWithSassEngine::start(). If SASS did not arm,
    // Deep degraded to PC sampling and NVTX markers are safe/useful again.
    if (opts_.profiling_engine == ProfilingEngine::Deep) {
        const auto* deep = dynamic_cast<const PcSamplingWithSassEngine*>(engine_.get());
        return deep && !deep->sassActive();
    }

    return false;
}

void CuptiBackend::EnableNvtxMarkerActivity_(const char* phase) {
    const CUptiResult res = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER);
    if (res == CUPTI_SUCCESS) {
        GFL_LOG_DEBUG("[CuptiBackend] NVTX MARKER activity enabled (", phase, ")");
    } else {
        LogCuptiIfUnexpected("CuptiBackend", "cuptiActivityEnable(MARKER)", res);
    }
}

void CuptiBackend::LogNvtxMarkerActivityDisabled_(const char* phase) {
    GFL_LOG_DEBUG(
        "[CuptiBackend] NVTX MARKER activity disabled (", phase,
        ") because SASS metrics are selected. Set "
        "GPUFL_SASS_ALLOW_MARKER_ACTIVITY=1 to test SASS + NVTX markers.");
}

void CuptiBackend::initialize(const MonitorOptions& opts) {
    opts_ = opts;
    profiling_request_ = MakeProfilingRequest(opts_);
    resolved_plan_ = NvidiaProfilingPolicy::Resolve(
        profiling_request_, device_facts_, EnvOverrides::FromProcess());

    DebugLogger::setEnabled(opts_.enable_debug_output);

    // GPUFL_ENGINE_COMBO=Trace,PcSampling,... runs an arbitrary engine set in
    // one process (compatibility-matrix testing + the redefined Deep). When
    // present it overrides the single-engine selection below. Trace/Monitor in
    // the list select the activity-record layer (collectsKernelEvents()), not an
    // engine object; the API engines are built in teardown-safe order
    // (PcSampling LAST) so SASS/Range disable before the PC Sampling API.
    combo_ = ParseEngineComboEnv();
    if (!combo_.empty()) {
        ApplyComboPlanOverrides(resolved_plan_, combo_);
    }
    engine_ = CreateProfilingEngine(opts_.profiling_engine, combo_);

    // Any engine that touches PerfWorks (PcSampling via cuptiPCSamplingEnable;
    // SassMetrics / RangeProfiler / Deep via cuptiProfilerInitialize) must use
    // the libnvperf_host.so that MATCHES our libcupti, or NVPW_CUDA_LoadDriver
    // segfaults on a split CUDA install. Preload it now, before any of those
    // CUPTI calls run. `engine_` is null only for Trace, which needs no
    // PerfWorks, so we skip the preload there.
    if (engine_) {
        PreloadMatchingPerfWorks();
        // Initialize the Profiler API here — before any CUDA context exists
        // and outside any CUPTI callback. Both matter on r590+ Windows
        // drivers (verified): from a callback or after the target's context
        // exists it returns CUPTI_ERROR_UNKNOWN, and without it PC sampling
        // enumerates zero stall reasons. NVIDIA's pc_sampling sample uses
        // the same position. Non-fatal: engines degrade and report why.
        CUpti_Profiler_Initialize_Params profInit = {
            CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
        const CUptiResult profRes = cuptiProfilerInitialize(&profInit);
        GFL_LOG_DEBUG("[CuptiBackend] cuptiProfilerInitialize (pre-context): ",
                      profRes == CUPTI_SUCCESS ? "OK" : "FAILED",
                      " (CUptiResult=", static_cast<int>(profRes), ")");
    }

    SetActiveCuptiBackend(this);

    // Internal handler registration
    RegisterHandler(std::make_shared<ResourceHandler>(this));
    RegisterHandler(std::make_shared<CudaCleanupHandler>(this));
    RegisterHandler(std::make_shared<KernelLaunchHandler>(this));
    RegisterHandler(std::make_shared<MemTransferHandler>(this));
    RegisterHandler(std::make_shared<SynchronizationHandler>(this));

    GFL_LOG_DEBUG("Subscribing to CUPTI...");
    CUPTI_CHECK_RETURN(
        cuptiSubscribe(&subscriber_,
                       reinterpret_cast<CUpti_CallbackFunc>(GflCallback), this),
        "[GPUFL Monitor] ERROR: Failed to subscribe to CUPTI\n"
        "[GPUFL Monitor] This may indicate:\n"
        "  - CUPTI library not found or incompatible\n"
        "  - Insufficient permissions\n"
        "  - CUDA driver issues");
    GFL_LOG_DEBUG("CUPTI subscription successful");

    std::set<CUpti_CallbackDomain> domains;
    std::set<std::pair<CUpti_CallbackDomain, CUpti_CallbackId>> callbacks;
    // handlers_ just registered above on this thread; read lock-free.
    for (const auto& h : handlers_) {
        for (auto d : h->requiredDomains()) domains.insert(d);
        for (auto cb : h->requiredCallbacks()) callbacks.insert(cb);
    }
    for (auto d : domains) CUPTI_CHECK(cuptiEnableDomain(1, subscriber_, d));
    for (auto& [domain, cbid] : callbacks)
        CUPTI_CHECK(cuptiEnableCallback(1, subscriber_, domain, cbid));

    CUptiResult resCb =
        cuptiActivityRegisterCallbacks(BufferRequested, BufferCompleted);
    if (resCb != CUPTI_SUCCESS) {
        GFL_LOG_ERROR("FATAL: Failed to register activity callbacks.");
        LogCuptiErrorIfFailed("CUPTI", "cuptiActivityRegisterCallbacks", resCb);
        initialized_ = false;
        return;
    }

    initialized_ = true;
    GFL_LOG_DEBUG("Callbacks registered successfully.");
}

void CuptiBackend::shutdown() {
    if (!initialized_) return;

    if (active_.load(std::memory_order_relaxed)) {
        stop();
    }
    // stop() already waited out an in-flight deferred start, but cover the
    // !active_ path too before tearing the engine down.
    engine_start_pending_.store(false, std::memory_order_release);
    { std::lock_guard lk(deferred_start_mu_); }

    // Delegate engine teardown first
    if (engine_) {
        engine_->stop();
        engine_->shutdown();
    }

    // Belt-and-suspenders final drain. stop() already disabled all
    // activity kinds and flushed, but engine teardown above can emit a
    // last burst (e.g. PcSamplingEngine::shutdown's StopAndCollectPcSampling_
    // collection). Disabling SOURCE_LOCATOR + FUNCTION here matches what
    // PcSamplingEngine enables on the PC-sampling paths (no-op for engines
    // that never enabled them). The final flush
    // guarantees every BufferCompleted callback has returned before
    // we clear the active backend pointer below - without this, late deliveries
    // raced the pointer-clear and surfaced as "No active backend!"
    // noise on benchmarks that init/shutdown gpufl repeatedly in a
    // single process (run_benchmark.py).
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_FUNCTION);
    if (detail::isProcessExitTeardown()) {
        // Windows injection at-exit: skip sync + final flush against the dying
        // context (they deadlock). See CuptiBackend::stop() / teardown_flag.hpp.
        GFL_LOG_DEBUG("CuptiBackend::shutdown: skip sync+flush (process-exit teardown)");
    } else {
        cudaDeviceSynchronize();
        LogCuptiIfUnexpected("shutdown", "cuptiActivityFlushAll(final)",
                             cuptiActivityFlushAll(1));
    }
    EmitCaptureCapabilities_();
    if (engine_) engine_.reset();

    cuptiUnsubscribe(subscriber_);
    SetActiveCuptiBackend(nullptr);
    initialized_ = false;
}

CUptiResult (*CuptiBackend::get_value())(CUpti_ActivityKind) {
    return cuptiActivityEnable;
}

void CuptiBackend::start() {
    if (!initialized_) return;
    // SASS safe mode keeps kernel-activity OFF (it deadlocks), so orphan launch
    // metas would become synthetic kernels with host-dispatch timing only. Keep
    // suppressing that path for SASS. PC sampling intentionally leaves the
    // synthetic drain enabled so callback-derived kernel rows can be emitted
    // when kernel activity is unavailable; drainSyntheticKernels filters
    // non-kernel memcpy/memset API metas before emitting rows.
    SetSuppressOrphanSyntheticKernels(
        opts_.profiling_engine == ProfilingEngine::SassMetrics);
    // Windows-injection PC sampling AND Deep both lose their final flush to the
    // process-exit teardown race (gpufl::shutdown runs during DLL detach, after
    // the OS starts tearing the process down, so the shutdown drain can be cut
    // off mid-write). Both emit callback-derived synthetic kernels, so drain
    // them DURING the run rather than only at shutdown - then a lost teardown
    // costs at most the last sub-second of kernels instead of all of them.
    // (SASS-only suppresses synthetic kernels, so it is intentionally excluded.)
    SetDrainSyntheticKernelsMidRun(
        IsWindowsInjectedPcSampling() ||
        (WindowsInjectedProcess() && combo_.empty() &&
         opts_.profiling_engine == ProfilingEngine::Deep));
    kernel_activity_seen_.store(0, std::memory_order_relaxed);
    kernel_activity_emitted_.store(0, std::memory_order_relaxed);
    kernel_activity_throttled_.store(0, std::memory_order_relaxed);
    memory_activity_emitted_.store(0, std::memory_order_relaxed);
    mem_transfer_activity_emitted_.store(0, std::memory_order_relaxed);
    sync_activity_emitted_.store(0, std::memory_order_relaxed);
    nvtx_marker_emitted_.store(0, std::memory_order_relaxed);
    graph_activity_emitted_.store(0, std::memory_order_relaxed);
    external_correlation_seen_.store(0, std::memory_order_relaxed);
    source_locator_seen_.store(0, std::memory_order_relaxed);
    function_record_seen_.store(0, std::memory_order_relaxed);
    kernel_launch_callback_count_.store(0, std::memory_order_relaxed);
    capture_capabilities_emitted_.store(false, std::memory_order_relaxed);

    // Reset the BufferCompleted companion maps for a clean per-session slate
    // (Step 5). These persist across BufferCompleted calls *within* a session
    // but were never cleared *between* sessions - across an init/shutdown cycle
    // CUPTI reuses sourceLocator / function / marker ids, so a stale entry from
    // a prior session could mis-attribute a PC sample or NVTX range. Safe to do
    // here: start() runs before any activity kind is enabled below, so no
    // BufferCompleted can be in flight yet (the mutexes are belt-and-suspenders).
    // g_extCorrMap is also cleared at stop(); clearing here too makes the slate
    // robust no matter how the previous session ended.
    ResetCuptiActivityCompanionState();

    // Capture the per-session CUPTI->wall clock anchor before enabling any
    // activity kind, so every activity record converts against a consistent,
    // per-session-fresh base. Replaces the old function-static anchor in
    // BufferCompleted (which leaked across init/shutdown cycles).
    base_cpu_ns_ = detail::GetTimestampNs();
    base_cupti_ts_ = 0;
    cuptiGetTimestamp(&base_cupti_ts_);

    // Resolve CUDA context/device before asking handlers for activity kinds.
    // SASS safe-mode policy is device dependent, and the policy log should
    // report the real SM version. Querying requiredActivityKinds() before this
    // point used device_id_=0 and could choose the wrong activity policy.
    const bool haveCudaContext =
        WindowsInjectedProcess() ? TryCurrentCudaContext(&ctx_)
                                 : EnsureCudaContext(&ctx_);
    if (haveCudaContext) {
        cuptiGetDeviceId(ctx_, &device_id_);
        GetSMProps(device_id_);
        chip_name_ = getChipName(device_id_);
        cached_device_name_ = GetCurrentDeviceName();
        const ComputeCapability cc =
            GetComputeCapability(static_cast<int>(device_id_));
        device_facts_.compute_major = cc.major;
        device_facts_.compute_minor = cc.minor;
        device_facts_.cupti_version = GetCuptiVersion();
        resolved_plan_ = NvidiaProfilingPolicy::Resolve(
            profiling_request_, device_facts_, EnvOverrides::FromProcess());
        ApplyComboPlanOverrides(resolved_plan_, combo_);
    } else if (engine_) {
        GFL_LOG_DEBUG(
            "[CuptiBackend] Failed to get CUDA context; "
            "engine will not start.");
    }

    if (WindowsInjectedProcess() && !haveCudaContext && engine_) {
        // Don't drop the engine - injection init runs during cuInit, before
        // the target creates any context, so this is the NORMAL case for an
        // injected `gpufl trace` run (it used to silently disable PC/SASS
        // sampling for every such session). The CONTEXT_CREATED resource
        // callback completes the start once the target's context exists.
        GFL_LOG_DEBUG(
            "[CuptiBackend] No CUDA context is current during Windows "
            "injection init; deferring engine start until the target "
            "creates one (CONTEXT_CREATED).");
        engine_start_pending_.store(true, std::memory_order_release);
    } else if (WindowsInjectedProcess() && !haveCudaContext) {
        GFL_LOG_DEBUG(
            "[CuptiBackend] No CUDA context is current during Windows "
            "injection init; starting activity trace without creating one.");
    }

    if (IsSassProfilerMode()) {
        GFL_LOG_DEBUG("[CuptiBackend] SASS activity policy: ",
                      UseSafeSassActivityDefaults() ? "safe" : "full",
                      " (sm=", device_facts_.compute_major,
                      device_facts_.compute_minor,
                      ", cupti_version=", device_facts_.cupti_version, ")");
    }

    // SOURCE_LOCATOR + FUNCTION activity records feed only the Activity-API
    // PC-sampling source-correlation maps (g_sourceLocatorMap /
    // g_functionNameMap, read solely by the CUPTI_ACTIVITY_KIND_PC_SAMPLING
    // handler). PcSamplingEngine now enables them on the path that consumes
    // them (both on the ActivityAPI branch; SOURCE_LOCATOR also on the
    // SamplingAPI branch), so engines that don't PC-sample - SassMetrics,
    // RangeProfiler, Trace - no longer emit records nothing reads, and
    // PcSampling stops enabling them when it falls back to the new SamplingAPI
    // on CUDA 13.x.
    // MARKER records capture NVTX push/pop ranges. NVTX is an annotation
    // layer for trace-style views (scope/kernel/memory/sync correlation), so
    // enable it by default for non-SASS engines. SASS metrics are the one
    // exception: keep MARKER off by default to avoid reintroducing the CUPTI
    // activity/SASS stability problems we guarded elsewhere. Deep is resolved
    // again after engine start, when we know whether it actually selected
    // SASS or fell back to PC sampling.
    if (ShouldEnableNvtxMarkerActivityBeforeEngine_()) {
        EnableNvtxMarkerActivity_("pre-engine");
    } else {
        LogNvtxMarkerActivityDisabled_("pre-engine");
    }

    // SYNCHRONIZATION records capture every cudaStreamSynchronize /
    // cudaDeviceSynchronize / cudaEventSynchronize / cuStreamWaitEvent
    // call with start/end timestamps. Volume is mid-scale, no anchor
    // activity kind required (CUPTI emits these regardless of which
    // API kinds are enabled). Soft-fail on enable so a CUPTI build that
    // doesn't support the kind still lets the rest of collection work.
    const bool timelineActivity = collectsKernelEvents();
    if (timelineActivity && opts_.enable_synchronization && AllowSassSyncActivity()) {
        const CUptiResult res_sync =
            cuptiActivityEnable(CUPTI_ACTIVITY_KIND_SYNCHRONIZATION);
        if (res_sync != CUPTI_SUCCESS) {
            LogCuptiIfUnexpected(
                "Synchronization",
                "cuptiActivityEnable(SYNCHRONIZATION)", res_sync);
        }
    }

    // MEMORY2 records capture cudaMalloc / cudaFree /
    // cudaMallocAsync / cudaFreeAsync / cudaMallocManaged with
    // address, bytes, and memoryKind. Gated on enable_memory_tracking
    // (default-off in the base InitOptions; opt-in until we validate
    // overhead) plus the SASS-safety gate AllowSassMemory2Activity().
    // NOT tied to kernel/timeline activity - allocation tracking is an
    // independent CUPTI activity, so it works in SASS / Deep mode too
    // (where kernel activity is off by default).
    //
    // Soft-fail on enable: older CUPTI versions (CUDA 11) shipped
    // without MEMORY2 - they had MEMORY (deprecated) which has a
    // different record shape. We don't try to fall back; if MEMORY2
    // isn't available we log and continue.
    if (opts_.enable_memory_tracking && AllowSassMemory2Activity()) {
        const CUptiResult res_mem =
            cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMORY2);
        if (res_mem != CUPTI_SUCCESS) {
            LogCuptiIfUnexpected(
                "MemoryTracking",
                "cuptiActivityEnable(MEMORY2)", res_mem);
        }
    }

    // GRAPH_TRACE captures cudaGraphLaunch with aggregate timing
    // (one record per launch, not per node). **Default-off** in v1
    // because some Blackwell driver builds reset PC sampling on first
    // graph launch - the planning doc has the full risk note. Soft-
    // fail on enable so older CUPTI without the kind keeps working.
    if (timelineActivity && opts_.enable_cuda_graphs_tracking && AllowSassGraphActivity()) {
        const CUptiResult res_g =
            cuptiActivityEnable(CUPTI_ACTIVITY_KIND_GRAPH_TRACE);
        if (res_g != CUPTI_SUCCESS) {
            LogCuptiIfUnexpected(
                "CudaGraphsTracking",
                "cuptiActivityEnable(GRAPH_TRACE)", res_g);
        }
    }

    // EXTERNAL_CORRELATION records appear when a framework brackets
    // its op with cuptiActivityPushExternalCorrelationId AND CUPTI has
    // a way to anchor each launch as an "event" worth correlating. The
    // anchoring uses CUPTI's activity-kind path (RUNTIME records),
    // NOT the cuptiSubscribe callback path we use for kernel-launch
    // metadata capture. So enabling EXTERNAL_CORRELATION alone is not
    // enough: CUPTI silently emits nothing because there's no
    // RUNTIME/DRIVER record to attach the external id to.
    //
    // We enable RUNTIME alongside EXTERNAL_CORRELATION as the smallest
    // sufficient anchor. RUNTIME captures cudaLaunchKernel / cudaMemcpy
    // / etc. as CUpti_ActivityAPI records - high volume in tight loops,
    // but we don't dispatch them anywhere; they fall through the
    // BufferCompleted handler chain and get freed with the buffer.
    // The cost is per-API-call activity record allocation in the CUPTI
    // buffer, not a per-call user callback.
    //
    // (DRIVER kind is RUNTIME's lower-level cousin. We choose RUNTIME
    // because PyTorch / TF / JAX call cudaLaunchKernel via the runtime
    // API, not the driver API directly. If a workload only uses cuLaunch
    // we may need DRIVER too - defer until we see a session that needs
    // it.)
    const bool enableExternalCorrelation =
        timelineActivity && opts_.enable_external_correlation &&
        AllowSassExternalCorrelation();
    if (opts_.enable_external_correlation && IsSassProfilerMode() &&
        !AllowSassExternalCorrelation()) {
        GFL_LOG_DEBUG(
            "[CuptiBackend] EXTERNAL_CORRELATION disabled in SASS profiler "
            "mode. Set GPUFL_SASS_ALLOW_EXTERNAL_CORRELATION=1 to test it.");
    }
    if (enableExternalCorrelation) {
        const CUptiResult res_ec =
            cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);
        if (res_ec != CUPTI_SUCCESS) {
            // Soft-fail: log and continue. If CUPTI doesn't know this kind
            // we still want kernel collection to work.
            LogCuptiIfUnexpected(
                "ExternalCorrelation",
                "cuptiActivityEnable(EXTERNAL_CORRELATION)", res_ec);
        }
        // RUNTIME + DRIVER are BOTH needed as anchors. CUPTI emits
        // EXTERNAL_CORRELATION records only for launches whose API
        // path was being tracked. PyTorch's `torch.randn` and most
        // memcpy ops go through the cudaLaunchKernel/cudaMemcpy
        // RUNTIME API, but optimized libraries like CUTLASS,
        // cuBLAS-Lt, and Triton's CUDA backend launch via the lower-
        // level cuLaunchKernel/cuLaunchKernelEx DRIVER API instead.
        //
        // Empirical evidence (Heavy_Stress_App, RTX 5060 + CUDA 13.1):
        // with only RUNTIME enabled, cutlass_sgemm kernels (51 out of
        // 53 in a typical session) get external_id = 0 because no
        // EXTERNAL_CORRELATION record is emitted for them. Adding
        // DRIVER fixes this without affecting RUNTIME-based ops.
        //
        // Cost: each cudaLaunchKernel produces TWO API records now
        // (one RUNTIME, one DRIVER) instead of one. They land in the
        // CUPTI buffer, fall through our handler chain unhandled,
        // and get freed with the buffer. Measured overhead in the
        // 50-iteration stress benchmark: <1% wall-clock difference.
        // Worth it for full F1 attribution coverage.
        const CUptiResult res_rt =
            cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);
        if (res_rt != CUPTI_SUCCESS) {
            LogCuptiIfUnexpected(
                "ExternalCorrelation",
                "cuptiActivityEnable(RUNTIME)", res_rt);
        }
        const CUptiResult res_drv =
            cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER);
        if (res_drv != CUPTI_SUCCESS) {
            LogCuptiIfUnexpected(
                "ExternalCorrelation",
                "cuptiActivityEnable(DRIVER)", res_drv);
        }
    }

    // Enable activity kinds required by registered handlers (always on).
    // handlers_ is immutable after initialize() → read lock-free.
    {
        std::set<CUpti_ActivityKind> kinds;
        for (const auto& h : handlers_)
            for (auto k : h->requiredActivityKinds()) kinds.insert(k);
        for (auto k : kinds) CUPTI_CHECK(cuptiActivityEnable(k));
    }

    // Initialize and start the engine (requires CUDA context)
    if (engine_ && haveCudaContext) {
        EngineContext ectx{ctx_, device_id_, chip_name_, &cubin_mu_,
                           &cubin_by_crc_, base_cpu_ns_, base_cupti_ts_};
        engine_->initialize(opts_, ectx);
        engine_->start();
    }

    ReenableActivityAfterEngineStart_();

    active_.store(true);
    StartActivityFlushThreadIfNeeded_();
    GFL_LOG_DEBUG("Backend started.");
}

void CuptiBackend::ReenableActivityAfterEngineStart_() {
    // Re-enable activity kinds after engine start. Some engines call
    // cuptiProfilerInitialize() or cuptiSassMetricsEnable(), which on some
    // systems (e.g. insufficient profiler privileges) can internally reset or
    // disable previously-enabled activity kinds including
    // CUPTI_ACTIVITY_KIND_KERNEL.  Re-enabling here is idempotent and ensures
    // kernel activity records are produced regardless of engine type.
    // Runs on the normal start() path AND after a deferred engine start
    // (the gating policies below are recomputed, not captured).
    {
        std::set<CUpti_ActivityKind> kinds;
        for (const auto& h : handlers_)
            for (auto k : h->requiredActivityKinds()) kinds.insert(k);
        for (auto k : kinds) cuptiActivityEnable(k);
    }

    // Re-resolve NVTX marker policy after engine start. This is primarily for
    // Deep: the request is SASS-capable, but the selected path may be PC
    // sampling if SASS declined/fell back. Engines may also reset activity
    // subscriptions during start(), so re-enable MARKER here when selected.
    if (ShouldEnableNvtxMarkerActivityForSelectedEngine_()) {
        EnableNvtxMarkerActivity_("post-engine-selected");
    } else {
        LogNvtxMarkerActivityDisabled_("post-engine-selected");
    }

    const bool timelineActivity = collectsKernelEvents();
    const bool enableExternalCorrelation =
        timelineActivity && opts_.enable_external_correlation &&
        AllowSassExternalCorrelation();

    // also re-enable EXTERNAL_CORRELATION + RUNTIME after engine
    // start. The engines above (PcSampling, SassMetrics, RangeProfiler)
    // reset ALL activity-kind subscriptions, not just kernel-related
    // ones. Neither kind is tied to any handler, so they get dropped
    // from the re-enable set - this block restores them.
    //
    // RUNTIME is the anchor that makes EXTERNAL_CORRELATION actually
    // emit records (see the pre-engine block in start() for the rationale).
    if (enableExternalCorrelation) {
        const CUptiResult ec_res =
            cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);
        const CUptiResult rt_res =
            cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);
        const CUptiResult drv_res =
            cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER);
        GFL_LOG_DEBUG(
            "[CuptiBackend] re-enable EXTERNAL_CORRELATION post-engine: ",
            (ec_res == CUPTI_SUCCESS ? "OK" : "FAILED"),
            " (CUptiResult=", static_cast<int>(ec_res), ")",
            "; RUNTIME: ",
            (rt_res == CUPTI_SUCCESS ? "OK" : "FAILED"),
            " (CUptiResult=", static_cast<int>(rt_res), ")",
            "; DRIVER: ",
            (drv_res == CUPTI_SUCCESS ? "OK" : "FAILED"),
            " (CUptiResult=", static_cast<int>(drv_res), ")");
    } else {
        GFL_LOG_DEBUG(
            "[CuptiBackend] enable_external_correlation = false; "
            "no F1 attribution will be captured");
    }

    // F2: re-enable SYNCHRONIZATION post-engine for the same reason -
    // engines reset all activity-kind subscriptions during their
    // initialize() phase. Idempotent; CUPTI ignores the second enable
    // when the kind is already on. SYNCHRONIZATION isn't tied to any
    // handler so it would otherwise be silently dropped.
    if (timelineActivity && opts_.enable_synchronization && AllowSassSyncActivity()) {
        const CUptiResult sync_res =
            cuptiActivityEnable(CUPTI_ACTIVITY_KIND_SYNCHRONIZATION);
        GFL_LOG_DEBUG(
            "[CuptiBackend] re-enable SYNCHRONIZATION post-engine: ",
            (sync_res == CUPTI_SUCCESS ? "OK" : "FAILED"),
            " (CUptiResult=", static_cast<int>(sync_res), ")");
    }

    // F3: matching post-engine re-enable for MEMORY2.
    if (opts_.enable_memory_tracking && AllowSassMemory2Activity()) {
        const CUptiResult mem_res =
            cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMORY2);
        GFL_LOG_DEBUG(
            "[CuptiBackend] re-enable MEMORY2 post-engine: ",
            (mem_res == CUPTI_SUCCESS ? "OK" : "FAILED"),
            " (CUptiResult=", static_cast<int>(mem_res), ")");
    }

    // F4: matching post-engine re-enable for GRAPH_TRACE.
    if (timelineActivity && opts_.enable_cuda_graphs_tracking && AllowSassGraphActivity()) {
        const CUptiResult g_res =
            cuptiActivityEnable(CUPTI_ACTIVITY_KIND_GRAPH_TRACE);
        GFL_LOG_DEBUG(
            "[CuptiBackend] re-enable GRAPH_TRACE post-engine: ",
            (g_res == CUPTI_SUCCESS ? "OK" : "FAILED"),
            " (CUptiResult=", static_cast<int>(g_res), ")");
    }
}

void CuptiBackend::RequestDeferredEngineStart(CUcontext ctx) {
    if (!engine_start_pending_.load(std::memory_order_acquire)) return;
    if (!ctx) return;
    // First context wins - the engines are single-context by design
    // (EngineContext carries one CUcontext).
    CUcontext expected = nullptr;
    if (!deferred_ctx_.compare_exchange_strong(expected, ctx,
                                               std::memory_order_acq_rel)) {
        return;
    }
    // Runs SYNCHRONOUSLY in the CONTEXT_CREATED callback, on the app thread
    // that created the context - the same pattern NVIDIA's
    // pc_sampling_continuous injection uses, and crucially the same thread
    // that will later run gpufl shutdown's cuptiPCSamplingStop (an earlier
    // worker-thread variant produced CUPTI_ERROR_UNKNOWN at stop). Only
    // CUPTI + driver-API calls are made here; cudart-based device facts
    // (GetSMProps) are left to their lazy call sites - cudart can re-enter
    // the driver if initialized from inside this callback.
    FinishDeferredEngineStart_();
}

void CuptiBackend::FinishDeferredEngineStart_() {
    std::lock_guard lk(deferred_start_mu_);
    if (!engine_start_pending_.load(std::memory_order_acquire)) return;
    if (!engine_) {
        engine_start_pending_.store(false, std::memory_order_release);
        return;
    }
    const CUcontext ctx = deferred_ctx_.load(std::memory_order_acquire);
    if (!ctx) return;

    // The engines' start paths guard on IsContextValid() = "current on the
    // calling thread". During CONTEXT_CREATED the new context may not be
    // bound yet - bind it, do the work, restore whatever was there.
    CUcontext prev = nullptr;
    cuCtxGetCurrent(&prev);
    if (prev != ctx && cuCtxSetCurrent(ctx) != CUDA_SUCCESS) {
        GFL_LOG_ERROR(
            "[CuptiBackend] Deferred engine start: cuCtxSetCurrent failed; "
            "engine stays disabled.");
        return;
    }
    ctx_ = ctx;

    // Device facts via CUPTI + driver API only (no cudart - see
    // RequestDeferredEngineStart).
    cuptiGetDeviceId(ctx_, &device_id_);
    chip_name_ = getChipName(device_id_);
    {
        CUdevice dev{};
        if (cuDeviceGet(&dev, static_cast<int>(device_id_)) == CUDA_SUCCESS) {
            char name[256]{};
            if (cuDeviceGetName(name, sizeof(name), dev) == CUDA_SUCCESS) {
                cached_device_name_ = name;
            }
            int cc_major = 0;
            int cc_minor = 0;
            cuDeviceGetAttribute(&cc_major,
                                 CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                 dev);
            cuDeviceGetAttribute(&cc_minor,
                                 CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                                 dev);
            device_facts_.compute_major = cc_major;
            device_facts_.compute_minor = cc_minor;
        }
    }
    device_facts_.cupti_version = GetCuptiVersion();
    resolved_plan_ = NvidiaProfilingPolicy::Resolve(
        profiling_request_, device_facts_, EnvOverrides::FromProcess());
    ApplyComboPlanOverrides(resolved_plan_, combo_);

    EngineContext ectx{ctx_, device_id_, chip_name_, &cubin_mu_,
                       &cubin_by_crc_};
    engine_->initialize(opts_, ectx);
    engine_->start();
    ReenableActivityAfterEngineStart_();

    if (prev != ctx) {
        cuCtxSetCurrent(prev);
    }

    engine_start_pending_.store(false, std::memory_order_release);
    GFL_LOG_DEBUG(
        "[CuptiBackend] Deferred engine start complete (device=", device_id_,
        ", chip=", chip_name_, ").");
}


bool CuptiBackend::collectsKernelEvents() const {
    if (!combo_.empty()) {
        // Combo: collect kernel activity iff a kernel-collecting engine is in
        // the set (Trace / PmSampling / RangeProfiler). PC sampling and SASS do
        // not collect kernel activity on their own.
        return BuildEngineRequestSet(opts_.profiling_engine, combo_)
            .ownsTimelineActivity();
    }
    // Single engine - preserve prior behavior exactly:
    //   PcSampling -> off; SASS / Deep -> off unless AllowSassKernelActivity;
    //   Trace / PmSampling / RangeProfiler -> on.
    if (opts_.profiling_engine == ProfilingEngine::PcSampling) return false;
    if (IsSassProfilerMode()) return AllowSassKernelActivity();
    return true;
}

void CuptiBackend::FlushProfilingDataBeforeCudaTeardown(const char* reason) {
    if (!initialized_ || !active_.load(std::memory_order_relaxed) || !engine_) {
        return;
    }

    // Throttle to ~50ms: cleanup APIs fire in bursts (cudaFree x3, internal
    // frees), and the drain below is idempotent, so re-running per burst is waste.
    const int64_t now = detail::GetTimestampNs();
    int64_t last = last_cleanup_flush_ns_.load(std::memory_order_relaxed);
    if (now - last < 50'000'000) return;
    if (!last_cleanup_flush_ns_.compare_exchange_strong(
            last, now, std::memory_order_acq_rel, std::memory_order_relaxed)) {
        return;
    }

    // Windows injection: the synthetic/launch-derived kernel rows (Deep/PcSampling)
    // are otherwise assembled + written only by gpufl's atexit shutdown, which
    // races the final process teardown and intermittently loses them all (the
    // collector is CPU-starved during short busy runs, so the ring isn't drained
    // until shutdown). cudaFree fires HERE on the app thread before that race
    // (cuCtxDestroy does NOT callback under injection - cudart defers context
    // teardown to DLL detach). Have the collector drain + write the kernels now
    // and wait for it. Drain-only (the collector keeps running), so it's safe for
    // workloads that free mid-run (e.g. PyTorch).
    if (WindowsInjectedProcess()) Monitor::RequestSyntheticDrainAndWait();

    if (IsSassProfilerMode()) engine_->flushBeforeCudaTeardown(reason);
}

void CuptiBackend::EngineLaunchTick() {
    if (!initialized_ || !active_.load(std::memory_order_relaxed)) return;
    if (engine_) engine_->onLaunchTick();
}

void CuptiBackend::DrainProfilingData() {
    if (!initialized_ || !active_.load(std::memory_order_relaxed)) return;
    if (engine_) {
        engine_->drainData();
    }
}

void CuptiBackend::StartActivityFlushThreadIfNeeded_() {
    // Windows injection cannot safely force-flush CUPTI activity at process
    // exit because the CUDA driver may already be tearing the context down.
    // Trace has no SamplingAPI/ProfilerAPI engine armed, so a small worker can
    // periodically force the activity buffer while the workload is still running
    // and the collector thread remains free to drain g_monitorBuffer.
    if (!WindowsInjectedProcess() || engine_ || !collectsKernelEvents()) return;

    bool expected = false;
    if (!activity_flush_thread_running_.compare_exchange_strong(
            expected, true, std::memory_order_acq_rel, std::memory_order_relaxed)) {
        return;
    }

    activity_flush_thread_ = std::thread([this] {
        while (activity_flush_thread_running_.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            if (!activity_flush_thread_running_.load(std::memory_order_acquire)) {
                break;
            }
            if (!active_.load(std::memory_order_relaxed)) continue;
            if (kernel_launch_callback_count_.load(std::memory_order_acquire) == 0) {
                continue;
            }
            LogCuptiIfUnexpected(
                "periodic-trace-drain", "cuptiActivityFlushAll",
                cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
        }
    });
}

void CuptiBackend::StopActivityFlushThread_() {
    activity_flush_thread_running_.store(false, std::memory_order_release);
    if (activity_flush_thread_.joinable()) {
        activity_flush_thread_.join();
    }
}

void CuptiBackend::stop() {
    if (!initialized_) return;
    active_.store(false);
    // A CONTEXT_CREATED callback on another app thread may be inside
    // FinishDeferredEngineStart_ - clear the pending flag and wait it out
    // so engine_->stop() below can't race engine_->start().
    engine_start_pending_.store(false, std::memory_order_release);
    { std::lock_guard lk(deferred_start_mu_); }
    StopActivityFlushThread_();

    // Stop the engine BEFORE flushing activity records.  PcSamplingEngine::stop()
    // disables the SamplingAPI session - while it's armed, cuptiActivityFlushAll
    // returns zero kernel records on driver 590+.
    if (engine_) {
        engine_->stop();
        if (const Runtime* rt = runtime(); rt && rt->logger) {
            for (auto& ev : engine_->takeKernelPerfEvents()) {
                ev.pid = detail::GetPid();
                ev.app = rt->app_name;
                ev.session_id = rt->session_id;
                rt->logger->write(model::KernelPerfMetricModel(ev));
            }
        }
    }

    // Disable all activity kinds FIRST, before the flush. The previous
    // order (sync → flush → disable) left activity tracking enabled
    // during the flush, so new records could be queued by the GPU
    // while we were draining. Those new records then arrived AFTER
    // shutdown() had cleared the active backend pointer, firing the noisy
    // "[CUPTI] BufferCompleted: No active backend!" log and (worse)
    // leaking activity into the next session's measurement on
    // benchmarks that init/shutdown gpufl repeatedly in one process -
    // run_benchmark.py's GEMM→PyTorch transition is the canonical
    // case where this surfaced (RTX 3090 + Linux). Disabling first
    // closes the queue so the subsequent flush truly drains
    // everything pending.
    //
    // Already-queued records are NOT dropped by cuptiActivityDisable;
    // they still come back through BufferCompleted during the flush
    // below. So we don't lose any data by disabling first.
    {
        std::set<CUpti_ActivityKind> kinds;
        for (const auto& h : handlers_)
            for (auto k : h->requiredActivityKinds()) kinds.insert(k);
        for (auto k : kinds) cuptiActivityDisable(k);
    }

    // Matching tear-down of the same anchor / supplementary kinds
    // start() enables. Same disable-before-flush rationale - keeps
    // the activity queue fully closed before drain.
    if (opts_.enable_external_correlation) {
        cuptiActivityDisable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);
        cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME);
        cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER);
    }
    if (opts_.enable_synchronization) {
        cuptiActivityDisable(CUPTI_ACTIVITY_KIND_SYNCHRONIZATION);
    }
    if (opts_.enable_memory_tracking) {
        cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMORY2);
    }
    if (opts_.enable_cuda_graphs_tracking) {
        cuptiActivityDisable(CUPTI_ACTIVITY_KIND_GRAPH_TRACE);
    }

    // Now drain. cuda sync first so any in-flight kernel finishes and
    // emits its end record; flush then blocks until every BufferCompleted
    // callback has returned. With kinds disabled above, no new records
    // can sneak into the queue during this window.
    if (gpufl::detail::isProcessExitTeardown()) {
        // Windows injection at-exit: the CUDA context is being destroyed by
        // cudart (its atexit runs before ours), so cudaDeviceSynchronize() and
        // cuptiActivityFlushAll(1) deadlock against the dying driver (the
        // process becomes unkillable). Skip both - activity records delivered
        // during the run via BufferCompleted are already drained; only the
        // final partial buffer is dropped. See gpufl/core/teardown_flag.hpp.
        GFL_LOG_DEBUG("CuptiBackend::stop: skip sync+flush (process-exit teardown)");
    } else {
        GFL_LOG_DEBUG("CuptiBackend::stop: cudaDeviceSynchronize() + flush");
        cudaDeviceSynchronize();
        LogCuptiIfUnexpected("Perfworks", "cuptiActivityFlushAll",
                             cuptiActivityFlushAll(1));
    }
    // Synthetic kernels (launches CUPTI delivered no activity record for) are
    // now flushed by the collector thread from its worker-local meta map once
    // the ring is fully drained (drainSyntheticKernels in monitor.cpp, invoked
    // at CollectorLoop teardown + Monitor::Shutdown's post-join drain) - see
    // Step 4b-2. This used to call FlushPendingKernels() here on the stop
    // thread. The summary counters below therefore now reflect real activity
    // records only; synthetic rows are counted/emitted later by the collector.
    ClearExternalCorrelationState();

    const uint64_t seen = kernel_activity_seen_.load(std::memory_order_relaxed);
    const uint64_t emitted =
        kernel_activity_emitted_.load(std::memory_order_relaxed);
    const uint64_t throttled =
        kernel_activity_throttled_.load(std::memory_order_relaxed);
    GFL_LOG_DEBUG("[KernelLaunchHandler] activity summary seen=", seen,
                  " emitted=", emitted, " throttled=", throttled);
}

void CuptiBackend::RegisterHandler(
    const std::shared_ptr<ICuptiHandler>& handler) {
    if (!handler) return;
    // No lock: called only from initialize() before CUPTI callbacks are enabled
    // (single-threaded setup). handlers_ is immutable once callbacks can fire.
    handlers_.push_back(handler);
}

void CuptiBackend::FlushOnContextDestroy() {
    if (!initialized_) return;

    // Skip when a profiling engine is active (PC sampling / SASS / Deep). While
    // the SamplingAPI is armed, cuptiActivityFlushAll returns zero kernel records
    // and can permanently kill the subscriber callback (driver 590+) - the same
    // reason stop() disables the engine BEFORE flushing. We can't safely stop the
    // engine from inside a context-destroy callback, so for engine modes we leave
    // the flush to the normal stop()/shutdown() path. This context-destroy flush
    // is only for the engine-less Trace/Monitor configuration (where it recovers
    // kernel rows for contexts destroyed mid-process).
    if (engine_) return;

    // Re-entrancy / concurrency guard. cuptiActivityFlushAll(1) below invokes
    // BufferCompleted synchronously on this same thread; never let any nested
    // resource callback recurse back into another flush.
    bool expected = false;
    if (!context_destroy_flushing_.compare_exchange_strong(
            expected, true, std::memory_order_acq_rel)) {
        return;
    }

    // This fires from the CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING callback:
    // the context is STILL ALIVE (the driver hasn't begun destroying it yet),
    // so cuptiActivityFlushAll is safe here and drains every completed + partial
    // activity buffer. BufferCompleted pushes the drained rows into the monitor
    // buffer for the collector to write out. This matters for contexts destroyed
    // MID-PROCESS (an explicit cudaDeviceReset/cuCtxDestroy, or multi-context
    // apps), where our at-exit shutdown() skips cuptiActivityFlushAll (it would
    // deadlock against a dying context; see teardown_flag.hpp). It does NOT fire
    // on Windows process exit - cudart leaves context teardown to driver
    // DLL-detach there, so no callback arrives; the final kernel records on
    // Windows-exit are recovered by Monitor::Shutdown's post-join drain instead.
    //
    // No cudaDeviceSynchronize(): re-entering cudart mid-teardown is exactly the
    // hazard we're avoiding. A force-flush (argument 1) is sufficient.
    GFL_LOG_DEBUG(
        "CuptiBackend::FlushOnContextDestroy: flushing activity before context "
        "teardown (context still alive)");
    // GetSMProps (the occupancy calc in the kernel-activity handler) must not
    // call cudaGetDeviceProperties while we flush here: the drained kernel
    // records are processed on THIS thread, and re-entering cudart against the
    // dying context deadlocks. Warm cache hits are fine; cold misses fall back.
    SetSmPropsTeardownSafe(true);
    LogCuptiIfUnexpected("ContextDestroy", "cuptiActivityFlushAll",
                         cuptiActivityFlushAll(1));
    SetSmPropsTeardownSafe(false);

    context_destroy_flushing_.store(false, std::memory_order_release);
}

void CuptiBackend::FlushActivityNow() {
    if (!initialized_ || !active_.load(std::memory_order_relaxed)) return;
    // Only the Windows-injection Trace case needs this (same gate as the
    // periodic flush thread): elsewhere the at-exit flush already drains.
    if (engine_ || !collectsKernelEvents() || !WindowsInjectedProcess()) return;
    // Throttle: a sync-heavy app must not force-flush on every synchronize.
    static std::atomic<int64_t> last_ns{0};
    const int64_t now = detail::GetTimestampNs();
    int64_t last = last_ns.load(std::memory_order_relaxed);
    if (now - last < 50'000'000) return;  // 50 ms
    if (!last_ns.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
        return;  // another synchronize raced us; it will flush
    }
    LogCuptiIfUnexpected("sync-flush", "cuptiActivityFlushAll",
                         cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
}

}  // namespace gpufl

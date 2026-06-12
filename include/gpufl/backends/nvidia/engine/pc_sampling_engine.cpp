#include "gpufl/backends/nvidia/engine/pc_sampling_engine.hpp"

#include <cuda_runtime.h>
#include <cupti.h>
#include <cupti_pcsampling.h>
#include <cupti_profiler_target.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>

#include "gpufl/backends/nvidia/cupti_utils.hpp"
#include "gpufl/backends/nvidia/sampler/cupti_sass.hpp"
#include "gpufl/core/activity_record.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/env_vars.hpp"
#include "gpufl/core/ring_buffer.hpp"
#include "gpufl/core/teardown_flag.hpp"

#ifndef CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SOURCE_REPORTING
#define CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SOURCE_REPORTING \
    ((CUpti_PCSamplingConfigurationAttributeType)10)
#endif

namespace gpufl {

namespace {
bool IsInsufficientPrivilege(const CUptiResult res) {
    if (res == CUPTI_ERROR_INSUFFICIENT_PRIVILEGES) return true;
#ifdef CUPTI_ERROR_VIRTUALIZED_DEVICE_INSUFFICIENT_PRIVILEGES
    if (res == CUPTI_ERROR_VIRTUALIZED_DEVICE_INSUFFICIENT_PRIVILEGES)
        return true;
#endif
    return false;
}

constexpr size_t kPcSamplingConfigAttrCount = 7;

std::array<CUpti_PCSamplingConfigurationInfo, kPcSamplingConfigAttrCount>
BuildPcSamplingConfig(const uint32_t samplingPeriod,
                      CUpti_PCSamplingData* const samplingData) {
    std::array<CUpti_PCSamplingConfigurationInfo, kPcSamplingConfigAttrCount>
        configInfo{};

    size_t configCount = 0;
    auto addConfig = [&](const CUpti_PCSamplingConfigurationInfo& info) {
        configInfo[configCount++] = info;
    };

    // Kernel-serialized collection plus explicit start/stop lets GPUFL own
    // the PC sampling lifetime while avoiding mid-session GetData drains.
    {
        CUpti_PCSamplingConfigurationInfo info = {};
        info.attributeType =
            CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_COLLECTION_MODE;
        info.attributeData.collectionModeData.collectionMode =
            CUPTI_PC_SAMPLING_COLLECTION_MODE_KERNEL_SERIALIZED;
        addConfig(info);
    }
    {
        CUpti_PCSamplingConfigurationInfo info = {};
        info.attributeType =
            CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_PERIOD;
        info.attributeData.samplingPeriodData.samplingPeriod = samplingPeriod;
        addConfig(info);
    }
    {
        CUpti_PCSamplingConfigurationInfo info = {};
        info.attributeType =
            CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SCRATCH_BUFFER_SIZE;
        // Host-resident staging between HW buffer and GetData. CUPTI sizing:
        // ~1 MB per ~5,500 PCs with all stall reasons, so 32 MB covers
        // ~175k distinct PCs per drain window - generous at our 1 s collect
        // cadence (the old 256 MB was wildly oversized per context).
        info.attributeData.scratchBufferSizeData.scratchBufferSize =
            32 * 1024 * 1024;
        addConfig(info);
    }
    {
        CUpti_PCSamplingConfigurationInfo info = {};
        info.attributeType =
            CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_HARDWARE_BUFFER_SIZE;
        info.attributeData.hardwareBufferSizeData.hardwareBufferSize =
            256 * 1024 * 1024;
        addConfig(info);
    }

    // Explicit start/stop is required before cuptiPCSamplingStart/Stop.
    // Do not call cuptiPCSamplingGetData while sampling is active; on CUDA
    // 13.x this can drain the buffer and leave the final collection empty.
    {
        CUpti_PCSamplingConfigurationInfo info = {};
        info.attributeType =
            CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_ENABLE_START_STOP_CONTROL;
        info.attributeData.enableStartStopControlData.enableStartStopControl =
            1;
        addConfig(info);
    }
    {
        CUpti_PCSamplingConfigurationInfo info = {};
        info.attributeType =
            CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_DATA_BUFFER;
        info.attributeData.samplingDataBufferData.samplingDataBuffer =
            samplingData;
        addConfig(info);
    }
    {
        CUpti_PCSamplingConfigurationInfo info = {};
        info.attributeType =
            CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_OUTPUT_DATA_FORMAT;
        info.attributeData.outputDataFormatData.outputDataFormat =
            CUPTI_PC_SAMPLING_OUTPUT_DATA_FORMAT_PARSED;
        addConfig(info);
    }

    return configInfo;
}
}  // namespace

// ---- PCSamplingDeleter -----------------------------------------------------

void PCSamplingDeleter::operator()(const PCSamplingBuffers* b) const {
    if (!b) return;
    if (b->pcRecords && b->data) {
        const size_t maxPcs = b->data->collectNumPcs;
        for (size_t i = 0; i < maxPcs; ++i) {
            if (b->pcRecords[i].stallReason) {
                std::free(b->pcRecords[i].stallReason);
            }
        }
        std::free(b->pcRecords);
    }
    if (b->data) std::free(b->data);
    delete b;
}

// ---- PcSamplingEngine ------------------------------------------------------

bool PcSamplingEngine::initialize(const MonitorOptions& opts,
                                  const EngineContext& ctx) {
    opts_ = opts;
    ctx_ = ctx;
    pc_sampling_method_ = Method::None;
    pc_sampling_ref_count_.store(0);
    sampling_api_ready_.store(false);
    sampling_api_started_.store(false);
    sampling_api_blocked_.store(false);

    // Kernel-timeline collection mode. PC/SASS already has launch-callback
    // kernel rows, so keep the periodic CUPTI path light by default; the
    // stop/flush/start drain is useful for experiments but has proven timing
    // sensitive with some PyTorch/CUPTI combinations.
    kernel_collect_ = KernelCollect::None;
    if (const char* v = std::getenv(env::kPcKernelCollect)) {
        if (std::strcmp(v, "all") == 0) kernel_collect_ = KernelCollect::All;
        else if (std::strcmp(v, "none") == 0) kernel_collect_ = KernelCollect::None;
    }
    GFL_LOG_DEBUG("[PcSamplingEngine] initialized (kernel_collect=",
                  static_cast<int>(kernel_collect_), ")");
    return true;
}

void PcSamplingEngine::start() {
    pc_sampling_ref_count_.store(0);
    sampling_api_started_.store(false);
    sampling_api_ready_.store(false);
    sampling_api_blocked_.store(false);

    CUptiResult pcRes = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_PC_SAMPLING);

    if (pcRes == CUPTI_SUCCESS) {
        pc_sampling_method_ = Method::ActivityAPI;
        // CUpti_ActivityPCSampling3 carries only sourceLocatorId + functionId,
        // so the ActivityAPI sampler needs the SOURCE_LOCATOR and FUNCTION
        // companion records to resolve file/line/function. These used to be
        // enabled unconditionally in CuptiBackend::start(); they live here now
        // so only the engine that consumes them turns them on
        // (CuptiBackend::shutdown() disables both).
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR);
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_FUNCTION);
        GFL_LOG_DEBUG(
            "[PC Sampling] Using Activity API "
            "(CUPTI_ACTIVITY_KIND_PC_SAMPLING)");
    } else if (pcRes == CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED) {
        GFL_LOG_DEBUG(
            "[PC Sampling] Activity API not supported, using PC Sampling "
            "API...");
        pc_sampling_method_ = Method::SamplingAPI;
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR);
        GFL_LOG_DEBUG("[PC Sampling] samplingPeriod=", opts_.pc_sampling_period,
                      " (2^", opts_.pc_sampling_period, " = ",
                      (1u << opts_.pc_sampling_period), " cycles/sample)");
        // Real kernel timeline alongside PC sampling — verified to coexist.
        // Independent of arm success: if sampling is unavailable the pass
        // still degrades to a kernel trace. Synthetic-kernel fallback stays
        // suppressed (cupti_backend.cpp start()), so only REAL records show.
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
        // Arm now — start() runs in the CONTEXT_CREATED callback, before the
        // app's first kernel, while the GPU is quiet. Enable/config/Start
        // return INVALID_OPERATION when kernels run concurrently (verified
        // live), so pre-first-kernel is the only reliable window. profiler-
        // init already ran pre-context, so stall enumeration succeeds here.
        {
            std::lock_guard lk(sampling_lifecycle_mu_);
            StartPcSampling_();
        }
        // Enable can internally disable kernel activity — re-assert it.
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
        StartCycleThread_();
    } else {
        LogCuptiErrorIfFailed(this->name(), "cuptiActivityEnable(PC_SAMPLING)",
                              pcRes);
        if (IsInsufficientPrivilege(pcRes)) {
            sampling_api_blocked_.store(true);
            GFL_LOG_ERROR(
                "[PC Sampling] CUPTI profiling permissions are restricted for "
                "this user. Enable GPU performance counter access or run with "
                "elevated privileges.");
        }
        pc_sampling_method_ = Method::None;
    }
}

void PcSamplingEngine::StartCycleThread_() {
    if (cycle_thread_running_.exchange(true)) return;
    // Let the first plain-thread drain happen on the first 250 ms tick. The
    // final Windows-injected teardown path cannot safely flush activity, so
    // waiting a whole second here drops short sessions.
    last_kernel_drain_ns_.store(0, std::memory_order_relaxed);
    last_sample_collect_ns_.store(0, std::memory_order_relaxed);
    GFL_LOG_DEBUG("[PC Sampling] launching collection cycle thread");
    cycle_thread_ = std::thread([this] {
        GFL_LOG_DEBUG("[PC Sampling] cycle thread running");
        // NO per-tick logging in this loop: under a debug-mode log flood
        // (launch callbacks logging from several app threads) the shared
        // stream lock starves this thread for tens of seconds - observed
        // live: ticks stopped the moment the kernel-launch flood began.
        // drainData logs only AFTER its CUPTI work, so even a starved log
        // call can't prevent collection, only delay the next cycle.
        while (cycle_thread_running_.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            if (!cycle_thread_running_.load(std::memory_order_acquire)) break;
            drainData();   // internally throttled (kCollectIntervalNs)
        }
    });
}

void PcSamplingEngine::StopCycleThread_() {
    cycle_thread_running_.store(false, std::memory_order_release);
    if (cycle_thread_.joinable()) cycle_thread_.join();
}

void PcSamplingEngine::stop() {
    // Disable the SamplingAPI session before the activity flush in
    // CuptiBackend::stop().  While the SamplingAPI is armed, any CUPTI
    // data-retrieval call (FlushAll, GetData, Disable) can permanently
    // kill the subscriber callback on driver 590+.  Disabling here - in
    // the correct order (before flush) - ensures cuptiActivityFlushAll
    // delivers real kernel activity records with correct GPU durations.
    StopCycleThread_();   // join BEFORE the lock - the cycle takes it too
    std::lock_guard lk(sampling_lifecycle_mu_);
    if (pc_sampling_method_ == Method::SamplingAPI &&
        sampling_api_ready_.load() && ctx_.cuda_ctx) {
        if (sampling_api_started_.load()) {
            pc_sampling_ref_count_.store(1);
            StopAndCollectPcSampling_();
        }
        CUpti_PCSamplingDisableParams dp = {};
        dp.size = sizeof(CUpti_PCSamplingDisableParams);
        dp.ctx = ctx_.cuda_ctx;
        cuptiPCSamplingDisable(&dp);
        sampling_api_ready_.store(false);
    }
}

void PcSamplingEngine::drainData() {
    // Periodic collection on the engine's cycle thread (deferring all of it to
    // session stop loses the session to process-exit teardown). Two paths by
    // kernel_collect_: light = armed GetData (no Stop; KERNEL_SERIALIZED still
    // returns completed kernels' samples); heavy = DrainKernelsAndCollect_,
    // which Stops to force a kernel-activity flush for a full timeline.
    if (pc_sampling_method_ != Method::SamplingAPI) return;
    if (!sampling_api_started_.load()) return;

    // GetData requires the context current on the calling thread. Binding
    // is thread-local and this thread is ours, so leave it bound.
    if (!ctx_.cuda_ctx) return;
    if (cuCtxSetCurrent(ctx_.cuda_ctx) != CUDA_SUCCESS) return;

    // Drain kernel activity (heavy: stop->flush->start) only when explicitly
    // requested; otherwise just collect PC samples (light).
    const bool want_drain =
        kernel_collect_ == KernelCollect::All &&
        !drain_unavailable_.load(std::memory_order_relaxed);
    if (want_drain) {
        DrainKernelsAndCollect_();
    } else {
        MaybePeriodicCollect_("cycle", /*force=*/false);
    }
}

void PcSamplingEngine::DrainKernelsAndCollect_() {
    const int64_t now = detail::GetTimestampNs();
    const int64_t last = last_kernel_drain_ns_.load(std::memory_order_relaxed);
    if (last != 0 && now - last < kCollectIntervalNs) return;
    if (!sampling_lifecycle_mu_.try_lock()) return;
    std::lock_guard lk(sampling_lifecycle_mu_, std::adopt_lock);
    if (!sampling_api_started_.load()) return;
    last_kernel_drain_ns_.store(now, std::memory_order_relaxed);

    // Stop sampling: required because a forced activity flush returns zero
    // kernel records while PC sampling is armed (driver 590+). Stop/Start
    // mid-run are privileged (INSUFFICIENT_PRIVILEGES under a non-elevated
    // run) — on that error, stop draining for the session and let armed
    // GetData carry the PC samples. Restart after the flush; it succeeds
    // even with kernels running (unlike the initial arm).
    CUpti_PCSamplingStopParams sp = {};
    sp.size = sizeof(sp);
    sp.ctx = ctx_.cuda_ctx;
    const CUptiResult rStop = cuptiPCSamplingStop(&sp);
    if (rStop != CUPTI_SUCCESS) {
        if (IsInsufficientPrivilege(rStop)) {
            drain_unavailable_.store(true, std::memory_order_relaxed);
            GFL_LOG_DEBUG("[PC Sampling] kernel drain needs elevation (stop=",
                          rStop, "); falling back to sample-only collection.");
        }
        return;  // sampling is still armed (Stop failed)
    }

    cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
    CollectPcSamplingData_();

    CUpti_PCSamplingStartParams stp = {};
    stp.size = sizeof(stp);
    stp.ctx = ctx_.cuda_ctx;
    if (const CUptiResult rStart = cuptiPCSamplingStart(&stp);
        rStart != CUPTI_SUCCESS) {
        sampling_api_started_.store(false);
        LogCuptiErrorIfFailed(this->name(), "cuptiPCSamplingStart(drain-restart)",
                              rStart);
    }
}

void PcSamplingEngine::shutdown() {
    StopCycleThread_();   // join BEFORE the lock - the cycle takes it too
    std::lock_guard lk(sampling_lifecycle_mu_);
    // Collect a still-active SamplingAPI session before teardown.
    if (sampling_api_started_.load() &&
        pc_sampling_method_ == Method::SamplingAPI) {
        pc_sampling_ref_count_.store(1);
        StopAndCollectPcSampling_();
    }

    if (sampling_api_ready_.load() && ctx_.cuda_ctx) {
        CUpti_PCSamplingDisableParams disableParams = {};
        disableParams.size = sizeof(CUpti_PCSamplingDisableParams);
        disableParams.ctx = ctx_.cuda_ctx;
        const CUptiResult disableRes = cuptiPCSamplingDisable(&disableParams);
        if (disableRes != CUPTI_SUCCESS &&
            disableRes != CUPTI_ERROR_NOT_INITIALIZED &&
            !IsInsufficientPrivilege(disableRes)) {
            LogCuptiErrorIfFailed(this->name(), "cuptiPCSamplingDisable",
                                  disableRes);
        }
    }

    sampling_api_ready_.store(false);
    sampling_api_started_.store(false);
    sampling_api_blocked_.store(false);
    pc_sampling_ref_count_.store(0);
    pc_sampling_buffers_.reset();
}

void PcSamplingEngine::onScopeStart(const char* /*name*/) {
    // Idempotent re-arm - start() already arms the whole session; this
    // only matters if a prior arm attempt failed (e.g. context raced).
    std::lock_guard lk(sampling_lifecycle_mu_);
    StartPcSampling_();
}

void PcSamplingEngine::onScopeStop(const char* /*name*/) {
    // Forced collect at scope end. For the process-wide scope this is the
    // last healthy moment before Windows process-exit teardown breaks
    // cuptiPCSamplingStop with CUPTI_ERROR_UNKNOWN. Re-arms afterwards, so
    // nested/subsequent scopes keep sampling.
    MaybePeriodicCollect_("scope-stop", /*force=*/true);
}

// ---- Private helpers -------------------------------------------------------

bool PcSamplingEngine::EnableSamplingFeatures_() {
    if (pc_sampling_method_ != Method::SamplingAPI) return false;
    if (sampling_api_blocked_.load()) return false;
    if (sampling_api_ready_.load()) return true;

    GFL_LOG_DEBUG("[PcSamplingEngine] Configuring PC Sampling...");

    if (!ctx_.cuda_ctx) {
        GFL_LOG_ERROR(
            "[GPUFL] Cannot configure PC Sampling: cuda_ctx is NULL!");
        return false;
    }

    CUpti_PCSamplingEnableParams enableParams = {};
    enableParams.size = sizeof(CUpti_PCSamplingEnableParams);
    enableParams.ctx = ctx_.cuda_ctx;
    const CUptiResult enableRes = cuptiPCSamplingEnable(&enableParams);
    if (enableRes != CUPTI_SUCCESS &&
        enableRes != CUPTI_ERROR_INVALID_OPERATION) {
        LogCuptiErrorIfFailed(this->name(), "cuptiPCSamplingEnable", enableRes);
        if (IsInsufficientPrivilege(enableRes)) {
            sampling_api_blocked_.store(true);
            pc_sampling_method_ = Method::None;
            GFL_LOG_ERROR(
                "[PC Sampling] Insufficient privileges: disabling PC "
                "sampling for this session.");
        }
        return false;
    }
    if (enableRes == CUPTI_ERROR_INVALID_OPERATION) {
        // Typically means PC sampling is already enabled, or the Profiler API
        // (cuptiProfilerInitialize) was called first and may conflict.
        GFL_LOG_DEBUG(
            "[PC Sampling] cuptiPCSamplingEnable returned "
            "INVALID_OPERATION - possibly already enabled or "
            "conflicting with Profiler API; continuing.");
    }

    if (!pc_sampling_buffers_) {
        constexpr size_t kMaxPcs = 65536;
        pc_sampling_buffers_ =
            std::unique_ptr<PCSamplingBuffers, PCSamplingDeleter>(
                new PCSamplingBuffers());
        pc_sampling_buffers_->pcRecords = static_cast<CUpti_PCSamplingPCData*>(
            std::calloc(kMaxPcs, sizeof(CUpti_PCSamplingPCData)));

        CUpti_PCSamplingGetNumStallReasonsParams numParams = {};
        numParams.size = sizeof(CUpti_PCSamplingGetNumStallReasonsParams);
        numParams.ctx = ctx_.cuda_ctx;
        size_t numStallReasons = 0;
        numParams.numStallReasons = &numStallReasons;

        const CUptiResult numRes = cuptiPCSamplingGetNumStallReasons(&numParams);
        if (numRes != CUPTI_SUCCESS || numStallReasons == 0) {
            // Zero stall reasons = no usable PC sampling counters (usually a
            // CUPTI older than the driver: profiler-init fails too). Enable
            // would still "succeed" but Stop/GetData then error and the
            // session collects nothing — disable cleanly and report instead.
            stall_reasons_unavailable_.store(true);
            pc_sampling_method_ = Method::None;
            GFL_LOG_ERROR(
                "[PC Sampling] cuptiPCSamplingGetNumStallReasons returned ",
                numRes, " with ", numStallReasons,
                " stall reasons - disabling for this session. This usually "
                "means the CUPTI runtime is older than the installed "
                "display driver supports (cuptiProfilerInitialize also "
                "fails with NOT_INITIALIZED in that state) - update gpufl "
                "or the CUDA toolkit it was built with to match the driver "
                "generation.");
            CUpti_PCSamplingDisableParams dp = {};
            dp.size = sizeof(CUpti_PCSamplingDisableParams);
            dp.ctx = ctx_.cuda_ctx;
            cuptiPCSamplingDisable(&dp);
            pc_sampling_buffers_.reset();
            return false;
        }
        {
            auto* stallIndices = static_cast<uint32_t*>(
                malloc(numStallReasons * sizeof(uint32_t)));
            char** stallReasonNames =
                static_cast<char**>(malloc(numStallReasons * sizeof(char*)));
            for (size_t i = 0; i < numStallReasons; i++) {
                stallReasonNames[i] =
                    static_cast<char*>(malloc(CUPTI_STALL_REASON_STRING_SIZE));
            }

            CUpti_PCSamplingGetStallReasonsParams getParams = {
                sizeof(CUpti_PCSamplingGetStallReasonsParams)};
            getParams.ctx = ctx_.cuda_ctx;
            getParams.pPriv = nullptr;
            getParams.numStallReasons = numStallReasons;
            getParams.stallReasonIndex = stallIndices;
            getParams.stallReasons = stallReasonNames;

            CUptiResult res = cuptiPCSamplingGetStallReasons(&getParams);
            if (res == CUPTI_SUCCESS) {
                std::lock_guard lk(stall_reason_mu_);
                for (size_t i = 0; i < numStallReasons; i++) {
                    stall_reason_map_[stallIndices[i]] =
                        std::string(stallReasonNames[i]);
                    GFL_LOG_DEBUG("Mapped Stall ", stallIndices[i], " to ",
                                  stallReasonNames[i]);
                    free(stallReasonNames[i]);
                }
            } else {
                GFL_LOG_ERROR(
                    "[PcSamplingEngine] cuptiPCSamplingGetStallReasons "
                    "failed: ",
                    res);
            }
            free(stallIndices);
            free(stallReasonNames);
        }

        for (size_t i = 0; i < kMaxPcs; ++i) {
            pc_sampling_buffers_->pcRecords[i].size =
                sizeof(CUpti_PCSamplingPCData);
            pc_sampling_buffers_->pcRecords[i].stallReasonCount =
                numStallReasons;
            pc_sampling_buffers_->pcRecords[i].stallReason =
                static_cast<CUpti_PCSamplingStallReason*>(std::calloc(
                    numStallReasons, sizeof(CUpti_PCSamplingStallReason)));
        }
        pc_sampling_buffers_->data = static_cast<CUpti_PCSamplingData*>(
            std::calloc(1, sizeof(CUpti_PCSamplingData)));
        pc_sampling_buffers_->data->size = sizeof(CUpti_PCSamplingData);
        pc_sampling_buffers_->data->collectNumPcs = kMaxPcs;
        pc_sampling_buffers_->data->pPcData = pc_sampling_buffers_->pcRecords;
        pc_sampling_buffers_->data->totalNumPcs = 0;
        num_stall_reasons_ = numStallReasons;
    }

    auto configInfo = BuildPcSamplingConfig(opts_.pc_sampling_period,
                                            pc_sampling_buffers_->data);

    CUpti_PCSamplingConfigurationInfoParams configParams = {};
    configParams.size = CUpti_PCSamplingConfigurationInfoParamsSize;
    configParams.ctx = ctx_.cuda_ctx;
    configParams.numAttributes =
        static_cast<decltype(configParams.numAttributes)>(configInfo.size());
    configParams.pPCSamplingConfigurationInfo = configInfo.data();

    const CUptiResult configRes =
        cuptiPCSamplingSetConfigurationAttribute(&configParams);
    if (configRes != CUPTI_SUCCESS &&
        configRes != CUPTI_ERROR_INVALID_OPERATION) {
        LogCuptiErrorIfFailed(this->name(),
                              "cuptiPCSamplingSetConfigurationAttribute",
                              configRes);
        if (IsInsufficientPrivilege(configRes)) {
            sampling_api_blocked_.store(true);
            pc_sampling_method_ = Method::None;
            GFL_LOG_ERROR(
                "[PC Sampling] Insufficient privileges: disabling PC "
                "sampling for this session.");
        }
        return false;
    }

    sampling_api_ready_.store(true);

    GFL_LOG_DEBUG("[PC Sampling] configured and enabled successfully.");
    return true;
}

void PcSamplingEngine::StartPcSampling_() {
    if (pc_sampling_method_ != Method::SamplingAPI ||
        sampling_api_blocked_.load()) {
        return;
    }

    if (int expected = 0;
        !pc_sampling_ref_count_.compare_exchange_strong(expected, 1)) {
        const int refs = pc_sampling_ref_count_.fetch_add(1) + 1;
        GFL_LOG_DEBUG("[PC Sampling] already active (RefCount=", refs, ")");
        return;
    }

    if (!EnableSamplingFeatures_()) {
        pc_sampling_ref_count_.store(0);
        return;
    }

    if (!ctx_.cuda_ctx || !IsContextValid(ctx_.cuda_ctx)) {
        pc_sampling_ref_count_.store(0);
        GFL_LOG_ERROR("[GPUFL] Cannot start PC Sampling: Context invalid.");
        return;
    }

    CUpti_PCSamplingStartParams startParams = {};
    startParams.size = sizeof(CUpti_PCSamplingStartParams);
    startParams.ctx = ctx_.cuda_ctx;
    const CUptiResult startRes = cuptiPCSamplingStart(&startParams);
    if (startRes != CUPTI_SUCCESS) {
        pc_sampling_ref_count_.store(0);
        if (IsInsufficientPrivilege(startRes)) {
            sampling_api_blocked_.store(true);
        }
        if (detail::isProcessExitTeardown()) {
            // Expected when a final collect re-arms during Windows
            // process-exit teardown - there is no next window to sample.
            GFL_LOG_DEBUG("[PC Sampling] cuptiPCSamplingStart failed during "
                          "process-exit teardown (", startRes,
                          ") - no further sampling windows.");
        } else {
            LogCuptiErrorIfFailed(this->name(), "cuptiPCSamplingStart",
                                  startRes);
        }
        return;
    }

    sampling_api_started_.store(true);
    GFL_LOG_DEBUG("[PC Sampling] >>> STARTED (Scope Begin) <<<");
}

void PcSamplingEngine::flushBeforeCudaTeardown(const char* reason) {
    MaybePeriodicCollect_(reason, /*force=*/false);
}

void PcSamplingEngine::onLaunchTick() {
    // Collect from the launch API_ENTER callback (app thread) — the only
    // beat that reliably fires on Windows-injected runs. Deferring to
    // session stop loses everything to the process-exit 999.
    MaybePeriodicCollect_("launch-tick", /*force=*/false);
}

void PcSamplingEngine::MaybePeriodicCollect_(const char* reason,
                                             const bool force) {
    if (pc_sampling_method_ != Method::SamplingAPI) return;
    if (!sampling_api_started_.load()) return;

    if (!force) {
        const int64_t now = detail::GetTimestampNs();
        const int64_t last =
            last_sample_collect_ns_.load(std::memory_order_relaxed);
        if (last != 0 && now - last < kCollectIntervalNs) return;
    }

    // try_lock: callers are inside CUPTI callbacks - never wait on a lock
    // the stop/shutdown path holds while it makes CUPTI calls.
    if (!sampling_lifecycle_mu_.try_lock()) return;
    std::lock_guard lk(sampling_lifecycle_mu_, std::adopt_lock);
    if (!sampling_api_started_.load()) return;
    last_sample_collect_ns_.store(detail::GetTimestampNs(),
                                  std::memory_order_relaxed);

    GFL_LOG_DEBUG("[PC Sampling] periodic collect (", reason ? reason : "?",
                  force ? ", forced" : "", ")");
    // Armed GetData, no Stop: Stop returns 999 inside a CUPTI callback, and
    // in KERNEL_SERIALIZED mode GetData mid-session returns every completed
    // kernel's samples. Sampling stays armed; no re-arm needed.
    CollectPcSamplingData_();
}

void PcSamplingEngine::StopAndCollectPcSampling_(const bool sync_device) {
    GFL_LOG_DEBUG("[PC Sampling] StopAndCollect entry: method=",
                  static_cast<int>(pc_sampling_method_),
                  " refCount=", pc_sampling_ref_count_.load(),
                  " started=", sampling_api_started_.load());
    if (pc_sampling_method_ != Method::SamplingAPI) {
        GFL_LOG_DEBUG("[PC Sampling] StopAndCollect: exit - method != SamplingAPI");
        return;
    }

    const int refs = pc_sampling_ref_count_.load();
    if (refs <= 0) {
        GFL_LOG_DEBUG("[PC Sampling] StopAndCollect: exit - no active scope");
        return;
    }
    if (refs > 1) {
        const int remaining = pc_sampling_ref_count_.fetch_sub(1) - 1;
        GFL_LOG_DEBUG("[PC Sampling] still active (RefCount=", remaining, ")");
        return;
    }
    pc_sampling_ref_count_.store(0);

    if (!sampling_api_started_.exchange(false)) {
        GFL_LOG_DEBUG("[PC Sampling] StopAndCollect: exit - already collected / not started");
        return;
    }

    if (!ctx_.cuda_ctx || !IsContextValid(ctx_.cuda_ctx)) {
        GFL_LOG_ERROR("[GPUFL] Aborting PC Sampling: Context invalid.");
        return;
    }

    if (!pc_sampling_buffers_ || !pc_sampling_buffers_->data) {
        GFL_LOG_ERROR("[GPUFL] No PC sampling buffers allocated!");
        return;
    }

    const auto collectNumPcs = pc_sampling_buffers_->data->collectNumPcs;
    GFL_LOG_DEBUG("[PC Sampling] <<< COLLECTING >>> collectNumPcs=",
                  collectNumPcs);
    if (sync_device) {
        cudaDeviceSynchronize();
    }

    CUpti_PCSamplingStopParams stopParams = {};
    stopParams.size = sizeof(CUpti_PCSamplingStopParams);
    stopParams.ctx = ctx_.cuda_ctx;
    const CUptiResult stopRes = cuptiPCSamplingStop(&stopParams);
    if (stopRes != CUPTI_SUCCESS) {
        if (IsInsufficientPrivilege(stopRes) ||
            stopRes == CUPTI_ERROR_NOT_INITIALIZED) {
            sampling_api_blocked_.store(true);
        }
        if (detail::isProcessExitTeardown()) {
            // Expected on Windows-injected exit: the driver is mid-teardown.
            // The periodic drainData() cycles already collected the session;
            // only the final (≤ one cycle) window is lost.
            GFL_LOG_DEBUG("[PC Sampling] cuptiPCSamplingStop failed during "
                          "process-exit teardown (", stopRes,
                          ") - last window lost, prior cycles already "
                          "collected.");
        } else {
            LogCuptiErrorIfFailed(this->name(), "cuptiPCSamplingStop", stopRes);
        }
        return;
    }

    CollectPcSamplingData_();
}

void PcSamplingEngine::CollectPcSamplingData_() {
    if (!ctx_.cuda_ctx || !pc_sampling_buffers_ || !pc_sampling_buffers_->data) {
        return;
    }

    CUpti_PCSamplingGetDataParams getDataParams = {};
    getDataParams.size = sizeof(CUpti_PCSamplingGetDataParams);
    getDataParams.ctx = ctx_.cuda_ctx;
    getDataParams.pcSamplingData = pc_sampling_buffers_->data;

    while (true) {
        // stallReasonCount is both input (available slots) and output (slots
        // written).  If the previous getData call wrote fewer than
        // num_stall_reasons_ stall reasons (including 0), those entries now
        // have a smaller capacity from CUPTI's perspective.  Reset to the
        // original allocation size so CUPTI can always fill at least one
        // record, preventing an infinite loop where hasMore is true but
        // totalNumPcs=0.
        if (num_stall_reasons_ > 0) {
            const size_t cap = pc_sampling_buffers_->data->collectNumPcs;
            for (size_t i = 0; i < cap; ++i)
                pc_sampling_buffers_->pcRecords[i].stallReasonCount =
                    num_stall_reasons_;
        }

        pc_sampling_buffers_->data->totalNumPcs = 0;
        CUptiResult getRes = cuptiPCSamplingGetData(&getDataParams);
        const bool hasMore = (getRes == CUPTI_ERROR_OUT_OF_MEMORY);

        if (getRes != CUPTI_SUCCESS && !hasMore) {
            if (IsInsufficientPrivilege(getRes) ||
                getRes == CUPTI_ERROR_NOT_INITIALIZED) {
                // NOT_INITIALIZED: Profiler API (cuptiProfilerInitialize) was
                // called before PC Sampling API - they are mutually exclusive
                // on Turing+ GPUs.  Disable to suppress repeated errors.
                GFL_LOG_DEBUG("[PC Sampling] getData failed (", getRes,
                              ") - "
                              "disabling PC sampling for this session.");
                sampling_api_blocked_.store(true);
            } else {
                LogCuptiErrorIfFailed(this->name(), "cuptiPCSamplingGetData",
                                      getRes);
            }
            break;
        }

        const auto numPcs = pc_sampling_buffers_->data->totalNumPcs;
        if (numPcs > 0) produced_data_.store(true, std::memory_order_relaxed);
        // The hardware-side counters tell zero-record collections apart:
        // totalSamples=0 means the GPU never sampled (period/perms/arming),
        // while totalSamples>0 with numPcs=0 means samples were taken but
        // attributed to non-user kernels or dropped before retrieval.
        // Copies, not field refs: CUpti_PCSamplingData is packed and GCC
        // refuses to bind packed fields to the logger's references.
        const uint64_t totalSamples = pc_sampling_buffers_->data->totalSamples;
        const uint64_t droppedSamples =
            pc_sampling_buffers_->data->droppedSamples;
        const uint64_t nonUsrSamples =
            pc_sampling_buffers_->data->nonUsrKernelsTotalSamples;
        GFL_LOG_DEBUG("[PC Sampling] Collected ", numPcs, " PC records",
                      (hasMore ? " (more remaining)" : ""),
                      "; totalSamples=", totalSamples,
                      " droppedSamples=", droppedSamples,
                      " nonUsrKernelsTotalSamples=", nonUsrSamples);

        for (size_t i = 0; i < numPcs; ++i) {
            const CUpti_PCSamplingPCData& pc =
                pc_sampling_buffers_->data->pPcData[i];
            if (pc.stallReasonCount > 0 && pc.stallReason) {
                for (uint32_t j = 0; j < pc.stallReasonCount; ++j) {
                    if (pc.stallReason[j].samples > 0) {
                        ActivityRecord out{};
                        out.type = TraceType::PC_SAMPLE;
                        if (CUptiResult res =
                                cuptiGetDeviceId(ctx_.cuda_ctx, &out.device_id);
                            res != CUPTI_SUCCESS) {
                            LogCuptiErrorIfFailed(this->name(),
                                                  "cuptiGetDeviceId", res);
                        }
                        out.corr_id = pc.correlationId;
                        out.pc_offset = static_cast<uint32_t>(pc.pcOffset);
                        std::snprintf(out.sample_kind, sizeof(out.sample_kind),
                                      "%s", "pc_sampling");
                        out.samples_count = pc.stallReason[j].samples;
                        out.stall_reason =
                            pc.stallReason[j].pcSamplingStallReasonIndex;
                        out.cpu_start_ns = detail::GetTimestampNs();

                        if (pc.functionName) {
                            std::snprintf(out.function_name,
                                          sizeof(out.function_name), "%s",
                                          pc.functionName);
                            if (std::strlen(pc.functionName) >=
                                sizeof(out.function_name)) {
                                GFL_LOG_DEBUG(
                                    "[PC Sampling] function name truncated in "
                                    "ActivityRecord; using original CUPTI "
                                    "functionName for source correlation "
                                    "(len=", std::strlen(pc.functionName),
                                    ")");
                            }
                        } else {
                            std::snprintf(out.function_name,
                                          sizeof(out.function_name), "unknown");
                        }

                        // Source correlation - grab data pointer under lock,
                        // then call CUPTI outside the lock to avoid deadlock
                        // when CUPTI triggers a module-load callback.
                        const uint8_t* cubinData = nullptr;
                        size_t cubinSize = 0;
                        if (ctx_.cubin_mu && ctx_.cubin_by_crc) {
                            std::lock_guard lk(*ctx_.cubin_mu);
                            auto it = ctx_.cubin_by_crc->find(pc.cubinCrc);
                            if (it != ctx_.cubin_by_crc->end()) {
                                cubinData = it->second.data.data();
                                cubinSize = it->second.data.size();
                            }
                        }
                        if (cubinData && cubinSize > 0 && pc.functionName &&
                            pc.functionName[0] != '\0') {
                            GFL_LOG_DEBUG("start getting source correlation");
                            auto [fileName, dirName, lineNumber] =
                                nvidia::CuptiSass::sampleSourceCorrelation(
                                    cubinData, cubinSize, pc.functionName,
                                    pc.pcOffset);
                            if (!fileName.empty()) {
                                const std::string fullPath =
                                    dirName.empty() ? fileName
                                                    : dirName + "/" + fileName;
                                std::snprintf(out.source_file,
                                              sizeof(out.source_file), "%s",
                                              fullPath.c_str());
                                out.source_line = lineNumber;
                            }
                        }

                        {
                            std::lock_guard lk(stall_reason_mu_);
                            auto it = stall_reason_map_.find(out.stall_reason);
                            if (it != stall_reason_map_.end()) {
                                out.reason_name = it->second;
                            } else {
                                out.reason_name =
                                    "Stall_" + std::to_string(out.stall_reason);
                            }
                        }

                        g_monitorBuffer.Push(out);
                    }
                }
            }
        }

        if (!hasMore) break;
    }

    // Copies, not field refs - see the packed-field note above.
    const uint64_t sumTotal = pc_sampling_buffers_->data->totalSamples;
    const uint64_t sumDropped = pc_sampling_buffers_->data->droppedSamples;
    const uint64_t sumNonUsr =
        pc_sampling_buffers_->data->nonUsrKernelsTotalSamples;
    const uint64_t sumRemaining = pc_sampling_buffers_->data->remainingNumPcs;
    GFL_LOG_DEBUG("[PC Sampling] collect summary: totalSamples=", sumTotal,
                  " dropped=", sumDropped, " nonUsrKernels=", sumNonUsr,
                  " remaining=", sumRemaining);
}

}  // namespace gpufl

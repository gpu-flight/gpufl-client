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
#include <map>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "gpufl/backends/nvidia/cupti_utils.hpp"
#include "gpufl/backends/nvidia/sampler/cupti_sass.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/env_vars.hpp"
#include "gpufl/core/monitor.hpp"
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

// Per-record stall-reason slots of the configured buffer. Sized to the API
// maximum rather than the device's actual stall-reason count so the records
// can be allocated before cuptiPCSamplingEnable - see the ordering note in
// EnableSamplingFeatures_. This is the value NVIDIA's pc_sampling sample
// hardcodes.
constexpr size_t kStallSlots = 128;

// Records per buffer. For the configured buffer this is how many per-kernel
// records CUPTI keeps; kernels past it are merged into per-PC overflow records.
constexpr size_t kMaxPcs = 65536;

// Guard for the GetData loop, which normally ends within a few calls.
constexpr int kMaxGetDataCalls = 1024;

// Rows per Monitor::PushProfileSamples call, bounding the staging vector.
constexpr size_t kRowsPerPush = 8192;

// CUPTI counts each sample under its warp state and, when the scheduler
// issued nothing that cycle, again under the state's _not_issued twin, so
// only the other reasons add up to totalSamples.
bool IsNotIssuedReason(const std::string& name) {
    static const std::string kSuffix = "_not_issued";
    return name.size() > kSuffix.size() &&
           name.compare(name.size() - kSuffix.size(), kSuffix.size(),
                        kSuffix) == 0;
}

PCSamplingBuffers* AllocatePcSamplingBuffers(const size_t numPcs,
                                             const size_t stallSlots) {
    auto* b = new PCSamplingBuffers();
    b->stallSlots = stallSlots;
    b->pcRecords = static_cast<CUpti_PCSamplingPCData*>(
        std::calloc(numPcs, sizeof(CUpti_PCSamplingPCData)));
    for (size_t i = 0; i < numPcs; ++i) {
        b->pcRecords[i].size = sizeof(CUpti_PCSamplingPCData);
        b->pcRecords[i].stallReasonCount = stallSlots;
        b->pcRecords[i].stallReason = static_cast<CUpti_PCSamplingStallReason*>(
            std::calloc(stallSlots, sizeof(CUpti_PCSamplingStallReason)));
    }
    b->data = static_cast<CUpti_PCSamplingData*>(
        std::calloc(1, sizeof(CUpti_PCSamplingData)));
    b->data->size = sizeof(CUpti_PCSamplingData);
    b->data->collectNumPcs = numPcs;
    b->data->pPcData = b->pcRecords;
    b->data->totalNumPcs = 0;
    return b;
}

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
    // the PC sampling lifetime. In this mode CUPTI moves each finished
    // kernel's records into the SAMPLING_DATA_BUFFER below by itself.
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
        // ~1 MB per ~5,500 PCs with all stall reasons, so 32 MB covers ~175k
        // distinct PCs - ample for the single end-of-scope read (the old
        // 256 MB was wildly oversized per context). Sizing this up does not
        // rescue a session that collected nothing: 256 MB was measured to
        // make no difference.
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
    {
        CUpti_PCSamplingConfigurationInfo info = {};
        info.attributeType =
            CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_ENABLE_START_STOP_CONTROL;
        info.attributeData.enableStartStopControlData.enableStartStopControl =
            1;
        addConfig(info);
    }
    // CUPTI writes into this buffer on its own, so GetData must use another
    // one - see CollectPcSamplingData_.
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
        // Arm as early as we can: start() runs in the CONTEXT_CREATED
        // callback, and profiler-init already ran pre-context, so stall
        // enumeration succeeds here.
        //
        // This used to claim Enable/config/Start need a quiet GPU because
        // concurrent kernels make them return INVALID_OPERATION. That is not
        // what happens - measured on driver 610.43 / CUDA 13.3, Start fails
        // the instant after configuration is rejected, before the target has
        // launched anything, and it fails identically whether the arm runs
        // in this callback or on a worker thread hundreds of ms later.
        //
        // WindowOnly splits that: enable + configure now (they must happen
        // while quiet), but leave the sampler unarmed until a deep window
        // opens. Only cuptiPCSamplingStart is deferred, and that one does
        // succeed with kernels running - the mid-run drain-restart below
        // has always relied on exactly that.
        {
            std::lock_guard lk(sampling_lifecycle_mu_);
            if (opts_.deep_arm_mode == DeepArmMode::WindowOnly) {
                EnableSamplingFeatures_();
            } else {
                StartPcSampling_();
            }
        }
        // Enable can internally disable kernel activity — re-assert it.
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
        // Only the experimental kernel-drain mode has anything to do on a
        // cycle; the sample-only path reads once at scope end.
        if (kernel_collect_ == KernelCollect::All) StartCycleThread_();
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
    // Windows-injection process exit: cudart already tore the context down (its
    // atexit runs before ours), so cuptiPCSamplingStop/Disable below crash
    // (0xC0000005) against the dying driver. The cycle thread is joined above
    // and the OS reclaims the sampler at process exit, so skip the fragile CUPTI
    // release - by here the run's data is already flushed + closed.
    if (detail::isProcessExitTeardown()) return;
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
    // Once process-exit teardown begins, cudart is destroying the CUDA context;
    // stop issuing CUPTI data-retrieval calls (GetData/Stop) so the cycle thread
    // can't fault against the dying driver while shutdown flushes + joins it.
    if (detail::isProcessExitTeardown()) return;
    // Only GPUFL_PC_KERNEL_COLLECT=all has cycle work. The sample-only path
    // does not collect mid-run at all - see the note above
    // StopAndCollectPcSampling_ - so the cycle thread is not even started for
    // it, and this is the one path left here.
    if (pc_sampling_method_ != Method::SamplingAPI) return;
    if (!sampling_api_started_.load()) return;

    // GetData requires the context current on the calling thread. Binding
    // is thread-local and this thread is ours, so leave it bound.
    if (!ctx_.cuda_ctx) return;
    if (cuCtxSetCurrent(ctx_.cuda_ctx) != CUDA_SUCCESS) return;

    if (kernel_collect_ == KernelCollect::All &&
        !drain_unavailable_.load(std::memory_order_relaxed)) {
        DrainKernelsAndCollect_();
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

    // Stop sampling first: a forced activity flush returns zero kernel records
    // while PC sampling is armed (driver 590+), and samples are only read with
    // sampling stopped. Stop/Start mid-run are privileged
    // (INSUFFICIENT_PRIVILEGES under a non-elevated run) — on that error, drop
    // to the sample-only cycle, which stops too and therefore fails the same
    // way and stands itself down. Restart after the flush; it succeeds even
    // with kernels running (unlike the initial arm).
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
    // Windows-injection process exit: skip the fragile cuptiPCSamplingDisable
    // (crashes against the context cudart already destroyed). Threads are joined
    // above; the OS reclaims CUPTI state at exit. See stop() for the full note.
    if (gpufl::detail::isProcessExitTeardown()) return;
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
    pc_read_buffers_.reset();
}

void PcSamplingEngine::onScopeStart(const char* /*name*/) {
    // Under WindowOnly this IS the arm: start() only enabled and configured
    // the sampler. Otherwise it's an idempotent re-arm that matters only if
    // a prior attempt failed (e.g. context raced).
    std::lock_guard lk(sampling_lifecycle_mu_);
    StartPcSampling_();
}

void PcSamplingEngine::onScopeStop(const char* /*name*/) {
    // Stop and read. This is the session's collection point in both modes:
    // for WindowOnly it disarms at the window edge (leaving the sampler
    // running past the window is the whole thing WindowOnly exists to avoid),
    // and for the process-wide scope it is the last healthy moment before
    // Windows process-exit teardown breaks cuptiPCSamplingStop with
    // CUPTI_ERROR_UNKNOWN. The ref count keeps a nested scope from collecting
    // early; a later scope re-arms through onScopeStart.
    std::lock_guard lk(sampling_lifecycle_mu_);
    StopAndCollectPcSampling_();
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

    // Allocate the configured buffer BEFORE enabling, and keep Enable and
    // SetConfigurationAttribute adjacent.
    //
    // This ordering is load-bearing, not style. Doing this allocation between
    // Enable and configure is what broke PC sampling under injection:
    // configure (and then every other PC-sampling call on the context) came
    // back CUPTI_ERROR_INVALID_OPERATION. Proven on driver 610.43 / CUDA 13.3
    // by injecting nothing but this allocation into an otherwise-working
    // sequence in the same process, on the same context - Enable returned
    // SUCCESS, the allocation ran, configure returned INVALID_OPERATION. It is
    // the allocation, not elapsed time: a 50 ms sleep in the same slot is
    // harmless. NVIDIA's own pc_sampling sample allocates up front too.
    //
    // Stall-reason enumeration is likewise deferred to after configure; it
    // only feeds the reason-name map and the read buffer's size. The
    // configured records are sized with the API's maximum stall-reason count
    // because the device's count is not known yet, which is what the sample
    // does.
    if (!pc_sampling_buffers_) {
        pc_sampling_buffers_.reset(
            AllocatePcSamplingBuffers(kMaxPcs, kStallSlots));
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

    auto configInfo = BuildPcSamplingConfig(opts_.pc_sampling_period,
                                            pc_sampling_buffers_->data);
    CUpti_PCSamplingConfigurationInfoParams configParams = {};
    configParams.size = CUpti_PCSamplingConfigurationInfoParamsSize;
    configParams.ctx = ctx_.cuda_ctx;
    configParams.numAttributes =
        configInfo.size();
    configParams.pPCSamplingConfigurationInfo = configInfo.data();

    const CUptiResult configRes =
        cuptiPCSamplingSetConfigurationAttribute(&configParams);
    // Any failure here is fatal to the session, INVALID_OPERATION included.
    // This used to be swallowed as benign, which was the single most
    // misleading thing in this file: when configuration is rejected,
    // ENABLE_START_STOP_CONTROL never applies, so cuptiPCSamplingStart then
    // fails with INVALID_OPERATION too - and the log still said "configured
    // and enabled successfully", pointing every investigation at Start.
    if (configRes != CUPTI_SUCCESS) {
        LogCuptiErrorIfFailed(this->name(),
                              "cuptiPCSamplingSetConfigurationAttribute",
                              configRes);
        for (size_t i = 0; i < configInfo.size(); ++i) {
            GFL_LOG_ERROR("[PC Sampling] rejected attribute type=",
                          static_cast<int>(configInfo[i].attributeType),
                          " status=",
                          static_cast<int>(configInfo[i].attributeStatus));
        }
        if (IsInsufficientPrivilege(configRes)) {
            sampling_api_blocked_.store(true);
            GFL_LOG_ERROR(
                "[PC Sampling] Insufficient privileges: disabling PC "
                "sampling for this session.");
        }
        pc_sampling_method_ = Method::None;
        CUpti_PCSamplingDisableParams dp = {};
        dp.size = sizeof(CUpti_PCSamplingDisableParams);
        dp.ctx = ctx_.cuda_ctx;
        cuptiPCSamplingDisable(&dp);
        return false;
    }

    // Stall-reason enumeration, after configuration: it only builds the
    // reason-name map and sizes the read buffer, and keeping it out of the
    // Enable->configure window is the point (see the ordering note above).
    {
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
            return false;
        }
        num_stall_reasons_ = numStallReasons;
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
    // Reached from a CUDA cleanup CUPTI callback, where cuptiPCSamplingStop
    // returns 999 - and samples are only read with sampling stopped. The
    // engine's cycle thread owns collection.
    GFL_LOG_DEBUG(
        "[PC Sampling] skipping collect from CUDA cleanup callback: ",
        reason ? reason : "unknown");
}

void PcSamplingEngine::onLaunchTick() {
    // Deliberately does not collect: this runs inside the launch API_ENTER
    // callback, where Stop is unavailable, and samples are only read with
    // sampling stopped. The cycle thread does the stop/collect/restart.
}

// Sample-only sessions arm once and read once, with sampling stopped, at scope
// end (onScopeStop) or session teardown (stop/shutdown). Reading late loses no
// samples to the configured buffer's capacity - CUPTI merges kernels past it
// into per-PC overflow records - but a long run can still overflow the hardware
// buffer, which the collect summary reports. A mid-run stop -> GetData -> start
// every second stalled the target (driver 610.43 / CUDA 13.3).

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
    // GetData moves the records CUPTI put in the configured buffer, then the
    // per-PC overflow, into this separate buffer. Reading into the configured
    // buffer itself would overwrite everything held there.
    if (!pc_read_buffers_) {
        pc_read_buffers_.reset(AllocatePcSamplingBuffers(
            kMaxPcs, num_stall_reasons_ > 0 ? num_stall_reasons_ : kStallSlots));
    }
    const CUpti_PCSamplingData* const configured = pc_sampling_buffers_->data;
    CUpti_PCSamplingData* const batch = pc_read_buffers_->data;

    CUpti_PCSamplingGetDataParams getDataParams = {};
    getDataParams.size = sizeof(CUpti_PCSamplingGetDataParams);
    getDataParams.ctx = ctx_.cuda_ctx;
    getDataParams.pcSamplingData = batch;

    // One row per (launch, PC, stall reason): CUPTI returns a record set per
    // launch until the configured buffer fills, then merges later launches
    // per PC under one correlation id. Rows go straight to the profile batch
    // because a collect can return far more of them than the monitor ring
    // holds.
    const int64_t collectTs = detail::GetTimestampNs();
    uint32_t deviceId = 0;
    bool deviceIdKnown = false;
    std::unordered_map<uint32_t, std::string> reasonNames;
    {
        std::lock_guard lk(stall_reason_mu_);
        reasonNames = stall_reason_map_;
    }

    // Source correlation depends only on the instruction, so it is resolved
    // once per PC rather than per launch.
    struct PcSource {
        std::string functionKey;  // "function_name@source_file"
        std::string sourceFile;
        uint32_t sourceLine = 0;
    };
    std::map<std::tuple<uint64_t, uint32_t, uint64_t>, PcSource> sourceByPc;
    auto resolveSource = [this](const uint64_t cubinCrc,
                                const char* functionName,
                                const uint64_t pcOffset) {
        PcSource src;
        // Grab the cubin pointer under lock, then call CUPTI outside it to
        // avoid deadlock when CUPTI triggers a module-load callback.
        const uint8_t* cubinData = nullptr;
        size_t cubinSize = 0;
        if (ctx_.cubin_mu && ctx_.cubin_by_crc) {
            std::lock_guard lk(*ctx_.cubin_mu);
            auto it = ctx_.cubin_by_crc->find(cubinCrc);
            if (it != ctx_.cubin_by_crc->end()) {
                cubinData = it->second.data.data();
                cubinSize = it->second.data.size();
            }
        }
        if (cubinData && cubinSize > 0 && functionName &&
            functionName[0] != '\0') {
            auto [fileName, dirName, lineNumber] =
                nvidia::CuptiSass::sampleSourceCorrelation(
                    cubinData, cubinSize, functionName, pcOffset);
            if (!fileName.empty()) {
                src.sourceFile =
                    dirName.empty() ? fileName : dirName + "/" + fileName;
                src.sourceLine = lineNumber;
            }
        }
        src.functionKey = std::string(functionName ? functionName : "unknown") +
                          "@" + src.sourceFile;
        return src;
    };

    std::vector<ProfileSampleInput> rows;
    size_t rowsEmitted = 0;
    uint64_t samplesEmitted = 0;
    uint64_t notIssuedEmitted = 0;
    auto pushRows = [&rows, &rowsEmitted] {
        if (rows.empty()) return;
        Monitor::PushProfileSamples(rows);
        rowsEmitted += rows.size();
        rows.clear();
    };

    uint64_t sumTotal = 0;
    uint64_t sumDropped = 0;
    uint64_t sumNonUsr = 0;
    bool hardwareBufferFull = false;
    int emptyCalls = 0;
    for (int call = 0; call < kMaxGetDataCalls; ++call) {
        // stallReasonCount is written per record; restore each record's
        // capacity before every call.
        for (size_t i = 0; i < batch->collectNumPcs; ++i)
            pc_read_buffers_->pcRecords[i].stallReasonCount =
                pc_read_buffers_->stallSlots;

        batch->totalNumPcs = 0;
        const CUptiResult getRes = cuptiPCSamplingGetData(&getDataParams);
        // OUT_OF_MEMORY reports a full hardware buffer (samples were lost),
        // not more records waiting.
        if (getRes == CUPTI_ERROR_OUT_OF_MEMORY) hardwareBufferFull = true;

        if (getRes != CUPTI_SUCCESS && getRes != CUPTI_ERROR_OUT_OF_MEMORY) {
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

        const size_t numPcs = batch->totalNumPcs;
        if (numPcs > 0) produced_data_.store(true, std::memory_order_relaxed);
        // The hardware-side counters tell zero-record collections apart:
        // totalSamples=0 means the GPU never sampled (period/perms/arming),
        // while totalSamples>0 with numPcs=0 means samples were taken but
        // attributed to non-user kernels or dropped before retrieval. They
        // are per call, so the summary adds them up.
        // Copies, not field refs: CUpti_PCSamplingData is packed and GCC
        // refuses to bind packed fields to the logger's references.
        const uint64_t totalSamples = batch->totalSamples;
        const uint64_t droppedSamples = batch->droppedSamples;
        const uint64_t nonUsrSamples = batch->nonUsrKernelsTotalSamples;
        const size_t remainingPcs = batch->remainingNumPcs;
        sumTotal += totalSamples;
        sumDropped += droppedSamples;
        sumNonUsr += nonUsrSamples;
        GFL_LOG_DEBUG("[PC Sampling] Collected ", numPcs, " PC records (",
                      remainingPcs, " remaining); totalSamples=", totalSamples,
                      " droppedSamples=", droppedSamples,
                      " nonUsrKernelsTotalSamples=", nonUsrSamples);

        uint64_t samplesThisCall = 0;
        uint64_t notIssuedThisCall = 0;
        for (size_t i = 0; i < numPcs; ++i) {
            const CUpti_PCSamplingPCData& pc = batch->pPcData[i];
            if (pc.stallReasonCount == 0 || !pc.stallReason) continue;
            // Copies: packed fields cannot bind to references.
            const uint64_t cubinCrc = pc.cubinCrc;
            const uint32_t functionIndex = pc.functionIndex;
            const uint64_t pcOffset = pc.pcOffset;
            const uint32_t correlationId = pc.correlationId;
            auto [source, inserted] = sourceByPc.try_emplace(
                std::make_tuple(cubinCrc, functionIndex, pcOffset));
            if (inserted) {
                source->second = resolveSource(cubinCrc, pc.functionName,
                                               pcOffset);
            }
            const size_t written = pc.stallReasonCount;
            const size_t reasons = written < pc_read_buffers_->stallSlots
                                       ? written
                                       : pc_read_buffers_->stallSlots;
            for (size_t j = 0; j < reasons; ++j) {
                const uint32_t samples = pc.stallReason[j].samples;
                const uint32_t reason =
                    pc.stallReason[j].pcSamplingStallReasonIndex;
                const auto name = reasonNames.find(reason);
                const bool notIssued = name != reasonNames.end() &&
                                       IsNotIssuedReason(name->second);
                (notIssued ? notIssuedThisCall : samplesThisCall) += samples;
                if (samples == 0) continue;
                if (!deviceIdKnown) {
                    deviceIdKnown = true;
                    if (const CUptiResult res =
                            cuptiGetDeviceId(ctx_.cuda_ctx, &deviceId);
                        res != CUPTI_SUCCESS) {
                        LogCuptiErrorIfFailed(this->name(), "cuptiGetDeviceId",
                                              res);
                    }
                }
                ProfileSampleInput s;
                s.ts_ns = collectTs;
                s.corr_id = correlationId;
                s.device_id = deviceId;
                s.function_key = source->second.functionKey;
                s.pc_offset = static_cast<uint32_t>(pcOffset);
                s.metric_name = name != reasonNames.end()
                                    ? name->second
                                    : "Stall_" + std::to_string(reason);
                s.metric_value = samples;
                s.stall_reason = reason;
                s.sample_kind = 0;  // pc_sampling
                s.source_file = source->second.sourceFile;
                s.source_line = source->second.sourceLine;
                rows.push_back(std::move(s));
                (notIssued ? notIssuedEmitted : samplesEmitted) += samples;
            }
            if (rows.size() >= kRowsPerPush) pushRows();
        }
        pushRows();

        GFL_LOG_DEBUG("[PC Sampling] GetData returned ", samplesThisCall,
                      " samples (+", notIssuedThisCall, " not issued) across ",
                      numPcs, " PC records");
        // Drained once a call returns nothing and CUPTI reports nothing
        // pending. Two empty calls in a row with something still pending end
        // it too, so a stuck report cannot spin.
        const bool pending = configured->totalNumPcs > 0 ||
                             configured->remainingNumPcs > 0 ||
                             remainingPcs > 0;
        if (numPcs > 0) {
            emptyCalls = 0;
        } else if (!pending || ++emptyCalls >= 2) {
            break;
        }
    }

    GFL_LOG_DEBUG("[PC Sampling] collect summary: ", rowsEmitted, " rows, ",
                  samplesEmitted, " samples (+", notIssuedEmitted,
                  " not issued) across ", sourceByPc.size(),
                  " PCs; totalSamples=", sumTotal, " dropped=", sumDropped,
                  " nonUsrKernels=", sumNonUsr,
                  hardwareBufferFull ? " (hardware buffer overflowed)" : "");
}

}  // namespace gpufl

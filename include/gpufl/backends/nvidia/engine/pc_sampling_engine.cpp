#include "gpufl/backends/nvidia/engine/pc_sampling_engine.hpp"

#include <cuda_runtime.h>
#include <cupti.h>
#include <cupti_pcsampling.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "gpufl/backends/nvidia/cupti_utils.hpp"
#include "gpufl/backends/nvidia/sampler/cupti_sass.hpp"
#include "gpufl/core/activity_record.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/ring_buffer.hpp"

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
        info.attributeData.scratchBufferSizeData.scratchBufferSize =
            256 * 1024 * 1024;
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
    GFL_LOG_DEBUG("[PcSamplingEngine] initialized");
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
        GFL_LOG_DEBUG("[PC Sampling] Enabled SOURCE_LOCATOR for Sampling API.");
        // Build-cache sanity log: prints the exact pc_sampling_period in
        // effect for this session. Without this, a stale wheel/.pyd from
        // a previous build silently uses the old default and any tuning
        // experiment is invalid. Period is a log2 exponent — expand it
        // so users don't have to do the mental math at debug time.
        GFL_LOG_DEBUG(
            "[PC Sampling] samplingPeriod=", opts_.pc_sampling_period,
            " (2^", opts_.pc_sampling_period, " = ",
            (1u << opts_.pc_sampling_period), " GPU cycles between samples)");
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

void PcSamplingEngine::stop() {
    // Disable the SamplingAPI session before the activity flush in
    // CuptiBackend::stop().  While the SamplingAPI is armed, any CUPTI
    // data-retrieval call (FlushAll, GetData, Disable) can permanently
    // kill the subscriber callback on driver 590+.  Disabling here — in
    // the correct order (before flush) — ensures cuptiActivityFlushAll
    // delivers real kernel activity records with correct GPU durations.
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
    // SamplingAPI data is collected only after cuptiPCSamplingStop().  Calling
    // cuptiPCSamplingGetData() from the periodic collector while kernels are
    // still inside a scope can drain CUPTI's PC buffer and make the final
    // stop-side collection report zero records.
}

void PcSamplingEngine::shutdown() {
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
    StartPcSampling_();
}

void PcSamplingEngine::onScopeStop(const char* /*name*/) {
    // Keep SamplingAPI armed across scopes; collect once at session stop.
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
            "INVALID_OPERATION — possibly already enabled or "
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

        if (cuptiPCSamplingGetNumStallReasons(&numParams) == CUPTI_SUCCESS &&
            numStallReasons > 0) {
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

    // cuptiPCSamplingEnable / SetConfigurationAttribute can internally
    // disable kernel activity tracing. Re-enable CONCURRENT_KERNEL so kernel
    // records keep flowing alongside PC samples. We deliberately do NOT
    // re-enable CUPTI_ACTIVITY_KIND_KERNEL — it serializes every kernel
    // (distorts timing, adds overhead), and PC sampling is compatible with
    // concurrent kernel tracing since CUDA 12.8 Update 1.
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);

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
        LogCuptiErrorIfFailed(this->name(), "cuptiPCSamplingStart", startRes);
        return;
    }

    sampling_api_started_.store(true);
    GFL_LOG_DEBUG("[PC Sampling] >>> STARTED (Scope Begin) <<<");
}

void PcSamplingEngine::StopAndCollectPcSampling_() {
    GFL_LOG_DEBUG("[PC Sampling] StopAndCollect entry: method=",
                  static_cast<int>(pc_sampling_method_),
                  " refCount=", pc_sampling_ref_count_.load(),
                  " started=", sampling_api_started_.load());
    if (pc_sampling_method_ != Method::SamplingAPI) {
        GFL_LOG_DEBUG("[PC Sampling] StopAndCollect: exit — method != SamplingAPI");
        return;
    }

    const int refs = pc_sampling_ref_count_.load();
    if (refs <= 0) {
        GFL_LOG_DEBUG("[PC Sampling] StopAndCollect: exit — no active scope");
        return;
    }
    if (refs > 1) {
        const int remaining = pc_sampling_ref_count_.fetch_sub(1) - 1;
        GFL_LOG_DEBUG("[PC Sampling] still active (RefCount=", remaining, ")");
        return;
    }
    pc_sampling_ref_count_.store(0);

    if (!sampling_api_started_.exchange(false)) {
        GFL_LOG_DEBUG("[PC Sampling] StopAndCollect: exit — already collected / not started");
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
    GFL_LOG_DEBUG("[PC Sampling] <<< COLLECTING (Session End) >>> collectNumPcs=",
                  collectNumPcs);
    cudaDeviceSynchronize();

    CUpti_PCSamplingStopParams stopParams = {};
    stopParams.size = sizeof(CUpti_PCSamplingStopParams);
    stopParams.ctx = ctx_.cuda_ctx;
    const CUptiResult stopRes = cuptiPCSamplingStop(&stopParams);
    if (stopRes != CUPTI_SUCCESS) {
        if (IsInsufficientPrivilege(stopRes) ||
            stopRes == CUPTI_ERROR_NOT_INITIALIZED) {
            sampling_api_blocked_.store(true);
        }
        LogCuptiErrorIfFailed(this->name(), "cuptiPCSamplingStop", stopRes);
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
                // called before PC Sampling API — they are mutually exclusive
                // on Turing+ GPUs.  Disable to suppress repeated errors.
                GFL_LOG_DEBUG("[PC Sampling] getData failed (", getRes,
                              ") — "
                              "disabling PC sampling for this session.");
                sampling_api_blocked_.store(true);
            } else {
                LogCuptiErrorIfFailed(this->name(), "cuptiPCSamplingGetData",
                                      getRes);
            }
            break;
        }

        auto numPcs = pc_sampling_buffers_->data->totalNumPcs;
        if (numPcs > 0) produced_data_.store(true, std::memory_order_relaxed);
        GFL_LOG_DEBUG("[PC Sampling] Collected ", numPcs, " PC records",
                      (hasMore ? " (more remaining)." : "."));

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
                        } else {
                            std::snprintf(out.function_name,
                                          sizeof(out.function_name), "unknown");
                        }

                        // Source correlation — grab data pointer under lock,
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
                        if (cubinData) {
                            GFL_LOG_DEBUG("start getting source correlation");
                            auto [fileName, dirName, lineNumber] =
                                nvidia::CuptiSass::sampleSourceCorrelation(
                                    cubinData, cubinSize, out.function_name,
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
}

}  // namespace gpufl

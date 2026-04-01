#include "gpufl/backends/nvidia/engine/pc_sampling_engine.hpp"

#include <cupti.h>
#include <cupti_pcsampling.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

extern RingBuffer<ActivityRecord, 1024> g_monitorBuffer;

namespace {
bool IsInsufficientPrivilege(CUptiResult res) {
    if (res == CUPTI_ERROR_INSUFFICIENT_PRIVILEGES) return true;
#ifdef CUPTI_ERROR_VIRTUALIZED_DEVICE_INSUFFICIENT_PRIVILEGES
    if (res == CUPTI_ERROR_VIRTUALIZED_DEVICE_INSUFFICIENT_PRIVILEGES)
        return true;
#endif
    return false;
}
}  // namespace

// ---- PCSamplingDeleter -----------------------------------------------------

void PCSamplingDeleter::operator()(PCSamplingBuffers* b) const {
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
    // Nothing additional to disable — activity disable is handled by
    // CuptiBackend::stop() for all registered kinds.
}

void PcSamplingEngine::shutdown() {
    if (sampling_api_ready_.load() && ctx_.cuda_ctx) {
        CUpti_PCSamplingDisableParams disableParams = {};
        disableParams.size = sizeof(CUpti_PCSamplingDisableParams);
        disableParams.ctx = ctx_.cuda_ctx;
        CUptiResult disableRes = cuptiPCSamplingDisable(&disableParams);
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

    CUpti_PCSamplingEnableParams enableParams = {};
    enableParams.size = sizeof(CUpti_PCSamplingEnableParams);
    enableParams.ctx = ctx_.cuda_ctx;
    CUptiResult enableRes = cuptiPCSamplingEnable(&enableParams);
    if (enableRes != CUPTI_SUCCESS &&
        enableRes != CUPTI_ERROR_INVALID_OPERATION) {
        LogCuptiErrorIfFailed(this->name(), "cuptiPCSamplingEnable",
                              enableRes);
        if (IsInsufficientPrivilege(enableRes)) {
            sampling_api_blocked_.store(true);
            pc_sampling_method_ = Method::None;
            GFL_LOG_ERROR(
                "[PC Sampling] Insufficient privileges: disabling PC "
                "sampling for this session.");
        }
        return false;
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
                std::lock_guard<std::mutex> lk(stall_reason_mu_);
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
    }

    CUpti_PCSamplingConfigurationInfo configInfo[7] = {};
    configInfo[0].attributeType =
        CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_COLLECTION_MODE;
    // KERNEL_SERIALIZED and SASS lazy patching both intercept cuLaunchKernel
    // at the driver level and deadlock on the first kernel launch. Use
    // CONTINUOUS mode when SASS metrics are also active so that the lazy
    // patching pass can complete before PC samples are drained.
    configInfo[0].attributeData.collectionModeData.collectionMode =
        (opts_.profiling_engine == ProfilingEngine::PcSamplingWithSass)
            ? CUPTI_PC_SAMPLING_COLLECTION_MODE_CONTINUOUS
            : CUPTI_PC_SAMPLING_COLLECTION_MODE_KERNEL_SERIALIZED;

    configInfo[1].attributeType =
        CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_PERIOD;
    configInfo[1].attributeData.samplingPeriodData.samplingPeriod =
        opts_.pc_sampling_period;

    configInfo[2].attributeType =
        CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SCRATCH_BUFFER_SIZE;
    configInfo[2].attributeData.scratchBufferSizeData.scratchBufferSize =
        256 * 1024 * 1024;

    configInfo[3].attributeType =
        CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_HARDWARE_BUFFER_SIZE;
    configInfo[3].attributeData.hardwareBufferSizeData.hardwareBufferSize =
        256 * 1024 * 1024;

    configInfo[4].attributeType =
        CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_ENABLE_START_STOP_CONTROL;
    configInfo[4]
        .attributeData.enableStartStopControlData.enableStartStopControl = 1;

    configInfo[5].attributeType =
        CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_DATA_BUFFER;
    configInfo[5].attributeData.samplingDataBufferData.samplingDataBuffer =
        pc_sampling_buffers_->data;

    configInfo[6].attributeType =
        CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_OUTPUT_DATA_FORMAT;
    configInfo[6].attributeData.outputDataFormatData.outputDataFormat =
        CUPTI_PC_SAMPLING_OUTPUT_DATA_FORMAT_PARSED;

    CUpti_PCSamplingConfigurationInfoParams configParams = {};
    configParams.size = CUpti_PCSamplingConfigurationInfoParamsSize;
    configParams.ctx = ctx_.cuda_ctx;
    configParams.numAttributes = 7;
    configParams.pPCSamplingConfigurationInfo = configInfo;

    CUptiResult configRes =
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

    int expected = 0;
    if (!pc_sampling_ref_count_.compare_exchange_strong(expected, 1)) {
        const int refs = pc_sampling_ref_count_.fetch_add(1) + 1;
        GFL_LOG_DEBUG("[PC Sampling] already active (RefCount=",
                      refs, ")");
        return;
    }

    if (!EnableSamplingFeatures_()) {
        pc_sampling_ref_count_.store(0);
        return;
    }

    if (!ctx_.cuda_ctx) {
        GFL_LOG_ERROR("[GPUFL] Cannot start PC Sampling: cuda_ctx is NULL!");
        pc_sampling_ref_count_.store(0);
        return;
    }

    GFL_LOG_DEBUG("[PC Sampling] Starting with ctx=",
                  static_cast<void*>(ctx_.cuda_ctx));

    CUpti_PCSamplingStartParams startParams = {};
    startParams.size = sizeof(CUpti_PCSamplingStartParams);
    startParams.ctx = ctx_.cuda_ctx;
    CUptiResult res = cuptiPCSamplingStart(&startParams);
    if (res == CUPTI_ERROR_INVALID_OPERATION) {
        GFL_LOG_DEBUG("[GPUFL] PC Sampling already active (Implicit Start).");
        sampling_api_started_.store(true);
    } else if (res == CUPTI_ERROR_NOT_SUPPORTED ||
               res == CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED) {
        GFL_LOG_DEBUG(
            "[GPUFL] PC Sampling not supported on this GPU/configuration.");
        sampling_api_ready_.store(false);
        sampling_api_started_.store(false);
        pc_sampling_ref_count_.store(0);
        pc_sampling_method_ = Method::None;
    } else if (res != CUPTI_SUCCESS) {
        LogCuptiErrorIfFailed(this->name(), "cuptiPCSamplingStart", res);
        if (IsInsufficientPrivilege(res)) {
            sampling_api_blocked_.store(true);
            pc_sampling_method_ = Method::None;
            GFL_LOG_ERROR(
                "[PC Sampling] Insufficient privileges: disabling PC "
                "sampling for this session.");
        }
        sampling_api_ready_.store(false);
        sampling_api_started_.store(false);
        pc_sampling_ref_count_.store(0);
    } else {
        sampling_api_started_.store(true);
        GFL_LOG_DEBUG("[PC Sampling] >>> STARTED (Scope Begin) <<<");
    }
}

void PcSamplingEngine::StopAndCollectPcSampling_() {
    if (pc_sampling_method_ != Method::SamplingAPI) return;

    if (pc_sampling_ref_count_.load() <= 0) return;

    int refs = pc_sampling_ref_count_.fetch_sub(1);
    if (refs > 1) {
        GFL_LOG_DEBUG("[PC Sampling] still active (RefCount=", refs - 1, ")");
        return;
    } else if (refs < 1) {
        GFL_LOG_ERROR("[PC Sampling] RefCount underflow!");
        pc_sampling_ref_count_.store(0);
        return;
    }

    if (!sampling_api_started_.exchange(false)) return;

    if (!ctx_.cuda_ctx || !IsContextValid(ctx_.cuda_ctx)) {
        GFL_LOG_ERROR("[GPUFL] Aborting PC Sampling: Context invalid.");
        return;
    }

    if (!pc_sampling_buffers_ || !pc_sampling_buffers_->data) {
        GFL_LOG_ERROR("[GPUFL] No PC sampling buffers allocated!");
        return;
    }

    GFL_LOG_DEBUG("[PC Sampling] <<< STOPPING (Scope End) >>>");

    cudaDeviceSynchronize();

    CUpti_PCSamplingStopParams stopParams = {};
    stopParams.size = sizeof(CUpti_PCSamplingStopParams);
    stopParams.ctx = ctx_.cuda_ctx;
    CUptiResult stopRes = cuptiPCSamplingStop(&stopParams);
    if (stopRes != CUPTI_SUCCESS) {
        if (stopRes != CUPTI_ERROR_INVALID_OPERATION) {
            LogCuptiErrorIfFailed(this->name(), "cuptiPCSamplingStop",
                                  stopRes);
        }
        if (IsInsufficientPrivilege(stopRes)) {
            sampling_api_blocked_.store(true);
            pc_sampling_method_ = Method::None;
        }
        return;
    }

    CUpti_PCSamplingGetDataParams getDataParams = {};
    getDataParams.size = sizeof(CUpti_PCSamplingGetDataParams);
    getDataParams.ctx = ctx_.cuda_ctx;
    getDataParams.pcSamplingData = pc_sampling_buffers_->data;

    while (true) {
        pc_sampling_buffers_->data->totalNumPcs = 0;
        CUptiResult getRes = cuptiPCSamplingGetData(&getDataParams);
        const bool hasMore = (getRes == CUPTI_ERROR_OUT_OF_MEMORY);

        if (getRes != CUPTI_SUCCESS && !hasMore) {
            LogCuptiErrorIfFailed(this->name(), "cuptiPCSamplingGetData",
                                  getRes);
            break;
        }

        auto numPcs = pc_sampling_buffers_->data->totalNumPcs;
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
                            std::lock_guard<std::mutex> lk(*ctx_.cubin_mu);
                            auto it = ctx_.cubin_by_crc->find(pc.cubinCrc);
                            if (it != ctx_.cubin_by_crc->end()) {
                                cubinData = it->second.data.data();
                                cubinSize = it->second.data.size();
                            }
                        }
                        if (cubinData) {
                            GFL_LOG_DEBUG("start getting source correlation");
                            auto sourceCorr =
                                nvidia::CuptiSass::sampleSourceCorrelation(
                                    cubinData, cubinSize,
                                    out.function_name, pc.pcOffset);
                            if (!sourceCorr.fileName.empty()) {
                                const std::string fullPath =
                                    sourceCorr.dirName.empty()
                                        ? sourceCorr.fileName
                                        : sourceCorr.dirName + "/" +
                                              sourceCorr.fileName;
                                std::snprintf(out.source_file,
                                              sizeof(out.source_file), "%s",
                                              fullPath.c_str());
                                out.source_line = sourceCorr.lineNumber;
                            }
                        }

                        {
                            std::lock_guard<std::mutex> lk(stall_reason_mu_);
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

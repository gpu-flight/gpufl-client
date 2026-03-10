#include "gpufl/backends/nvidia/engine/pc_sampling_engine.hpp"

#include <cupti.h>
#include <cupti_pcsampling.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "gpufl/backends/nvidia/cupti_utils.hpp"
#include "gpufl/backends/nvidia/sampler/cupti_sass.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/ring_buffer.hpp"

#ifndef CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SOURCE_REPORTING
#define CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SOURCE_REPORTING \
    ((CUpti_PCSamplingConfigurationAttributeType)10)
#endif

namespace gpufl {

extern RingBuffer<ActivityRecord, 1024> g_monitorBuffer;

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
    ctx_  = ctx;
    GFL_LOG_DEBUG("[PcSamplingEngine] initialized");
    return true;
}

void PcSamplingEngine::start() {
    CUptiResult pcRes =
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_PC_SAMPLING);

    if (pcRes == CUPTI_SUCCESS) {
        pc_sampling_method_ = Method::ActivityAPI;
        GFL_LOG_DEBUG("[PC Sampling] Using Activity API (CUPTI_ACTIVITY_KIND_PC_SAMPLING)");
    } else if (pcRes == CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED) {
        GFL_LOG_DEBUG("[PC Sampling] Activity API not supported, using PC Sampling API...");
        pc_sampling_method_ = Method::SamplingAPI;
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR);
        GFL_LOG_DEBUG("[PC Sampling] Enabled SOURCE_LOCATOR for Sampling API.");
    } else {
        LogCuptiErrorIfFailed(this->name(),
                              "cuptiActivityEnable(PC_SAMPLING)", pcRes);
        pc_sampling_method_ = Method::None;
    }
}

void PcSamplingEngine::stop() {
    // Nothing additional to disable — activity disable is handled by
    // CuptiBackend::stop() for all registered kinds.
}

void PcSamplingEngine::shutdown() {
    pc_sampling_buffers_.reset();
}

void PcSamplingEngine::onScopeStart(const char* /*name*/) {
    StartPcSampling_();
}

void PcSamplingEngine::onScopeStop(const char* /*name*/) {
    StopAndCollectPcSampling_();
}

// ---- Private helpers -------------------------------------------------------

void PcSamplingEngine::EnableSamplingFeatures_() {
    if (pc_sampling_method_ == Method::None) return;

    GFL_LOG_DEBUG("[PcSamplingEngine] Configuring PC Sampling...");

    if (!ctx_.cuda_ctx) {
        GFL_LOG_ERROR("[GPUFL] Cannot configure PC Sampling: cuda_ctx is NULL!");
        return;
    }

    CUpti_PCSamplingEnableParams enableParams = {};
    enableParams.size = sizeof(CUpti_PCSamplingEnableParams);
    enableParams.ctx  = ctx_.cuda_ctx;
    CUptiResult enableRes = cuptiPCSamplingEnable(&enableParams);
    if (LogCuptiErrorIfFailed(this->name(), "cuptiPCSamplingEnable", enableRes)) {
        return;
    }

    if (!pc_sampling_buffers_) {
        constexpr size_t kMaxPcs = 65536;
        pc_sampling_buffers_ =
            std::unique_ptr<PCSamplingBuffers, PCSamplingDeleter>(
                new PCSamplingBuffers());
        pc_sampling_buffers_->pcRecords =
            static_cast<CUpti_PCSamplingPCData*>(
                std::calloc(kMaxPcs, sizeof(CUpti_PCSamplingPCData)));

        CUpti_PCSamplingGetNumStallReasonsParams numParams = {};
        numParams.size           = sizeof(CUpti_PCSamplingGetNumStallReasonsParams);
        numParams.ctx            = ctx_.cuda_ctx;
        size_t numStallReasons   = 0;
        numParams.numStallReasons = &numStallReasons;

        if (cuptiPCSamplingGetNumStallReasons(&numParams) == CUPTI_SUCCESS &&
            numStallReasons > 0) {
            auto* stallIndices =
                static_cast<uint32_t*>(malloc(numStallReasons * sizeof(uint32_t)));
            char** stallReasonNames =
                static_cast<char**>(malloc(numStallReasons * sizeof(char*)));
            for (size_t i = 0; i < numStallReasons; i++) {
                stallReasonNames[i] =
                    static_cast<char*>(malloc(CUPTI_STALL_REASON_STRING_SIZE));
            }

            CUpti_PCSamplingGetStallReasonsParams getParams = {
                sizeof(CUpti_PCSamplingGetStallReasonsParams)};
            getParams.ctx              = ctx_.cuda_ctx;
            getParams.pPriv            = nullptr;
            getParams.numStallReasons  = numStallReasons;
            getParams.stallReasonIndex = stallIndices;
            getParams.stallReasons     = stallReasonNames;

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
                GFL_LOG_ERROR("[PcSamplingEngine] cuptiPCSamplingGetStallReasons failed: ", res);
            }
            free(stallIndices);
            free(stallReasonNames);
        }

        for (size_t i = 0; i < kMaxPcs; ++i) {
            pc_sampling_buffers_->pcRecords[i].size = sizeof(CUpti_PCSamplingPCData);
            pc_sampling_buffers_->pcRecords[i].stallReasonCount = numStallReasons;
            pc_sampling_buffers_->pcRecords[i].stallReason =
                static_cast<CUpti_PCSamplingStallReason*>(
                    std::calloc(numStallReasons,
                                sizeof(CUpti_PCSamplingStallReason)));
        }
        pc_sampling_buffers_->data =
            static_cast<CUpti_PCSamplingData*>(
                std::calloc(1, sizeof(CUpti_PCSamplingData)));
        pc_sampling_buffers_->data->size          = sizeof(CUpti_PCSamplingData);
        pc_sampling_buffers_->data->collectNumPcs = kMaxPcs;
        pc_sampling_buffers_->data->pPcData       = pc_sampling_buffers_->pcRecords;
        pc_sampling_buffers_->data->totalNumPcs   = 0;
    }

    CUpti_PCSamplingConfigurationInfo configInfo[7] = {};
    configInfo[0].attributeType =
        CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_COLLECTION_MODE;
    configInfo[0].attributeData.collectionModeData.collectionMode =
        CUPTI_PC_SAMPLING_COLLECTION_MODE_KERNEL_SERIALIZED;

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
    configInfo[4].attributeData.enableStartStopControlData
        .enableStartStopControl = 1;

    configInfo[5].attributeType =
        CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_DATA_BUFFER;
    configInfo[5].attributeData.samplingDataBufferData.samplingDataBuffer =
        pc_sampling_buffers_->data;

    configInfo[6].attributeType =
        CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_OUTPUT_DATA_FORMAT;
    configInfo[6].attributeData.outputDataFormatData.outputDataFormat =
        CUPTI_PC_SAMPLING_OUTPUT_DATA_FORMAT_PARSED;

    CUpti_PCSamplingConfigurationInfoParams configParams = {};
    configParams.size           = CUpti_PCSamplingConfigurationInfoParamsSize;
    configParams.ctx            = ctx_.cuda_ctx;
    configParams.numAttributes  = 7;
    configParams.pPCSamplingConfigurationInfo = configInfo;

    CUptiResult configRes =
        cuptiPCSamplingSetConfigurationAttribute(&configParams);
    if (!LogCuptiErrorIfFailed(this->name(),
                               "cuptiPCSamplingSetConfigurationAttribute",
                               configRes)) {
        GFL_LOG_DEBUG("[PC Sampling] configured and enabled successfully.");
    }
}

void PcSamplingEngine::StartPcSampling_() {
    if (pc_sampling_method_ != Method::SamplingAPI) return;

    if (pc_sampling_ref_count_.fetch_add(1) > 0) {
        GFL_LOG_DEBUG("[PC Sampling] already active (RefCount=",
                      pc_sampling_ref_count_.load(), ")");
        return;
    }

    EnableSamplingFeatures_();

    if (!ctx_.cuda_ctx) {
        GFL_LOG_ERROR("[GPUFL] Cannot start PC Sampling: cuda_ctx is NULL!");
        return;
    }

    GFL_LOG_DEBUG("[PC Sampling] Starting with ctx=",
                  static_cast<void*>(ctx_.cuda_ctx));

    CUpti_PCSamplingStartParams startParams = {};
    startParams.size = sizeof(CUpti_PCSamplingStartParams);
    startParams.ctx  = ctx_.cuda_ctx;
    CUptiResult res = cuptiPCSamplingStart(&startParams);
    if (res == CUPTI_ERROR_INVALID_OPERATION) {
        GFL_LOG_DEBUG("[GPUFL] PC Sampling already active (Implicit Start).");
    } else if (res == CUPTI_ERROR_NOT_SUPPORTED ||
               res == CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED) {
        GFL_LOG_DEBUG(
            "[GPUFL] PC Sampling not supported on this GPU/configuration.");
    } else if (res != CUPTI_SUCCESS) {
        LogCuptiErrorIfFailed(this->name(), "cuptiPCSamplingStart", res);
    } else {
        GFL_LOG_DEBUG("[PC Sampling] >>> STARTED (Scope Begin) <<<");
    }
}

void PcSamplingEngine::StopAndCollectPcSampling_() {
    if (pc_sampling_method_ != Method::SamplingAPI) return;

    int refs = pc_sampling_ref_count_.fetch_sub(1);
    if (refs > 1) {
        GFL_LOG_DEBUG("[PC Sampling] still active (RefCount=", refs - 1, ")");
        return;
    } else if (refs < 1) {
        GFL_LOG_ERROR("[PC Sampling] RefCount underflow!");
        pc_sampling_ref_count_.store(0);
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

    GFL_LOG_DEBUG("[PC Sampling] <<< STOPPING (Scope End) >>>");

    cudaDeviceSynchronize();

    CUpti_PCSamplingStopParams stopParams = {};
    stopParams.size = sizeof(CUpti_PCSamplingStopParams);
    stopParams.ctx  = ctx_.cuda_ctx;
    cuptiPCSamplingStop(&stopParams);

    CUpti_PCSamplingGetDataParams getDataParams = {};
    getDataParams.size           = sizeof(CUpti_PCSamplingGetDataParams);
    getDataParams.ctx            = ctx_.cuda_ctx;
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
                        if (CUptiResult res = cuptiGetDeviceId(
                                ctx_.cuda_ctx, &out.device_id);
                            res != CUPTI_SUCCESS) {
                            LogCuptiErrorIfFailed(this->name(), "cuptiGetDeviceId",
                                                  res);
                        }
                        out.corr_id      = pc.correlationId;
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

                        // Source correlation via cubin map
                        if (ctx_.cubin_mu && ctx_.cubin_by_crc) {
                            std::lock_guard<std::mutex> lk(*ctx_.cubin_mu);
                            auto it = ctx_.cubin_by_crc->find(pc.cubinCrc);
                            if (it != ctx_.cubin_by_crc->end()) {
                                GFL_LOG_DEBUG("start getting source correlation");
                                auto sourceCorr =
                                    nvidia::CuptiSass::sampleSourceCorrelation(
                                        it->second.data.data(),
                                        it->second.data.size(),
                                        out.function_name, pc.pcOffset);
                                if (!sourceCorr.fileName.empty()) {
                                    std::snprintf(out.source_file,
                                                  sizeof(out.source_file), "%s",
                                                  sourceCorr.fileName.c_str());
                                    out.source_line = sourceCorr.lineNumber;
                                }
                            }
                        }

                        {
                            std::lock_guard<std::mutex> lk(stall_reason_mu_);
                            auto it =
                                stall_reason_map_.find(out.stall_reason);
                            if (it != stall_reason_map_.end()) {
                                out.reason_name = it->second;
                            } else {
                                out.reason_name =
                                    "Stall_" +
                                    std::to_string(out.stall_reason);
                            }
                        }

                        g_monitorBuffer.Push(out);
                    }
                }
            }
        }

        if (!hasMore) break;
    }

    CUpti_PCSamplingDisableParams disableParams = {};
    disableParams.size = sizeof(CUpti_PCSamplingDisableParams);
    disableParams.ctx  = ctx_.cuda_ctx;
    cuptiPCSamplingDisable(&disableParams);
}

}  // namespace gpufl

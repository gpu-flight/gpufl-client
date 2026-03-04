#include "gpufl/backends/nvidia/cupti_backend.hpp"

#include <cupti_pcsampling.h>
#include <cupti_profiler_target.h>
#include <cupti_sass_metrics.h>
#include <cupti_target.h>

#include <cstring>
#include <set>

#include "gpufl/backends/nvidia/cupti_utils.hpp"
#include "gpufl/backends/nvidia/kernel_launch_handler.hpp"
#include "gpufl/backends/nvidia/mem_transfer_handler.hpp"
#include "gpufl/backends/nvidia/resource_handler.hpp"
#include "gpufl/backends/nvidia/sampler/cupti_sass.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/ring_buffer.hpp"
#include "gpufl/core/trace_type.hpp"

#ifndef CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SOURCE_REPORTING
#define CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SOURCE_REPORTING \
    ((CUpti_PCSamplingConfigurationAttributeType)10)
#endif

#include "gpufl/backends/nvidia/cuda_collector.hpp"
#include "gpufl/core/scope_registry.hpp"
#include "gpufl/core/stack_registry.hpp"
#include "gpufl/core/stack_trace.hpp"
#include "sampler/cupti_sass.hpp"

namespace gpufl {
std::atomic<gpufl::CuptiBackend*> g_activeBackend{nullptr};

// External ring buffer (defined in monitor.cpp)
extern RingBuffer<ActivityRecord, 1024> g_monitorBuffer;

void CuptiBackend::initialize(const MonitorOptions& opts) {
    opts_ = opts;
    if (opts_.is_profiling) {
        mode_ = static_cast<MonitorMode>(
            static_cast<int>(mode_) | static_cast<int>(MonitorMode::Profiling));
    }

    DebugLogger::setEnabled(opts_.enable_debug_output);

    g_activeBackend.store(this, std::memory_order_release);

    // Internal handler registration
    RegisterHandler(std::make_shared<ResourceHandler>(this));
    RegisterHandler(std::make_shared<KernelLaunchHandler>(this));
    RegisterHandler(std::make_shared<MemTransferHandler>(this));

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
    {
        std::lock_guard<std::mutex> lk(handler_mu_);
        for (const auto& h : handlers_) {
            for (auto d : h->requiredDomains()) domains.insert(d);
            for (auto cb : h->requiredCallbacks()) callbacks.insert(cb);
        }
    }
    for (auto d : domains) CUPTI_CHECK(cuptiEnableDomain(1, subscriber_, d));
    for (auto& [domain, cbid] : callbacks)
        CUPTI_CHECK(cuptiEnableCallback(1, subscriber_, domain, cbid));

    CUptiResult resCb =
        cuptiActivityRegisterCallbacks(BufferRequested, BufferCompleted);
    if (resCb != CUPTI_SUCCESS) {
        const char* errStr = nullptr;
        cuptiGetResultString(resCb, &errStr);
        GFL_LOG_ERROR("FATAL: Failed to register activity callbacks.");
        GFL_LOG_ERROR("Error: ", (errStr ? errStr : "unknown"), " (Code ",
                      resCb, ")");

        initialized_ = false;
        return;
    }

    initialized_ = true;
    GFL_LOG_DEBUG("Callbacks registered successfully.");
}

void CuptiBackend::shutdown() {
    if (!initialized_) return;

    cuptiActivityFlushAll(1);

    if (sass_metrics_buffers_) {
        if (sass_metrics_buffers_->config) {
            std::free(sass_metrics_buffers_->config);
        }
        if (sass_metrics_buffers_->data) {
            std::free(sass_metrics_buffers_->data);
        }
        delete sass_metrics_buffers_;
        sass_metrics_buffers_ = nullptr;
    }
    pc_sampling_buffers_.reset();
    {
        std::lock_guard<std::mutex> lk(handler_mu_);
        std::set<CUpti_CallbackDomain> domains;
        for (const auto& h : handlers_)
            for (auto d : h->requiredDomains()) domains.insert(d);
        for (auto d : domains) cuptiEnableDomain(0, subscriber_, d);
    }

    cuptiUnsubscribe(subscriber_);
    g_activeBackend.store(nullptr, std::memory_order_release);
    initialized_ = false;
}

CUptiResult (*CuptiBackend::get_value())(CUpti_ActivityKind) {
    return cuptiActivityEnable;
}

void CuptiBackend::start() {
    if (!initialized_) return;

    CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR));

    if (IsProfilingMode()) {
        // Ensure context is initialized
        if (!this->ctx_) {
            cuCtxGetCurrent(&this->ctx_);
            if (!this->ctx_) {
                cudaFree(nullptr);  // Force init
                cuCtxGetCurrent(&this->ctx_);
            }
        }

        if (this->ctx_) {
            EnableSassMetrics();
        }
    }

    if (IsMonitoringMode()) {
        std::set<CUpti_ActivityKind> kinds;
        {
            std::lock_guard<std::mutex> lk(handler_mu_);
            for (const auto& h : handlers_)
                for (auto k : h->requiredActivityKinds()) kinds.insert(k);
        }
        for (auto k : kinds) CUPTI_CHECK(cuptiActivityEnable(k));
    }

    if (IsProfilingMode()) {
        // STRATEGY 1: Try Activity API first (works on older GPUs)
        CUptiResult pcRes =
            cuptiActivityEnable(CUPTI_ACTIVITY_KIND_PC_SAMPLING);
        if (pcRes == CUPTI_SUCCESS) {
            pc_sampling_method_ = PCSamplingMethod::ActivityAPI;
            GFL_LOG_DEBUG(
                "[PC Sampling] Using Activity API "
                "(CUPTI_ACTIVITY_KIND_PC_SAMPLING)");
        } else if (pcRes == CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED) {
            // STRATEGY 2: Fallback to PC Sampling API (works on newer GPUs like
            // RTX 40xx/50xx)
            GFL_LOG_DEBUG(
                "[PC Sampling] Activity API not supported, trying PC Sampling "
                "API...");
            pc_sampling_method_ = PCSamplingMethod::SamplingAPI;
            GFL_LOG_DEBUG(
                "[PC Sampling] Using PC Sampling API (cuptiPCSampling*)");
            cuptiActivityEnable(CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR);
            GFL_LOG_DEBUG(
                "[PC Sampling] Enabled SOURCE_LOCATOR activity kind for "
                "Sampling API.");
        } else {
            const char* err;
            cuptiGetResultString(pcRes, &err);
            GFL_LOG_ERROR("[PC Sampling] Failed to enable: ", err);
            pc_sampling_method_ = PCSamplingMethod::None;
        }
    }

    active_.store(true);
    GFL_LOG_DEBUG("Backend started. Mode bitmask: ", static_cast<int>(mode_));
}

bool CuptiBackend::IsMonitoringMode() {
    return hasFlag(mode_, MonitorMode::Monitoring) ||
           hasFlag(mode_, MonitorMode::Profiling);
}

bool CuptiBackend::IsProfilingMode() {
    return hasFlag(mode_, MonitorMode::Profiling);
}

void CuptiBackend::stop() {
    if (!initialized_) return;
    active_.store(false);

    cuptiActivityFlushAll(1);

    if (IsMonitoringMode()) {
        std::set<CUpti_ActivityKind> kinds;
        {
            std::lock_guard<std::mutex> lk(handler_mu_);
            for (const auto& h : handlers_)
                for (auto k : h->requiredActivityKinds()) kinds.insert(k);
        }
        for (auto k : kinds) cuptiActivityDisable(k);
    }
}

void CuptiBackend::RegisterHandler(std::shared_ptr<ICuptiHandler> handler) {
    if (!handler) return;
    std::lock_guard<std::mutex> lk(handler_mu_);
    handlers_.push_back(handler);
}

void CuptiBackend::StartPcSampling() {
    // Only use PC Sampling API if Activity API failed
    if (pc_sampling_method_ != PCSamplingMethod::SamplingAPI) {
        return;
    }

    if (pc_sampling_ref_count_.fetch_add(1) > 0) {
        GFL_LOG_DEBUG("[PC Sampling] already active (RefCount=",
                      pc_sampling_ref_count_.load(), ")");
        return;
    }

    EnableProfilingFeatures();

    if (!this->ctx_) {
        GFL_LOG_ERROR("[GPUFL] Cannot start PC Sampling: ctx_ is NULL!");
        return;
    }

    GFL_LOG_DEBUG("[PC Sampling] Starting with ctx=", (void*)this->ctx_);

    CUpti_PCSamplingStartParams startParams = {};
    startParams.size = sizeof(CUpti_PCSamplingStartParams);
    startParams.ctx = this->ctx_;
    CUptiResult res = cuptiPCSamplingStart(&startParams);
    if (res == CUPTI_ERROR_INVALID_OPERATION) {
        // This is fine! It means Enable() implicitly started the sampler.
        GFL_LOG_DEBUG("[GPUFL] PC Sampling already active (Implicit Start).");
    } else if (res == CUPTI_ERROR_NOT_SUPPORTED ||
               res == CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED) {
        GFL_LOG_DEBUG(
            "[GPUFL] PC Sampling not supported on this GPU/configuration.");
    } else if (res != CUPTI_SUCCESS) {
        const char* err;
        cuptiGetResultString(res, &err);
        GFL_LOG_ERROR("[PC Sampling] cuptiPCSamplingStart failed: ", err,
                      " (Code: ", res, ")");
    } else {
        GFL_LOG_DEBUG("[PC Sampling] >>> STARTED (Scope Begin) <<<");
    }
}

void CuptiBackend::StopAndCollectPcSampling() const {
    // Only use PC Sampling API if Activity API failed
    if (pc_sampling_method_ != PCSamplingMethod::SamplingAPI) {
        return;
    }

    int refs = const_cast<std::atomic<int>&>(pc_sampling_ref_count_).fetch_sub(1);
    if (refs > 1) {
        GFL_LOG_DEBUG("[PC Sampling] still active (RefCount=", refs - 1, ")");
        return;
    } else if (refs < 1) {
        GFL_LOG_ERROR("[PC Sampling] RefCount underflow!");
        const_cast<std::atomic<int>&>(pc_sampling_ref_count_).store(0);
        return;
    }

    if (!this->ctx_ || !IsContextValid(this->ctx_)) {
        GFL_LOG_ERROR("[GPUFL] Aborting PC Sampling: Context invalid.");
        return;
    }

    if (!pc_sampling_buffers_ || !pc_sampling_buffers_->data) {
        GFL_LOG_ERROR("[GPUFL] No PC sampling buffers allocated!");
        return;
    }

    GFL_LOG_DEBUG("[PC Sampling] <<< STOPPING (Scope End) >>>");

    // Sync before stop to ensure all samples are in the buffer
    cudaDeviceSynchronize();

    // Stop PC Sampling
    CUpti_PCSamplingStopParams stopParams = {};
    stopParams.size = sizeof(CUpti_PCSamplingStopParams);
    stopParams.ctx = this->ctx_;
    cuptiPCSamplingStop(&stopParams);

    CUpti_PCSamplingGetDataParams getDataParams = {};
    getDataParams.size = sizeof(CUpti_PCSamplingGetDataParams);
    getDataParams.ctx = this->ctx_;
    getDataParams.pcSamplingData = pc_sampling_buffers_->data;

    // Drain loop: CUPTI_ERROR_OUT_OF_MEMORY signals "more data remains",
    // not a fatal error. Keep calling GetData until all samples are consumed.
    while (true) {
        pc_sampling_buffers_->data->totalNumPcs = 0;
        CUptiResult getRes = cuptiPCSamplingGetData(&getDataParams);
        const bool hasMore = (getRes == CUPTI_ERROR_OUT_OF_MEMORY);

        if (getRes != CUPTI_SUCCESS && !hasMore) {
            const char* err;
            cuptiGetResultString(getRes, &err);
            GFL_LOG_ERROR("[PC Sampling] cuptiPCSamplingGetData FAILED: ", err);
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
                                cuptiGetDeviceId(this->ctx_, &out.device_id);
                            res != CUPTI_SUCCESS) {
                            GFL_LOG_ERROR("[GPUFL] cuptiGetDeviceId FAILED: ",
                                          res);
                        }
                        out.corr_id = pc.correlationId;
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
                        // Source Correlation
                        {
                            std::lock_guard<std::mutex> lk(
                                const_cast<CuptiBackend*>(this)->cubin_mu_);
                            auto it = cubin_by_crc_.find(pc.cubinCrc);
                            if (it != cubin_by_crc_.end()) {
                                GFL_LOG_DEBUG(
                                    "start getting source correlation");
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
                                    std ::cout << "sourceFile is "
                                               << sourceCorr.fileName
                                               << std ::endl;
                                }
                            }
                        }

                        {
                            std::lock_guard<std::mutex> lk(
                                this->stall_reason_mu_);
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

    // Disable PC Sampling
    CUpti_PCSamplingDisableParams disableParams = {};
    disableParams.size = sizeof(CUpti_PCSamplingDisableParams);
    disableParams.ctx = this->ctx_;
    cuptiPCSamplingDisable(&disableParams);
}

void CuptiBackend::StopAndCollectSassMetrics() const {
    if (!this->ctx_) return;

    CUpti_SassMetricsGetDataProperties_Params props = {
        CUpti_SassMetricsGetDataProperties_Params_STRUCT_SIZE};
    props.ctx = this->ctx_;
    if (cuptiSassMetricsGetDataProperties(&props) != CUPTI_SUCCESS ||
        props.numOfPatchedInstructionRecords == 0) {
        return;
    }

    // Allocate memory for records
    size_t nRecords = props.numOfPatchedInstructionRecords;
    size_t nInstances = props.numOfInstances;
    auto* data = static_cast<CUpti_SassMetrics_Data*>(
        std::calloc(nRecords, sizeof(CUpti_SassMetrics_Data)));
    auto* instances = static_cast<CUpti_SassMetrics_InstanceValue*>(std::calloc(
        nRecords * nInstances, sizeof(CUpti_SassMetrics_InstanceValue)));

    for (size_t i = 0; i < nRecords; ++i) {
        data[i].structSize = sizeof(CUpti_SassMetrics_Data);
        data[i].pInstanceValues = &instances[i * nInstances];
    }

    CUpti_SassMetricsFlushData_Params flushParams = {
        CUpti_SassMetricsFlushData_Params_STRUCT_SIZE};
    flushParams.ctx = this->ctx_;
    flushParams.numOfPatchedInstructionRecords = nRecords;
    flushParams.numOfInstances = nInstances;
    flushParams.pMetricsData = data;

    if (cuptiSassMetricsFlushData(&flushParams) == CUPTI_SUCCESS) {
        for (size_t i = 0; i < nRecords; ++i) {
            // Correlation using your 13.1 specific struct
            CUpti_GetSassToSourceCorrelationParams corrParams = {
                sizeof(CUpti_GetSassToSourceCorrelationParams)};

            // Find the cubin in your map using the CRC
            std::lock_guard<std::mutex> lk(
                const_cast<CuptiBackend*>(this)->cubin_mu_);
            auto it = cubin_by_crc_.find(data[i].cubinCrc);

            if (it != cubin_by_crc_.end()) {
                corrParams.cubin = it->second.data.data();
                corrParams.cubinSize = it->second.data.size();
                corrParams.functionName = data[i].functionName;
                corrParams.pcOffset = data[i].pcOffset;

                // Perform the offline correlation
                CUptiResult res = cuptiGetSassToSourceCorrelation(&corrParams);
                if (res == CUPTI_SUCCESS) {
                    ActivityRecord out{};
                    out.type = TraceType::PC_SAMPLE;
                    out.source_line = corrParams.lineNumber;
                    std::strncpy(out.source_file, corrParams.fileName,
                                 sizeof(out.source_file) - 1);

                    if (corrParams.lineNumber == 0) {
                        // This is where you detect the "0" result
                        GFL_LOG_DEBUG(
                            "Correlation successful, but Line Number is 0. "
                            "Check for -lineinfo.");
                    } else {
                        std::snprintf(out.source_file, sizeof(out.source_file),
                                      "%s", corrParams.fileName);
                        out.source_line = corrParams.lineNumber;
                    }

                    // Cleanup strings allocated by CUPTI
                    std::free(corrParams.fileName);
                    std::free(corrParams.dirName);

                    g_monitorBuffer.Push(out);
                } else {
                    const char* errStr;
                    cuptiGetResultString(res, &errStr);
                    GFL_LOG_ERROR("Correlation failed with error: ", errStr);
                }
            }
        }
    }
    std::free(instances);
    std::free(data);
}

void CuptiBackend::EnableSassMetrics() {
    CUpti_Device_GetChipName_Params getChipNameParams = {
        CUpti_Device_GetChipName_Params_STRUCT_SIZE};
    getChipNameParams.deviceIndex = 0;
    char chipName[256];
    getChipNameParams.pChipName = chipName;
    if (cuptiDeviceGetChipName(&getChipNameParams) != CUPTI_SUCCESS) return;

    CUpti_SassMetrics_GetProperties_Params propParams = {
        CUpti_SassMetrics_GetProperties_Params_STRUCT_SIZE};
    propParams.pChipName = chipName;
    propParams.pMetricName = "smsp__sass_inst_executed";
    if (cuptiSassMetricsGetProperties(&propParams) != CUPTI_SUCCESS) return;

    if (!sass_metrics_buffers_) {
        sass_metrics_buffers_ = new SassMetricsBuffers();
        sass_metrics_buffers_->config = static_cast<CUpti_SassMetrics_Config*>(
            std::malloc(sizeof(CUpti_SassMetrics_Config)));
        sass_metrics_buffers_->config[0].metricId = propParams.metric.metricId;
        sass_metrics_buffers_->config[0].outputGranularity =
            CUPTI_SASS_METRICS_OUTPUT_GRANULARITY_GPU;
        sass_metrics_buffers_->numMetrics = 1;
    }

    CUpti_SassMetricsSetConfig_Params setConfigParams = {
        CUpti_SassMetricsSetConfig_Params_STRUCT_SIZE};
    setConfigParams.deviceIndex = 0;
    setConfigParams.numOfMetricConfig = sass_metrics_buffers_->numMetrics;
    setConfigParams.pConfigs = sass_metrics_buffers_->config;
    CUptiResult res = cuptiSassMetricsSetConfig(&setConfigParams);
    if (res == CUPTI_SUCCESS || res == CUPTI_ERROR_INVALID_OPERATION) {
        CUpti_SassMetricsEnable_Params enableParams = {
            CUpti_SassMetricsEnable_Params_STRUCT_SIZE};
        enableParams.ctx = this->ctx_;
        enableParams.enableLazyPatching = 1;
        cuptiSassMetricsEnable(&enableParams);
    }
}

void CuptiBackend::EnableProfilingFeatures() {
    if (pc_sampling_method_ == PCSamplingMethod::None) return;

    GFL_LOG_DEBUG("Configuring PC Sampling...");

    // SamplingAPI Path
    if (!this->ctx_) {
        cuCtxGetCurrent(&this->ctx_);
        if (!this->ctx_) {
            cudaFree(nullptr);  // Force init
            cuCtxGetCurrent(&this->ctx_);
        }
        if (!this->ctx_) {
            CUdevice dev;
            cuDeviceGet(&dev, 0);
            cuDevicePrimaryCtxRetain(&this->ctx_, dev);
            cuCtxPushCurrent(this->ctx_);
        }
    }

    if (!this->ctx_) {
        GFL_LOG_ERROR("[FATAL] No Context for Profiling.");
        return;
    }

    CUdevice dev;
    cuCtxGetDevice(&dev);
    char nameBuf[256];
    if (cuDeviceGetName(nameBuf, sizeof(nameBuf), dev) == CUDA_SUCCESS) {
        this->cached_device_name_ = std::string(nameBuf);
    }

    CUpti_PCSamplingEnableParams enableParams = {};
    enableParams.size = sizeof(CUpti_PCSamplingEnableParams);
    enableParams.ctx = this->ctx_;
    CUptiResult enableRes = cuptiPCSamplingEnable(&enableParams);

    if (enableRes ==
        7) {  // CUPTI_ERROR_INVALID_OPERATION or CUPTI_ERROR_ALREADY_ENABLED
        GFL_LOG_DEBUG("[PC Sampling] cuptiPCSamplingEnable: Already enabled.");
    } else if (enableRes != CUPTI_SUCCESS) {
        const char* err;
        cuptiGetResultString(enableRes, &err);
        GFL_LOG_ERROR("[PC Sampling] cuptiPCSamplingEnable FAILED: ", err);
        return;
    }

    if (!pc_sampling_buffers_) {
        constexpr size_t kMaxPcs = 65536;
        pc_sampling_buffers_ =
            std::unique_ptr<PCSamplingBuffers, PCSamplingDeleter>(
                new PCSamplingBuffers());
        pc_sampling_buffers_->pcRecords = static_cast<CUpti_PCSamplingPCData*>(
            std::calloc(kMaxPcs, sizeof(CUpti_PCSamplingPCData)));

        CUpti_PCSamplingGetNumStallReasonsParams numParams = {};
        // Manually set the size if the macro is missing
        numParams.size = sizeof(CUpti_PCSamplingGetNumStallReasonsParams);
        numParams.ctx = this->ctx_;

        size_t numStallReasons = 0;
        numParams.numStallReasons = &numStallReasons;

        if (cuptiPCSamplingGetNumStallReasons(&numParams) == CUPTI_SUCCESS &&
            numStallReasons > 0) {
            // 2. Prepare the buffer for names (each string is 64 bytes)
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
            getParams.ctx = this->ctx_;
            getParams.pPriv = nullptr;
            getParams.numStallReasons = numStallReasons;
            getParams.stallReasonIndex = stallIndices;
            getParams.stallReasons = stallReasonNames;

            CUptiResult res = cuptiPCSamplingGetStallReasons(&getParams);
            if (res == CUPTI_SUCCESS) {
                std::lock_guard<std::mutex> lk(stall_reason_mu_);
                for (size_t i = 0; i < numStallReasons; i++) {
                    // Map the hardware index to the descriptive name
                    uint32_t hwIndex = stallIndices[i];
                    stall_reason_map_[hwIndex] = std::string(stallReasonNames[i]);
                    GFL_LOG_DEBUG("Mapped Stall ", hwIndex, " to ",
                                  stallReasonNames[i]);
                    free(stallReasonNames[i]);
                }
            } else {
                std::cout << "error " << res << std::endl;
            }
            free(stallIndices);
            free(stallReasonNames);
        }

        for (size_t i = 0; i < kMaxPcs; ++i) {
            pc_sampling_buffers_->pcRecords[i].size =
                sizeof(CUpti_PCSamplingPCData);
            pc_sampling_buffers_->pcRecords[i].stallReasonCount = numStallReasons;
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
    configParams.ctx = this->ctx_;
    configParams.numAttributes = 7;
    configParams.pPCSamplingConfigurationInfo = configInfo;

    CUptiResult configRes =
        cuptiPCSamplingSetConfigurationAttribute(&configParams);
    if (configRes != CUPTI_SUCCESS) {
        const char* err;
        cuptiGetResultString(configRes, &err);
        GFL_LOG_ERROR("[PC Sampling] SetConfigurationAttribute Failed: ", err);
    } else {
        GFL_LOG_DEBUG("[PC Sampling] configured and enabled successfully.");
    }

    // Fetch Stall Reasons after enable
}

// Static callback implementations
void CUPTIAPI CuptiBackend::BufferRequested(uint8_t** buffer, size_t* size,
                                            size_t* maxNumRecords) {
    *size = 64 * 1024;
    *buffer = static_cast<uint8_t*>(malloc(*size));
    *maxNumRecords = 0;
}

void CUPTIAPI CuptiBackend::BufferCompleted(CUcontext context,
                                            uint32_t streamId, uint8_t* buffer,
                                            size_t size,
                                            const size_t validSize) {
    auto* backend = g_activeBackend.load(std::memory_order_acquire);
    if (!backend) {
        ::gpufl::DebugLogger::error("[CUPTI] ",
                                    "BufferCompleted: No active backend!");
        if (buffer) free(buffer);
        return;
    }

    static int64_t baseCpuNs = detail::GetTimestampNs();
    static uint64_t baseCuptiTs = 0;
    if (baseCuptiTs == 0) cuptiGetTimestamp(&baseCuptiTs);

    std::vector<std::shared_ptr<ICuptiHandler>> handlers;
    {
        std::lock_guard<std::mutex> lk(backend->handler_mu_);
        handlers = backend->handlers_;
    }

    if (validSize > 0) {
        CUpti_Activity* record = nullptr;
        while (true) {
            const CUptiResult st =
                cuptiActivityGetNextRecord(buffer, validSize, &record);
            if (st == CUPTI_SUCCESS) {
                bool handled = false;
                for (const auto& h : handlers) {
                    if (h->handleActivityRecord(record, baseCpuNs,
                                                baseCuptiTs)) {
                        handled = true;
                        break;
                    }
                }
                // PC_SAMPLING (Activity API path) — profiling-specific, not
                // handler-owned
                if (!handled &&
                    record->kind == CUPTI_ACTIVITY_KIND_PC_SAMPLING) {
                    auto* pc =
                        reinterpret_cast<CUpti_ActivityPCSampling3*>(record);
                    ActivityRecord out{};
                    out.type = TraceType::PC_SAMPLE;
                    out.corr_id = pc->correlationId;
                    out.samples_count = pc->samples;
                    out.stall_reason = pc->stallReason;
                    out.device_id =
                        reinterpret_cast<const CUpti_ActivityKernel11*>(record)
                            ->deviceId;
                    g_monitorBuffer.Push(out);
                }
            } else if (st == CUPTI_ERROR_MAX_LIMIT_REACHED) {
                break;
            } else {
                ::gpufl::DebugLogger::error("[CUPTI] ",
                                            "Error parsing buffer: ", st);
                break;
            }
        }
    }

    free(buffer);
}

void CuptiBackend::GflCallback(void* userdata, CUpti_CallbackDomain domain,
                               CUpti_CallbackId cbid, const void* cbdata) {
    if (!cbdata) return;

    auto* backend = static_cast<CuptiBackend*>(userdata);
    if (!backend) return;

    std::vector<std::shared_ptr<ICuptiHandler>> handlers;
    {
        std::lock_guard<std::mutex> lk(backend->handler_mu_);
        handlers = backend->handlers_;
    }

    bool apiHandled = false;

    for (const auto& handler : handlers) {
        if (handler->shouldHandle(domain, cbid)) {
            if (domain == CUPTI_CB_DOMAIN_RUNTIME_API ||
                domain == CUPTI_CB_DOMAIN_DRIVER_API) {
                if (apiHandled) {
                    continue;
                }
                apiHandled = true;
            }
            GFL_LOG_DEBUG("Calling handler: ", handler->getName());
            handler->handle(domain, cbid, cbdata);
        }
    }
}
}  // namespace gpufl

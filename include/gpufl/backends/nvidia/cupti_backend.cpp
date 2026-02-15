#include "gpufl/backends/nvidia/cupti_backend.hpp"
#include "gpufl/core/ring_buffer.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/trace_type.hpp"
#include "gpufl/core/debug_logger.hpp"

#include "gpufl/backends/nvidia/sampler/cupti_sass.hpp"
#include <cupti_pcsampling.h>
#include <cupti_sass_metrics.h>
#include <cupti_target.h>
#include <cupti_profiler_target.h>
#include <cstring>

#ifndef CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SOURCE_REPORTING
#define CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SOURCE_REPORTING ((CUpti_PCSamplingConfigurationAttributeType)10)
#endif

#include "gpufl/backends/nvidia/cuda_collector.hpp"
#include "gpufl/core/scope_registry.hpp"
#include "gpufl/core/stack_registry.hpp"
#include "gpufl/core/stack_trace.hpp"
#include "sampler/cupti_sass.hpp"

#define CUPTI_CHECK(call) \
    do { \
        CUptiResult res = (call); \
        if (res != CUPTI_SUCCESS) { \
            const char* errStr;  \
            cuptiGetResultString(res, &errStr); \
            ::gpufl::DebugLogger::error("[GPUFL Monitor] ", errStr); \
        } \
    } while(0)

#define CUPTI_CHECK_RETURN(call, failMsg) \
    do { \
        CUptiResult res = (call); \
        if (res != CUPTI_SUCCESS) { \
            ::gpufl::DebugLogger::error("[GPUFL Monitor] ", (failMsg)); \
            return; \
        } \
    } while(0)

namespace gpufl {
    std::atomic<gpufl::CuptiBackend*> g_activeBackend{nullptr};

    // External ring buffer (defined in monitor.cpp)
    extern RingBuffer<ActivityRecord, 1024> g_monitorBuffer;

    static int GetMaxThreadsPerSM(int deviceId) {
        static std::mutex mu;
        static std::unordered_map<int, int> cache;

        std::lock_guard<std::mutex> lock(mu);
        if (cache.find(deviceId) == cache.end()) {
            cudaDeviceProp prop{};
            if (cudaGetDeviceProperties(&prop, deviceId) == cudaSuccess) {
                cache[deviceId] = prop.maxThreadsPerMultiProcessor;
            } else {
                return 2048; // Fallback for most modern architecture
            }
        }
        return cache[deviceId];
    }

    static void CalculateOccupancy(LaunchMeta& meta, const void* funcPtr) {
        int deviceId = 0;
        cudaGetDevice(&deviceId); // Get current device for this thread

        int maxActiveBlocks = 0;
        int blockSize = meta.blockX * meta.blockY * meta.blockZ;

        // Ask the driver: "Given this function pointer and block size, how many blocks fit?"
        const cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxActiveBlocks,
            funcPtr, // The kernel function pointer
            blockSize,
            meta.dynShared
        );

        if (err == cudaSuccess) {
            const int maxThreadsPerSM = GetMaxThreadsPerSM(deviceId);

            // Theoretical Occupancy = (Active Warps) / (Max Warps)
            if (maxThreadsPerSM > 0) {
                meta.occupancy = static_cast<float>(maxActiveBlocks * blockSize) / static_cast<float>(maxThreadsPerSM);
            }
            meta.maxActiveBlocks = maxActiveBlocks;

            GFL_LOG_DEBUG("Occupancy: ", meta.occupancy * 100.0f, "% (", maxActiveBlocks, " blocks/SM)");
        } else {
            meta.occupancy = 0.0f;
            meta.maxActiveBlocks = 0;
        }
    }

    void CuptiBackend::initialize(const MonitorOptions &opts) {
        opts_ = opts;
        if (opts_.isProfiling) {
            mode_ |= MonitorMode::Profiling;
        }
        DebugLogger::setEnabled(opts_.enableDebugOutput);

        g_activeBackend.store(this, std::memory_order_release);
        GFL_LOG_DEBUG("Subscribing to CUPTI...");
        CUPTI_CHECK_RETURN(
            cuptiSubscribe(&subscriber_, reinterpret_cast<CUpti_CallbackFunc>(GflCallback), this),
            "[GPUFL Monitor] ERROR: Failed to subscribe to CUPTI\n"
            "[GPUFL Monitor] This may indicate:\n"
            "  - CUPTI library not found or incompatible\n"
            "  - Insufficient permissions\n"
            "  - CUDA driver issues"
        );
        GFL_LOG_DEBUG("CUPTI subscription successful");

        // Enable resource domain immediately to catch context creation
        cuptiEnableDomain(1, subscriber_, CUPTI_CB_DOMAIN_RESOURCE);
        cuptiEnableDomain(1, subscriber_, CUPTI_CB_DOMAIN_RUNTIME_API);
        cuptiEnableDomain(1, subscriber_, CUPTI_CB_DOMAIN_DRIVER_API);

        CUptiResult resCb = cuptiActivityRegisterCallbacks(BufferRequested, BufferCompleted);
        if (resCb != CUPTI_SUCCESS) {
            const char* errStr = nullptr;
            cuptiGetResultString(resCb, &errStr);
            GFL_LOG_ERROR("FATAL: Failed to register activity callbacks.");
            GFL_LOG_ERROR("Error: ", (errStr ? errStr : "unknown"), " (Code ", resCb, ")");

            initialized_ = false;
            return;
        }

        initialized_ = true;
        GFL_LOG_DEBUG("Callbacks registered successfully.");
    }

    void CuptiBackend::shutdown() {
        if (!initialized_) return;

        cuptiActivityFlushAll(1);

        if (sassMetricsBuffers_) {
            if (sassMetricsBuffers_->config) {
                std::free(sassMetricsBuffers_->config);
            }
            if (sassMetricsBuffers_->data) {
                std::free(sassMetricsBuffers_->data);
            }
            delete sassMetricsBuffers_;
            sassMetricsBuffers_ = nullptr;
        }

        if (pcSamplingBuffers_) {
            if (pcSamplingBuffers_->pcRecords) {
                size_t maxPcs = pcSamplingBuffers_->data->collectNumPcs;
                for (size_t i = 0; i < maxPcs; ++i) {
                    if (pcSamplingBuffers_->pcRecords[i].stallReason) {
                        std::free(pcSamplingBuffers_->pcRecords[i].stallReason);
                    }
                }
                std::free(pcSamplingBuffers_->data);
            }
            if (pcSamplingBuffers_->pcRecords) {
                std::free(pcSamplingBuffers_->pcRecords);
            }
            delete pcSamplingBuffers_;
            pcSamplingBuffers_ = nullptr;
        }

        cuptiEnableDomain(0, subscriber_, CUPTI_CB_DOMAIN_RESOURCE);
        cuptiEnableDomain(0, subscriber_, CUPTI_CB_DOMAIN_RUNTIME_API);
        cuptiEnableDomain(0, subscriber_, CUPTI_CB_DOMAIN_DRIVER_API);

        cuptiUnsubscribe(subscriber_);
        g_activeBackend.store(nullptr, std::memory_order_release);
        initialized_ = false;
    }

    CUptiResult(* CuptiBackend::get_value())(CUpti_ActivityKind) {
        return cuptiActivityEnable;
    }

    void CuptiBackend::start() {
        if (!initialized_) return;

        CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR));

        if (isProfilingMode()) {
            // Ensure context is initialized
            if (!this->ctx_) {
                cuCtxGetCurrent(&this->ctx_);
                if (!this->ctx_) {
                    cudaFree(nullptr); // Force init
                    cuCtxGetCurrent(&this->ctx_);
                }
            }

            if (this->ctx_) {
                CUdevice dev;
                cuCtxGetDevice(&dev);
                int deviceIndex = 0; // Assuming device 0 for now or getting from ctx
                CUpti_Device_GetChipName_Params getChipNameParams = {CUpti_Device_GetChipName_Params_STRUCT_SIZE};
                getChipNameParams.deviceIndex = 0;
                char chipName[256];
                getChipNameParams.pChipName = chipName;
                if (cuptiDeviceGetChipName(&getChipNameParams) == CUPTI_SUCCESS) {
                    CUpti_SassMetrics_GetProperties_Params propParams = {CUpti_SassMetrics_GetProperties_Params_STRUCT_SIZE};
                    propParams.pChipName = chipName;
                    propParams.pMetricName = "smsp__sass_inst_executed";
                    if (cuptiSassMetricsGetProperties(&propParams) == CUPTI_SUCCESS) {
                        if (!sassMetricsBuffers_) {
                            sassMetricsBuffers_ = new SassMetricsBuffers();
                            sassMetricsBuffers_->config = static_cast<CUpti_SassMetrics_Config*>(std::malloc(sizeof(CUpti_SassMetrics_Config)));
                            sassMetricsBuffers_->config[0].metricId = propParams.metric.metricId;
                            sassMetricsBuffers_->config[0].outputGranularity = CUPTI_SASS_METRICS_OUTPUT_GRANULARITY_GPU;
                            sassMetricsBuffers_->numMetrics = 1;
                        }

                        CUpti_SassMetricsSetConfig_Params setConfigParams = {CUpti_SassMetricsSetConfig_Params_STRUCT_SIZE};
                        setConfigParams.deviceIndex = 0;
                        setConfigParams.numOfMetricConfig = sassMetricsBuffers_->numMetrics;
                        setConfigParams.pConfigs = sassMetricsBuffers_->config;
                        CUptiResult res = cuptiSassMetricsSetConfig(&setConfigParams);
                        if (res == CUPTI_SUCCESS || res == CUPTI_ERROR_INVALID_OPERATION) {
                             CUpti_SassMetricsEnable_Params enableParams = {CUpti_SassMetricsEnable_Params_STRUCT_SIZE};
                             enableParams.ctx = this->ctx_;
                             enableParams.enableLazyPatching = 1;
                             cuptiSassMetricsEnable(&enableParams);
                        }
                    }
                }
            }
        }

        if (isMonitoringMode()) {
            CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
        }

        if (isProfilingMode()) {
            // STRATEGY 1: Try Activity API first (works on older GPUs)
            CUptiResult pcRes = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_PC_SAMPLING);
            if (pcRes == CUPTI_SUCCESS) {
                pcSamplingMethod_ = PCSamplingMethod::ActivityAPI;
                GFL_LOG_DEBUG("[PC Sampling] Using Activity API (CUPTI_ACTIVITY_KIND_PC_SAMPLING)");
            } else if (pcRes == CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED) {
                // STRATEGY 2: Fallback to PC Sampling API (works on newer GPUs like RTX 40xx/50xx)
                GFL_LOG_DEBUG("[PC Sampling] Activity API not supported, trying PC Sampling API...");
                pcSamplingMethod_ = PCSamplingMethod::SamplingAPI;
                GFL_LOG_DEBUG("[PC Sampling] Using PC Sampling API (cuptiPCSampling*)");
                cuptiActivityEnable(CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR);
                GFL_LOG_DEBUG("[PC Sampling] Enabled SOURCE_LOCATOR activity kind for Sampling API.");
            } else {
                const char* err;
                cuptiGetResultString(pcRes, &err);
                GFL_LOG_ERROR("[PC Sampling] Failed to enable: ", err);
                pcSamplingMethod_ = PCSamplingMethod::None;
            }
        }

        active_.store(true);
        GFL_LOG_DEBUG("Backend started. Mode bitmask: ", static_cast<int>(mode_));
    }

    bool CuptiBackend::isMonitoringMode() {
        return hasFlag(mode_, MonitorMode::Monitoring) ||
                         hasFlag(mode_, MonitorMode::Profiling);
    }

    bool CuptiBackend::isProfilingMode() {
        return hasFlag(mode_, MonitorMode::Profiling);
    }


    void CuptiBackend::stop() {
        if (!initialized_) return;
        active_.store(false);

        cuptiActivityFlushAll(1);

        if (isMonitoringMode()) {
            cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL);
            cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
        }
    }

    void CuptiBackend::startPCSampling() {
        // Only use PC Sampling API if Activity API failed
        if (pcSamplingMethod_ != PCSamplingMethod::SamplingAPI) {
            return;
        }

        if (pcSamplingRefCount_.fetch_add(1) > 0) {
            GFL_LOG_DEBUG("[PC Sampling] already active (RefCount=", pcSamplingRefCount_.load(), ")");
            return;
        }

        enableProfilingFeatures();

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
        } else if (res == CUPTI_ERROR_NOT_SUPPORTED || res == CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED) {
            GFL_LOG_DEBUG("[GPUFL] PC Sampling not supported on this GPU/configuration.");
        } else if (res != CUPTI_SUCCESS) {
            const char* err; cuptiGetResultString(res, &err);
            GFL_LOG_ERROR("[PC Sampling] cuptiPCSamplingStart failed: ", err, " (Code: ", res, ")");
        } else {
            GFL_LOG_DEBUG("[PC Sampling] >>> STARTED (Scope Begin) <<<");
        }
    }

    static bool IsContextValid(CUcontext ctx) {
        CUcontext current;
        return (cuCtxGetCurrent(&current) == CUDA_SUCCESS);
    }

    void CuptiBackend::stopAndCollectPCSampling() const {
        // Only use PC Sampling API if Activity API failed
        if (pcSamplingMethod_ != PCSamplingMethod::SamplingAPI) {
            return;
        }

        int refs = const_cast<std::atomic<int>&>(pcSamplingRefCount_).fetch_sub(1);
        if (refs > 1) {
            GFL_LOG_DEBUG("[PC Sampling] still active (RefCount=", refs - 1, ")");
            return;
        } else if (refs < 1) {
            GFL_LOG_ERROR("[PC Sampling] RefCount underflow!");
            const_cast<std::atomic<int>&>(pcSamplingRefCount_).store(0);
            return;
        }

        if (!this->ctx_ || !IsContextValid(this->ctx_)) {
            GFL_LOG_ERROR("[GPUFL] Aborting PC Sampling: Context invalid.");
            return;
        }

        if (!pcSamplingBuffers_ || !pcSamplingBuffers_->data) {
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

        // Get PC Sampling data
        CUpti_PCSamplingGetDataParams getDataParams = {};
        getDataParams.size = sizeof(CUpti_PCSamplingGetDataParams);
        getDataParams.ctx = this->ctx_;
        getDataParams.pcSamplingData = pcSamplingBuffers_->data;
        CUptiResult getRes = cuptiPCSamplingGetData(&getDataParams);
        if (getRes != CUPTI_SUCCESS) {
            const char* err; cuptiGetResultString(getRes, &err);
            GFL_LOG_ERROR("[PC Sampling] cuptiPCSamplingGetData FAILED: ", err);
        } else {
            auto numPcs = pcSamplingBuffers_->data->totalNumPcs;
            GFL_LOG_DEBUG("[PC Sampling] Collected ", numPcs, " PC records.");

            for (size_t i = 0; i < numPcs; ++i) {
                const CUpti_PCSamplingPCData& pc = pcSamplingBuffers_->data->pPcData[i];
                if (pc.stallReasonCount > 0 && pc.stallReason) {
                    for (uint32_t j = 0; j < pc.stallReasonCount; ++j) {
                        if (pc.stallReason[j].samples > 0) {
                            ActivityRecord out{};
                            out.type = TraceType::PC_SAMPLE;
                            if (CUptiResult res = cuptiGetDeviceId(this->ctx_, &out.deviceId); res != CUPTI_SUCCESS) {
                                GFL_LOG_ERROR("[GPUFL] cuptiGetDeviceId FAILED: ", res);
                            }
                            out.corrId = pc.correlationId;
                            out.samplesCount = pc.stallReason[j].samples;
                            out.stallReason = pc.stallReason[j].pcSamplingStallReasonIndex;
                            out.cpuStartNs = detail::getTimestampNs();

                            if (pc.functionName) {
                                std::snprintf(out.functionName, sizeof(out.functionName), "%s", pc.functionName);
                            } else {
                                std::snprintf(out.functionName, sizeof(out.functionName), "unknown");
                            }
                            // Source Correlation
                            {
                                std::lock_guard<std::mutex> lk(const_cast<CuptiBackend*>(this)->cubinMu_);
                                auto it = cubinByCrc_.find(pc.cubinCrc);
                                if (it != cubinByCrc_.end()) {
                                    GFL_LOG_DEBUG("start getting source correlation");
                                    auto sourceCorr = nvidia::CuptiSass::sampleSourceCorrelation(
                                        it->second.data.data(),
                                        it->second.data.size(),
                                        out.functionName,
                                        pc.pcOffset
                                    );
                                    if (!sourceCorr.fileName.empty()) {
                                        std::snprintf(out.sourceFile, sizeof(out.sourceFile), "%s", sourceCorr.fileName.c_str());
                                        out.sourceLine = sourceCorr.lineNumber;
                                        std :: cout << "sourceFile is " << sourceCorr.fileName << std :: endl;
                                    }
                                }
                            }

                            {
                                std::lock_guard<std::mutex> lk(const_cast<CuptiBackend*>(this)->stallReasonMu_);
                                auto it = stallReasonMap_.find(out.stallReason);
                                if (it != stallReasonMap_.end()) {
                                    out.reasonName = it->second;
                                } else {
                                    out.reasonName = "Stall_" + std::to_string(out.stallReason);
                                }
                            }

                            g_monitorBuffer.Push(out);
                        }
                    }
                }
            }
        }

        // Disable PC Sampling
        CUpti_PCSamplingDisableParams disableParams = {};
        disableParams.size = sizeof(CUpti_PCSamplingDisableParams);
        disableParams.ctx = this->ctx_;
        cuptiPCSamplingDisable(&disableParams);
    }

    void CuptiBackend::stopAndCollectSassMetrics() const {
        if (!this->ctx_) return;

        CUpti_SassMetricsGetDataProperties_Params props = {CUpti_SassMetricsGetDataProperties_Params_STRUCT_SIZE};
        props.ctx = this->ctx_;
        if (cuptiSassMetricsGetDataProperties(&props) != CUPTI_SUCCESS || props.numOfPatchedInstructionRecords == 0) {
            return;
        }

        size_t nRecords = props.numOfPatchedInstructionRecords;
        size_t nInstances = props.numOfInstances;

        CUpti_SassMetrics_Data* data = static_cast<CUpti_SassMetrics_Data*>(std::calloc(nRecords, sizeof(CUpti_SassMetrics_Data)));
        CUpti_SassMetrics_InstanceValue* instances = static_cast<CUpti_SassMetrics_InstanceValue*>(std::calloc(nRecords * nInstances, sizeof(CUpti_SassMetrics_InstanceValue)));

        for (size_t i = 0; i < nRecords; ++i) {
            data[i].structSize = sizeof(CUpti_SassMetrics_Data);
            data[i].pInstanceValues = &instances[i * nInstances];
        }

        CUpti_SassMetricsFlushData_Params flushParams = {CUpti_SassMetricsFlushData_Params_STRUCT_SIZE};
        flushParams.ctx = this->ctx_;
        flushParams.numOfPatchedInstructionRecords = nRecords;
        flushParams.numOfInstances = nInstances;
        flushParams.pMetricsData = data;

        CUptiResult res = cuptiSassMetricsFlushData(&flushParams);
        if (res != CUPTI_SUCCESS) {
             const char* err; cuptiGetResultString(res, &err);
             GFL_LOG_ERROR("[SASS Metrics] cuptiSassMetricsFlushData FAILED: ", err);
        } else {
            GFL_LOG_DEBUG("[SASS Metrics] Flushed ", nRecords, " SASS records.");
            for (size_t i = 0; i < nRecords; ++i) {
                for (size_t j = 0; j < nInstances; ++j) {
                    CUpti_SassMetrics_InstanceValue& iv = data[i].pInstanceValues[j];
                    if (iv.value > 0) {
                        ActivityRecord out{};
                        out.type = TraceType::PC_SAMPLE;
                        out.cpuStartNs = detail::getTimestampNs();
                        cuptiGetDeviceId(this->ctx_, &out.deviceId);
                        out.pcOffset = data[i].pcOffset;
                        out.metricValue = iv.value;
                        std::snprintf(out.functionName, sizeof(out.functionName), "%s", data[i].functionName ? data[i].functionName : "unknown");

                        // Try to find metric name
                        std::snprintf(out.metricName, sizeof(out.metricName), "sass_metric_%lu", iv.metricId);
                        if (sassMetricsBuffers_) {
                            for (size_t k = 0; k < sassMetricsBuffers_->numMetrics; ++k) {
                                if (sassMetricsBuffers_->config[k].metricId == iv.metricId) {
                                    // In a real scenario we'd store the name in SassMetricsBuffers
                                    std::snprintf(out.metricName, sizeof(out.metricName), "smsp__sass_inst_executed");
                                    break;
                                }
                            }
                        }

                        // Source Correlation
                        {
                            std::lock_guard<std::mutex> lk(const_cast<CuptiBackend*>(this)->cubinMu_);
                            auto it = cubinByCrc_.find(data[i].cubinCrc);
                            if (it != cubinByCrc_.end()) {
                                auto sourceCorr = nvidia::CuptiSass::sampleSourceCorrelation(
                                    it->second.data.data(),
                                    it->second.data.size(),
                                    out.functionName,
                                    out.pcOffset
                                );
                                if (!sourceCorr.fileName.empty()) {
                                    std::snprintf(out.sourceFile, sizeof(out.sourceFile), "%s", sourceCorr.fileName.c_str());
                                    out.sourceLine = sourceCorr.lineNumber;
                                }
                            }
                        }

                        g_monitorBuffer.Push(out);
                    }
                }
            }
        }

        std::free(instances);
        std::free(data);
    }

    void CuptiBackend::enableProfilingFeatures() {
        if (pcSamplingMethod_ == PCSamplingMethod::None) return;

        GFL_LOG_DEBUG("Configuring PC Sampling...");

        // SamplingAPI Path
        if (!this->ctx_) {
            cuCtxGetCurrent(&this->ctx_);
            if (!this->ctx_) {
                cudaFree(nullptr); // Force init
                cuCtxGetCurrent(&this->ctx_);
            }
            if (!this->ctx_) {
                CUdevice dev; cuDeviceGet(&dev, 0);
                cuDevicePrimaryCtxRetain(&this->ctx_, dev);
                cuCtxPushCurrent(this->ctx_);
            }
        }

        if (!this->ctx_) {
            GFL_LOG_ERROR("[FATAL] No Context for Profiling.");
            return;
        }

        CUdevice dev; cuCtxGetDevice(&dev);
        char nameBuf[256];
        if (cuDeviceGetName(nameBuf, sizeof(nameBuf), dev) == CUDA_SUCCESS) {
            this->cachedDeviceName_ = std::string(nameBuf);
        }

        CUpti_PCSamplingEnableParams enableParams = {};
        enableParams.size = sizeof(CUpti_PCSamplingEnableParams);
        enableParams.ctx = this->ctx_;
        CUptiResult enableRes = cuptiPCSamplingEnable(&enableParams);

        if (enableRes == 7) { // CUPTI_ERROR_INVALID_OPERATION or CUPTI_ERROR_ALREADY_ENABLED
            GFL_LOG_DEBUG("[PC Sampling] cuptiPCSamplingEnable: Already enabled.");
        } else if (enableRes != CUPTI_SUCCESS) {
            const char* err; cuptiGetResultString(enableRes, &err);
            GFL_LOG_ERROR("[PC Sampling] cuptiPCSamplingEnable FAILED: ", err);
            return;
        }


        if (!pcSamplingBuffers_) {
            constexpr size_t kMaxPcs = 65536;
            pcSamplingBuffers_ = new PCSamplingBuffers();
            pcSamplingBuffers_->pcRecords = static_cast<CUpti_PCSamplingPCData*>(std::calloc(kMaxPcs, sizeof(CUpti_PCSamplingPCData)));

            CUpti_PCSamplingGetNumStallReasonsParams numParams = {};
            // Manually set the size if the macro is missing
            numParams.size = sizeof(CUpti_PCSamplingGetNumStallReasonsParams);
            numParams.ctx = this->ctx_;

            size_t numStallReasons = 0;
            numParams.numStallReasons = &numStallReasons;

            if (cuptiPCSamplingGetNumStallReasons(&numParams) == CUPTI_SUCCESS && numStallReasons > 0) {
                // 2. Prepare the buffer for names (each string is 64 bytes)
                auto* stallIndices = static_cast<uint32_t*>(malloc(numStallReasons * sizeof(uint32_t)));
                char** stallReasonNames = static_cast<char**>(malloc(numStallReasons * sizeof(char*)));
                for (size_t i = 0; i < numStallReasons; i++) {
                    stallReasonNames[i] = static_cast<char*>(malloc(CUPTI_STALL_REASON_STRING_SIZE));
                }

                CUpti_PCSamplingGetStallReasonsParams getParams = {sizeof(CUpti_PCSamplingGetStallReasonsParams)};
                getParams.ctx = this->ctx_;
                getParams.pPriv = nullptr;
                getParams.numStallReasons = numStallReasons;
                getParams.stallReasonIndex = stallIndices;
                getParams.stallReasons = stallReasonNames;

                CUptiResult res = cuptiPCSamplingGetStallReasons(&getParams);
                if (res == CUPTI_SUCCESS) {
                    std::lock_guard<std::mutex> lk(stallReasonMu_);
                    for (size_t i = 0; i < numStallReasons; i++) {
                        // Map the hardware index to the descriptive name
                        uint32_t hwIndex = stallIndices[i];
                        stallReasonMap_[hwIndex] = std::string(stallReasonNames[i]);
                        GFL_LOG_DEBUG("Mapped Stall ", hwIndex, " to ", stallReasonNames[i]);
                        free(stallReasonNames[i]);
                    }
                } else {
                    std::cout << "error " << res << std::endl;
                }
                free(stallReasonNames);
            }

            for (size_t i = 0; i < kMaxPcs; ++i) {
                pcSamplingBuffers_->pcRecords[i].size = sizeof(CUpti_PCSamplingPCData);
                pcSamplingBuffers_->pcRecords[i].stallReasonCount = numStallReasons;
                pcSamplingBuffers_->pcRecords[i].stallReason = static_cast<CUpti_PCSamplingStallReason*>(std::calloc(numStallReasons, sizeof(CUpti_PCSamplingStallReason)));
            }
            pcSamplingBuffers_->data = static_cast<CUpti_PCSamplingData*>(std::calloc(1, sizeof(CUpti_PCSamplingData)));
            pcSamplingBuffers_->data->size = sizeof(CUpti_PCSamplingData);
            pcSamplingBuffers_->data->collectNumPcs = kMaxPcs;
            pcSamplingBuffers_->data->pPcData = pcSamplingBuffers_->pcRecords;
        }

        CUpti_PCSamplingConfigurationInfo configInfo[10] = {};
        configInfo[0].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_COLLECTION_MODE;
        configInfo[0].attributeData.collectionModeData.collectionMode = CUPTI_PC_SAMPLING_COLLECTION_MODE_KERNEL_SERIALIZED;

        configInfo[1].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_PERIOD;
        configInfo[1].attributeData.samplingPeriodData.samplingPeriod = 10;

        configInfo[2].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SCRATCH_BUFFER_SIZE;
        configInfo[2].attributeData.scratchBufferSizeData.scratchBufferSize = 256 * 1024 * 1024;

        configInfo[3].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_HARDWARE_BUFFER_SIZE;
        configInfo[3].attributeData.hardwareBufferSizeData.hardwareBufferSize = 256 * 1024 * 1024;

        configInfo[4].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_ENABLE_START_STOP_CONTROL;
        configInfo[4].attributeData.enableStartStopControlData.enableStartStopControl = 1;

        configInfo[5].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_DATA_BUFFER;
        configInfo[5].attributeData.samplingDataBufferData.samplingDataBuffer = pcSamplingBuffers_->data;

        configInfo[6].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_OUTPUT_DATA_FORMAT;
        configInfo[6].attributeData.outputDataFormatData.outputDataFormat = CUPTI_PC_SAMPLING_OUTPUT_DATA_FORMAT_PARSED;

        configInfo[7].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SOURCE_REPORTING;
        configInfo[7].attributeData.invalidData.data[0] = 1; // union field for source reporting might not exist in old headers

        CUpti_PCSamplingConfigurationInfoParams configParams = {};
        configParams.size = sizeof(CUpti_PCSamplingConfigurationInfoParams);
        configParams.ctx = this->ctx_;
        configParams.numAttributes = 8;
        configParams.pPCSamplingConfigurationInfo = configInfo;

        CUptiResult configRes = cuptiPCSamplingSetConfigurationAttribute(&configParams);
        if (configRes != CUPTI_SUCCESS) {
            const char* err; cuptiGetResultString(configRes, &err);
            GFL_LOG_ERROR("[PC Sampling] SetConfigurationAttribute Failed: ", err);
        } else {
            GFL_LOG_DEBUG("[PC Sampling] configured and enabled successfully.");
        }

        // Fetch Stall Reasons after enable
    }


    // Static callback implementations
    void CUPTIAPI CuptiBackend::BufferRequested(uint8_t **buffer, size_t *size,
                                                size_t *maxNumRecords) {
        *size = 64 * 1024;
        *buffer = static_cast<uint8_t *>(malloc(*size));
        *maxNumRecords = 0;
    }

    void CUPTIAPI CuptiBackend::BufferCompleted(CUcontext context,
                                                uint32_t streamId,
                                                uint8_t *buffer, size_t size,
                                                const size_t validSize) {
        auto* backend = g_activeBackend.load(std::memory_order_acquire);
        if (!backend) {
            ::gpufl::DebugLogger::error("[CUPTI] ", "BufferCompleted: No active backend!");
            if (buffer) free(buffer);
            return;
        }

        GFL_LOG_DEBUG("[CUPTI] BufferCompleted validSize=", validSize);
        CUpti_Activity *record = nullptr;

        static int64_t baseCpuNs = detail::getTimestampNs();
        static uint64_t baseCuptiTs = 0;
        if (baseCuptiTs == 0) cuptiGetTimestamp(&baseCuptiTs);

        if (validSize > 0) {
            while (true) {
                const CUptiResult st = cuptiActivityGetNextRecord(
                    buffer, validSize, &record);
                if (st == CUPTI_SUCCESS) {
                    CUpti_ActivityKind recKind = record->kind; // Copy to avoid packed field binding issue
                    GFL_LOG_DEBUG("[CUPTI] Got activity record kind=", recKind);

                    const auto *k = reinterpret_cast<const
                        CUpti_ActivityKernel9 *>(record);

                    if (record->kind == CUPTI_ACTIVITY_KIND_KERNEL ||
                        record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {

                        ActivityRecord out{};
                        out.deviceId = k->deviceId;
                        out.type = TraceType::KERNEL;
                        std::snprintf(out.name, sizeof(out.name), "%s", (k->name ? k->name : "kernel"));
                        out.cpuStartNs = baseCpuNs + static_cast<int64_t>(k->start - baseCuptiTs);
                        out.durationNs = static_cast<int64_t>(k->end - k->start);
                        out.dynShared = k->dynamicSharedMemory;
                        out.staticShared = k->staticSharedMemory;
                        out.numRegs = k->registersPerThread;

                        out.hasDetails = false;
                        {
                            const uint64_t corr = k->correlationId;
                            out.corrId = corr;
                            std::lock_guard lk(backend->metaMu_);
                            if (auto it = backend->metaByCorr_.find(corr); it != backend->metaByCorr_.end()) {
                                const LaunchMeta &m = it->second;

                                out.scopeDepth = m.scopeDepth;
                                out.stackId = m.stackId;
                                std::copy(std::begin(m.userScope), std::end(m.userScope), std::begin(out.userScope));
                                out.apiStartNs = m.apiEnterNs;
                                out.apiExitNs = m.apiExitNs;

                                if (m.hasDetails) {
                                    out.hasDetails = true;
                                    out.gridX = m.gridX; out.gridY = m.gridY; out.gridZ = m.gridZ;
                                    out.blockX = m.blockX; out.blockY = m.blockY; out.blockZ = m.blockZ;
                                    out.localBytes = static_cast<int>(k->localMemoryPerThread);
                                    out.constBytes = m.constBytes;
                                    out.occupancy = m.occupancy;
                                    out.maxActiveBlocks = m.maxActiveBlocks;
                                    GFL_LOG_DEBUG("[BufferCompleted] Found metadata for CorrID ", corr,
                                                  " with occupancy=", out.occupancy);
                                } else {
                                    GFL_LOG_DEBUG("[BufferCompleted] Found metadata for CorrID ", corr,
                                                  " but hasDetails=false");
                                }

                                backend->metaByCorr_.erase(it);
                            } else {
                                GFL_LOG_DEBUG("[BufferCompleted] No metadata found for CorrID ", corr);
                            }
                        }

                        g_monitorBuffer.Push(out);
                    } else if (record->kind == CUPTI_ACTIVITY_KIND_PC_SAMPLING) {
                        auto* pc = reinterpret_cast<CUpti_ActivityPCSampling3 *>(record);

                        ActivityRecord out{};
                        out.type = TraceType::PC_SAMPLE;
                        out.corrId = pc->correlationId;
                        out.samplesCount = pc->samples;
                        out.stallReason = pc->stallReason;
                        out.deviceId = k->deviceId;

                        g_monitorBuffer.Push(out);
                    }
                } else if (st == CUPTI_ERROR_MAX_LIMIT_REACHED) {
                    // No more records in this buffer
                    break;
                } else {
                    ::gpufl::DebugLogger::error("[CUPTI] ", "Error parsing buffer: ", st);
                    break;
                }
            }
        }

        free(buffer);
    }

    void CUPTIAPI CuptiBackend::GflCallback(void *userdata,
                                            CUpti_CallbackDomain domain,
                                            CUpti_CallbackId cbid,
                                            const void *cbdata) {
        if (!cbdata) return;

        auto *backend = static_cast<CuptiBackend *>(userdata);
        if (!backend || !backend->isActive()) return;

        if (domain == CUPTI_CB_DOMAIN_RESOURCE && cbid == CUPTI_CBID_RESOURCE_CONTEXT_CREATED) {
            auto *resData = static_cast<const CUpti_ResourceData *>(cbdata);
            GFL_LOG_DEBUG("[DEBUG-CALLBACK] Context Created! Enabling Runtime/Driver domains...");
            cuptiEnableDomain(1, backend->getSubscriber(), CUPTI_CB_DOMAIN_RUNTIME_API);
            cuptiEnableDomain(1, backend->getSubscriber(), CUPTI_CB_DOMAIN_DRIVER_API);
            return;
        }

        if (domain == CUPTI_CB_DOMAIN_RESOURCE && cbid == CUPTI_CBID_RESOURCE_MODULE_LOADED) {
            auto *modData = static_cast<const CUpti_ModuleResourceData *>(cbdata);
            if (modData && modData->pCubin && modData->cubinSize > 0) {
                CUpti_GetCubinCrcParams params = {CUpti_GetCubinCrcParamsSize};
                params.cubinSize = modData->cubinSize;
                params.cubin = modData->pCubin;
                if (cuptiGetCubinCrc(&params) == CUPTI_SUCCESS) {
                    std::lock_guard<std::mutex> lk(backend->cubinMu_);
                    auto& info = backend->cubinByCrc_[params.cubinCrc];
                    info.crc = params.cubinCrc;
                    info.data.assign(reinterpret_cast<const uint8_t *>(modData->pCubin), reinterpret_cast<const uint8_t *>(modData->pCubin) + modData->cubinSize);
                    GFL_LOG_DEBUG("[DEBUG-CALLBACK] Module Loaded: CRC=", params.cubinCrc, " Size=", modData->cubinSize);
                }
            }
            return;
        }

        if (!backend->isActive()) {
            GFL_LOG_DEBUG("[DEBUG-CALLBACK] Backend not active, skipping callback.");
            return;
        };
        if (domain == CUPTI_CB_DOMAIN_STATE) return;

        // Only care about runtime/driver API for launch metadata
        if (domain != CUPTI_CB_DOMAIN_RUNTIME_API && domain !=
            CUPTI_CB_DOMAIN_DRIVER_API) return;

        auto *cbInfo = static_cast<const CUpti_CallbackData *>(cbdata);

        bool isKernelLaunch = false;

        if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
            if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020 ||
                cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) {
                isKernelLaunch = true;
            }
        } else {
            // DRIVER API
            if (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunch ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchGrid ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchGridAsync ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz) {
                isKernelLaunch = true;
            }
        }
        if (isKernelLaunch) {
            GFL_LOG_DEBUG("[DEBUG-CALLBACK] >>> KERNEL LAUNCH DETECTED <<< (CorrID ",
                          cbInfo->correlationId, ")");
        }

        if (!isKernelLaunch) return;

        if (cbInfo->callbackSite == CUPTI_API_ENTER) {
            LaunchMeta meta{};
            meta.apiEnterNs = detail::getTimestampNs();

            const char *nm = cbInfo->symbolName
                                 ? cbInfo->symbolName
                                 : cbInfo->functionName;
            if (!nm) nm = "kernel_launch";
            std::snprintf(meta.name, sizeof(meta.name), "%s", nm);

            if (backend->getOptions().enableStackTrace) {
                const std::string trace = gpufl::core::GetCallStack(2);
                const std::string cleanTrace = detail::sanitizeStackTrace(trace);
                meta.stackId = gpufl::StackRegistry::instance().getOrRegister(cleanTrace);
            } else {
                meta.stackId = 0;
            }

            auto& stack = getThreadScopeStack();

            if (!stack.empty()) {
                std::string fullPath;
                for (size_t i = 0; i < stack.size(); ++i) {
                    if (i > 0) fullPath += "|";
                    fullPath += stack[i];
                }
                fullPath += "|";
                fullPath += meta.name;
                std::snprintf(meta.userScope, sizeof(meta.userScope), "%s", fullPath.c_str());
                meta.scopeDepth = stack.size();
            } else {
                std::string fullPath = "global|";
                fullPath += meta.name;
                std::snprintf(meta.userScope, sizeof(meta.userScope), "%s", fullPath.c_str());
                meta.scopeDepth = 0;
            }

            if (backend->getOptions().collectKernelDetails &&
                domain == CUPTI_CB_DOMAIN_RUNTIME_API &&
                cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 &&
                cbInfo->functionParams != nullptr) {
                meta.hasDetails = true;

                const auto *params = (cudaLaunchKernel_v7000_params *) (cbInfo->
                    functionParams);

                meta.gridX = params->gridDim.x;
                meta.gridY = params->gridDim.y;
                meta.gridZ = params->gridDim.z;
                meta.blockX = params->blockDim.x;
                meta.blockY = params->blockDim.y;
                meta.blockZ = params->blockDim.z;
                meta.dynShared = static_cast<int>(params->sharedMem);

                CalculateOccupancy(meta, params->func);
            } else if (backend->getOptions().collectKernelDetails &&
                domain == CUPTI_CB_DOMAIN_DRIVER_API &&
                cbInfo->functionParams != nullptr) {
                if (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel) {
                    meta.hasDetails = true;
                    const auto *params = (cudaLaunchKernel_v7000_params *) (cbInfo->
                    functionParams);

                    meta.gridX = params->gridDim.x;
                    meta.gridY = params->gridDim.y;
                    meta.gridZ = params->gridDim.z;
                    meta.blockX = params->blockDim.x;
                    meta.blockY = params->blockDim.y;
                    meta.blockZ = params->blockDim.z;
                    meta.dynShared = static_cast<int>(params->sharedMem);

                    CalculateOccupancy(meta, params->func);
                }
            }

            std::lock_guard<std::mutex> lk(backend->metaMu_);
            auto& existing = backend->metaByCorr_[cbInfo->correlationId];

            // If the existing entry has details, but the new one (e.g. from Driver API) does not,
            // KEEP the existing one. Do not overwrite it.
            if (existing.hasDetails && !meta.hasDetails) {
                GFL_LOG_DEBUG("[DEBUG-CALLBACK] Skipping overwrite of rich metadata for CorrID ",
                              cbInfo->correlationId, " by Driver API.");
            } else {
                // Otherwise (it's new, or the new one has details and the old one didn't), update it.
                existing = meta;
            }
        } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
            const int64_t t = detail::getTimestampNs();
            std::lock_guard<std::mutex> lk(backend->metaMu_);
            auto it = backend->metaByCorr_.find(cbInfo->correlationId);
            if (it != backend->metaByCorr_.end()) {
                it->second.apiExitNs = t;
            }
        }
    }
} // namespace gpufl

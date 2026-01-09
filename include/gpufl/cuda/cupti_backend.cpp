#include "gpufl/cuda/cupti_backend.hpp"
#include "gpufl/core/ring_buffer.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/trace_type.hpp"
#include "gpufl/core/debug_logger.hpp"

#include <cupti_pcsampling.h>
#include <cstring>

#include "gpufl/backends/nvidia/cuda_collector.hpp"
#include "gpufl/core/scope_registry.hpp"
#include "gpufl/core/stack_registry.hpp"
#include "gpufl/core/stack_trace.hpp"

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
#pragma pack(push, 8)
    struct CUpti_PCSamplingGetDataParams_NEW {
        size_t size;               // [0-8]
        void* pPriv;               // [8-16]
        CUcontext ctx;             // [16-24]
        void* pcSamplingData;      // [24-32]
        size_t pcSamplingDataSize; // [32-40]
    };

    // LAYOUT B: Legacy CUDA (Pre-11.3)
    // Context is at Offset 8. Size is 24.
    // ** CRITICAL **: If we don't use this layout, the driver reads NULL from pPriv and crashes.
    struct CUpti_PCSamplingGetDataParams_OLD {
        size_t size;            // [0-8]
        CUcontext ctx;          // [8-16]  <-- Context MUST be here for old drivers
        void* pcSamplingData;   // [16-24]
    };
#pragma pack(pop)
    std::atomic<gpufl::CuptiBackend*> g_activeBackend{nullptr};

    // External ring buffer (defined in monitor.cpp)
    extern RingBuffer<ActivityRecord, 1024> g_monitorBuffer;

    std::mutex CuptiBackend::sourceMapMu_;
    std::unordered_map<uint32_t, SourceLocation> CuptiBackend::sourceMap_;

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
        cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxActiveBlocks,
            funcPtr, // The kernel function pointer
            blockSize,
            meta.dynShared
        );

        if (err == cudaSuccess) {
            int maxThreadsPerSM = GetMaxThreadsPerSM(deviceId);

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

        if (isMonitoringMode()) {
            CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
        }

        if (isProfilingMode()) {
            // STRATEGY 1: Try Activity API first (works on older GPUs)
            CUptiResult pcRes = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_PC_SAMPLING);
            if (pcRes == CUPTI_SUCCESS) {
                cuptiActivityEnable(CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR);
                pcSamplingMethod_ = PCSamplingMethod::ActivityAPI;
                GFL_LOG_DEBUG("[PC Sampling] Using Activity API (CUPTI_ACTIVITY_KIND_PC_SAMPLING)");
            } else if (pcRes == CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED) {
                // STRATEGY 2: Fallback to PC Sampling API (works on newer GPUs like RTX 40xx/50xx)
                GFL_LOG_DEBUG("[PC Sampling] Activity API not supported, trying PC Sampling API...");
                pcSamplingMethod_ = PCSamplingMethod::SamplingAPI;
                GFL_LOG_DEBUG("[PC Sampling] Using PC Sampling API (cuptiPCSampling*)");
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

        enableProfilingFeatures();

        if (!this->ctx_) {
            GFL_LOG_ERROR("[GPUFL] Cannot start PC Sampling: ctx_ is NULL!");
            return;
        }

        GFL_LOG_DEBUG("[GPUFL] Starting PC Sampling with ctx=", (void*)this->ctx_);

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
            GFL_LOG_ERROR("[GPUFL] cuptiPCSamplingStart failed: ", err, " (Code: ", res, ")");
        } else {
            GFL_LOG_DEBUG("[GPUFL] >>> PC Sampling STARTED (Scope Begin) <<<");
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

        if (!this->ctx_ || !IsContextValid(this->ctx_)) {
            GFL_LOG_ERROR("[GPUFL] Aborting PC Sampling: Context invalid.");
            return;
        }

        GFL_LOG_DEBUG("[GPUFL] <<< PC Sampling STOPPING (Scope End) >>>");

        // Stop PC Sampling
        CUpti_PCSamplingStopParams stopParams = {};
        stopParams.size = sizeof(CUpti_PCSamplingStopParams);
        stopParams.ctx = this->ctx_;
        cuptiPCSamplingStop(&stopParams);

#ifndef _WIN32
        // On non-Windows platforms, try to collect PC Sampling data via GetData
        GFL_LOG_DEBUG("[GPUFL] Attempting to collect PC Sampling data via cuptiPCSamplingGetData...");

        // Prepare buffer (32MB, 8-byte aligned)
        const size_t bufferSizeBytes = 32 * 1024 * 1024;
        std::vector<uint64_t> alignedBuffer(bufferSizeBytes / sizeof(uint64_t));
        uint8_t* rawBufferBytes = reinterpret_cast<uint8_t*>(alignedBuffer.data());
        std::memset(rawBufferBytes, 0, bufferSizeBytes);

        // Setup data header
        auto* header = reinterpret_cast<CUpti_PCSamplingData*>(rawBufferBytes);
        header->size = sizeof(CUpti_PCSamplingData);

        // Call GetData using official CUPTI struct
        CUpti_PCSamplingGetDataParams params = {};
        params.size = CUpti_PCSamplingGetDataParamsSize;
        params.pPriv = nullptr;
        params.ctx = this->ctx_;
        params.pcSamplingData = rawBufferBytes;

        CUptiResult res = cuptiPCSamplingGetData(&params);
        if (res == CUPTI_SUCCESS) {
            size_t numRecords = header->collectNumPcs;
            GFL_LOG_DEBUG("[GPUFL] GetData succeeded. Collected ", numRecords, " PC Samples.");

            if (numRecords > 0 && numRecords < (bufferSizeBytes / sizeof(CUpti_PCSamplingPCData))) {
                for (size_t i = 0; i < numRecords; ++i) {
                    if ((uint8_t*)&header->pPcData[i] >= (rawBufferBytes + bufferSizeBytes)) break;

                    CUpti_PCSamplingPCData& pcRec = header->pPcData[i];
                    if (pcRec.stallReason == nullptr) continue;

                    uint32_t totalSamples = 0;
                    uint32_t mainReason = 0;
                    for (size_t k = 0; k < pcRec.stallReasonCount; ++k) {
                        totalSamples += pcRec.stallReason[k].samples;
                        if (pcRec.stallReason[k].samples > 0)
                            mainReason = pcRec.stallReason[k].pcSamplingStallReasonIndex;
                    }

                    if (totalSamples > 0) {
                        ActivityRecord rec = {};
                        rec.type = TraceType::PC_SAMPLE;
                        rec.corrId = pcRec.correlationId;
                        rec.stallReason = mainReason;
                        rec.samplesCount = totalSamples;
                        std::snprintf(rec.sourceFile, sizeof(rec.sourceFile), "PC:0x%llx",
                                    (unsigned long long)pcRec.pcOffset);

                        g_monitorBuffer.Push(rec);
                        uint64_t pcOffset = pcRec.pcOffset; // Copy to avoid packed field binding issue
                        GFL_LOG_DEBUG("[PC_SAMPLING] Pushed sample: PC=0x", std::hex, pcOffset, std::dec,
                                     " samples=", totalSamples, " stallReason=", mainReason);
                    }
                }
            }
        } else {
            const char* err;
            cuptiGetResultString(res, &err);
            GFL_LOG_ERROR("[GPUFL] GetData Failed: ", err, " (Code: ", res, ")");
        }
#else
        // On Windows, skip cuptiPCSamplingGetData - it crashes with CUDA 13.1
        GFL_LOG_DEBUG("[GPUFL] Skipping cuptiPCSamplingGetData on Windows (known to crash with CUDA 13.1)");
#endif

        // Disable PC Sampling
        CUpti_PCSamplingDisableParams disableParams = {};
        disableParams.size = sizeof(CUpti_PCSamplingDisableParams);
        disableParams.ctx = this->ctx_;
        cuptiPCSamplingDisable(&disableParams);
    }

    void CuptiBackend::enableProfilingFeatures() {
        GFL_LOG_DEBUG("Configuring PC Sampling...");
        CUptiResult pcRes = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_PC_SAMPLING);
        if (pcRes == CUPTI_SUCCESS) {
            cuptiActivityEnable(CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR);
            GFL_LOG_DEBUG("PC Sampling activities enabled (Activity API)");
        } else {
            GFL_LOG_DEBUG("PC Sampling Activity API not supported on this GPU (legacy profiler)");
            GFL_LOG_DEBUG("Note: Newer GPUs require Profiling API instead of Activity API for PC Sampling");
            const char* err;
            cuptiGetResultString(pcRes, &err);
            GFL_LOG_ERROR("Failed to enable PC Sampling activities: ", err);

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

            CUpti_PCSamplingConfigurationInfo configInfo[3] = {};

            configInfo[0].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_COLLECTION_MODE;
            configInfo[0].attributeData.collectionModeData.collectionMode = CUPTI_PC_SAMPLING_COLLECTION_MODE_CONTINUOUS;

            configInfo[1].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_PERIOD;
            configInfo[1].attributeData.samplingPeriodData.samplingPeriod = 20;

            configInfo[2].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SCRATCH_BUFFER_SIZE;
            configInfo[2].attributeData.scratchBufferSizeData.scratchBufferSize = 4 * 1024 * 1024;

            CUpti_PCSamplingConfigurationInfoParams configParams = {};
            configParams.size = sizeof(CUpti_PCSamplingConfigurationInfoParams);
            configParams.ctx = this->ctx_;
            configParams.numAttributes = 3;
            configParams.pPCSamplingConfigurationInfo = configInfo;

            CUptiResult configRes = cuptiPCSamplingSetConfigurationAttribute(&configParams);
            if (configRes != CUPTI_SUCCESS) {
                const char* err; cuptiGetResultString(configRes, &err);
                GFL_LOG_ERROR("Config Failed: ", err);
            }

            CUpti_PCSamplingConfigurationInfo startStopInfo = {};
            startStopInfo.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_ENABLE_START_STOP_CONTROL;
            startStopInfo.attributeData.enableStartStopControlData.enableStartStopControl = true;

            configParams.numAttributes = 1;
            configParams.pPCSamplingConfigurationInfo = &startStopInfo;
            cuptiPCSamplingSetConfigurationAttribute(&configParams);

            CUpti_PCSamplingEnableParams enableParams = {};
            enableParams.size = sizeof(CUpti_PCSamplingEnableParams);
            enableParams.ctx = this->ctx_;
            CUptiResult enableRes = cuptiPCSamplingEnable(&enableParams);
            if (enableRes == CUPTI_ERROR_NOT_SUPPORTED || enableRes == CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED) {
                GFL_LOG_DEBUG("[GPUFL] PC Sampling not supported on this GPU (newer GPUs require Profiling API)");
            } else if (enableRes != CUPTI_SUCCESS) {
                const char* err; cuptiGetResultString(enableRes, &err);
                GFL_LOG_ERROR("[GPUFL] cuptiPCSamplingEnable FAILED: ", err, " (Code: ", enableRes, ")");
                GFL_LOG_ERROR("[GPUFL]   PC Sampling will not work. Possible causes:");
                GFL_LOG_ERROR("[GPUFL]   - GPU does not support PC Sampling (compute capability < 5.2)");
                GFL_LOG_ERROR("[GPUFL]   - CUPTI permissions issue");
                GFL_LOG_ERROR("[GPUFL]   - Driver/CUPTI version mismatch");
            } else {
                GFL_LOG_DEBUG("[GPUFL] PC Sampling ENABLED successfully for ctx=", (void*)this->ctx_);
            }
        }
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
                    } else if (record->kind == CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR) {
                        auto* sl = reinterpret_cast<CUpti_ActivitySourceLocator *>(record);
                        std::lock_guard<std::mutex> lk(sourceMapMu_);
                        sourceMap_[sl->id] = {
                            (sl->fileName ? sl->fileName : "unknown"),
                            sl->lineNumber
                        };
                    } else if (record->kind == CUPTI_ACTIVITY_KIND_PC_SAMPLING) {
                        auto* pc = reinterpret_cast<CUpti_ActivityPCSampling3 *>(record);

                        ActivityRecord out{};
                        out.type = TraceType::PC_SAMPLE;
                        out.corrId = pc->correlationId;
                        out.samplesCount = pc->samples;
                        out.stallReason = pc->stallReason;
                        out.deviceId = k->deviceId;

                        // Look up source file from sourceLocatorId
                        {
                            std::lock_guard<std::mutex> lk(sourceMapMu_);
                            if (auto it = sourceMap_.find(pc->sourceLocatorId); it != sourceMap_.end()) {
                                std::snprintf(out.sourceFile, sizeof(out.sourceFile), "%s:%u",
                                            it->second.fileName.c_str(), it->second.lineNumber);
                                GFL_LOG_DEBUG("[PC_SAMPLING] Got sample: sourceFile=", out.sourceFile,
                                             " samples=", out.samplesCount, " stallReason=", out.stallReason,
                                             " corrId=", out.corrId);
                            } else {
                                // Fallback to PC offset if source not found
                                uint64_t pcOffset = pc->pcOffset; // Copy to avoid packed field binding issue
                                uint32_t sourceLocId = pc->sourceLocatorId; // Copy to avoid packed field binding issue
                                std::snprintf(out.sourceFile, sizeof(out.sourceFile), "PC:0x%llx",
                                            (unsigned long long)pcOffset);
                                GFL_LOG_DEBUG("[PC_SAMPLING] Got sample: PC=0x", std::hex, pcOffset, std::dec,
                                             " samples=", out.samplesCount, " stallReason=", out.stallReason,
                                             " corrId=", out.corrId, " (sourceLocatorId=", sourceLocId, " not found)");
                            }
                        }

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
                                            CUpti_CallbackData *cbInfo) {
        if (!cbInfo) return;

        auto *backend = static_cast<CuptiBackend *>(userdata);
        if (!backend || !backend->isActive()) return;

        const char* funcName = cbInfo->functionName ? cbInfo->functionName : "unknown";
        const char* symbName = cbInfo->symbolName ? cbInfo->symbolName : "unknown";

        if (domain == CUPTI_CB_DOMAIN_RESOURCE && cbid == CUPTI_CBID_RESOURCE_CONTEXT_CREATED) {
            GFL_LOG_DEBUG("[DEBUG-CALLBACK] Context Created! Enabling Runtime/Driver domains...");
            cuptiEnableDomain(1, backend->getSubscriber(), CUPTI_CB_DOMAIN_RUNTIME_API);
            cuptiEnableDomain(1, backend->getSubscriber(), CUPTI_CB_DOMAIN_DRIVER_API);
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

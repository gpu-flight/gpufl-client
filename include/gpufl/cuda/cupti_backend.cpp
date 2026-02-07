#include "gpufl/cuda/cupti_backend.hpp"
#include "gpufl/core/ring_buffer.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/trace_type.hpp"
#include "gpufl/core/debug_logger.hpp"

#include <cupti_pcsampling.h>
#include <cstdio>
#include <cstring>
#include <vector>

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

        // Only configure PC sampling once
        if (!pcSamplingConfigured_) {
            enableProfilingFeatures();
            pcSamplingConfigured_ = true;
        }

        if (!this->ctx_) {
            cuCtxGetCurrent(&this->ctx_);
            if (!this->ctx_) {
                cudaFree(nullptr);
                cuCtxGetCurrent(&this->ctx_);
            }
        }
        if (!this->ctx_) {
            GFL_LOG_ERROR("[GPUFL] Cannot start PC Sampling: ctx_ is NULL!");
            return;
        }

        GFL_LOG_DEBUG("[GPUFL] Starting PC Sampling with ctx=", (void*)this->ctx_);

        CUpti_PCSamplingStartParams startParams = {};
        startParams.size = sizeof(CUpti_PCSamplingStartParams);
        startParams.ctx = this->ctx_;
        startParams.pPriv = nullptr;

        CUptiResult res = cuptiPCSamplingStart(&startParams);
        if (res == CUPTI_ERROR_INVALID_OPERATION) {
            // This is fine! It means Enable() implicitly started the sampler, or it's already running
            GFL_LOG_DEBUG("[GPUFL] PC Sampling already active (Implicit Start or already started).");
            pcSamplingStarted_ = true;
        } else if (res == CUPTI_ERROR_NOT_SUPPORTED || res == CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED) {
            GFL_LOG_ERROR("[GPUFL] PC Sampling not supported on this GPU/configuration.");
            pcSamplingStarted_ = false;
            return;
        } else if (res != CUPTI_SUCCESS) {
            const char* err; cuptiGetResultString(res, &err);
            GFL_LOG_ERROR("[GPUFL] cuptiPCSamplingStart failed: ", err, " (Code: ", res, ")");
            pcSamplingStarted_ = false;
            return;
        } else {
            GFL_LOG_DEBUG("[GPUFL] >>> PC Sampling STARTED (Scope Begin) <<<");
            pcSamplingStarted_ = true;
        }

        // Ensure kernels are launched after Start
        cudaDeviceSynchronize();
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

        // Check if PC sampling was actually started
        if (!pcSamplingStarted_) {
            GFL_LOG_DEBUG("[GPUFL] Skipping Stop - PC Sampling was never started");
            return;
        }

        if (!this->ctx_ || !IsContextValid(this->ctx_)) {
            GFL_LOG_ERROR("[GPUFL] Aborting PC Sampling: Context invalid.");
            return;
        }

        GFL_LOG_DEBUG("[GPUFL] <<< PC Sampling STOPPING (Scope End) >>>");

        // Ensure all kernels have completed before stopping
        cudaError_t syncErr = cudaDeviceSynchronize();
        if (syncErr != cudaSuccess) {
            GFL_LOG_ERROR("[GPUFL] cudaDeviceSynchronize failed: ", cudaGetErrorString(syncErr));
        }

        // Verify context is still valid
        CUcontext currentCtx = nullptr;
        CUresult ctxRes = cuCtxGetCurrent(&currentCtx);
        if (ctxRes != CUDA_SUCCESS || currentCtx != this->ctx_) {
            GFL_LOG_ERROR("[GPUFL] Context mismatch or invalid! stored=", (void*)this->ctx_,
                         " current=", (void*)currentCtx, " cuResult=", ctxRes);
            // Try to use current context
            if (currentCtx) {
                const_cast<CuptiBackend*>(this)->ctx_ = currentCtx;
            }
        }

        // Stop PC Sampling
        CUpti_PCSamplingStopParams stopParams = {};
        stopParams.size = sizeof(CUpti_PCSamplingStopParams);
        stopParams.ctx = this->ctx_;
        stopParams.pPriv = nullptr;  // Ensure pPriv is initialized

        // WORKAROUND: cuptiPCSamplingStop causes segfaults on RTX 3090 (Ampere) and newer GPUs
        // The PC Sampling API appears to have reliability issues with Start/Stop on these architectures.
        // Skip Stop/Disable entirely to avoid crashes. The sampling will remain active but won't crash.
        GFL_LOG_DEBUG("[GPUFL] Skipping cuptiPCSamplingStop (causes crashes on RTX 3090/Ampere+)");
        GFL_LOG_DEBUG("[GPUFL] PC Sampling data collection not fully supported on this GPU architecture");
        GFL_LOG_DEBUG("[GPUFL] Kernel monitoring continues to work normally");

        // Since Stop/Disable crash, we can't collect PC sampling data safely on this GPU.
        // Kernel monitoring via Activity API continues to work normally.
        // PC sampling feature is disabled for RTX 3090 (Ampere) and newer architectures.
    }

    void CuptiBackend::enableProfilingFeatures() {
        GFL_LOG_DEBUG("Configuring PC Sampling using Profiling API...");

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
        configInfo[1].attributeData.samplingPeriodData.samplingPeriod = 5;  // 2^5 = 32 cycles (more frequent sampling)

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

        // Configure the data buffer for PC sampling collection
        const size_t maxPcs = 10000;
        const_cast<CuptiBackend*>(this)->pcDataBuffer_.resize(maxPcs);
        std::memset(const_cast<CuptiBackend*>(this)->pcDataBuffer_.data(), 0, sizeof(CUpti_PCSamplingPCData) * maxPcs);

        for (auto& pc : const_cast<CuptiBackend*>(this)->pcDataBuffer_) {
            pc.size = sizeof(CUpti_PCSamplingPCData);
        }

        const_cast<CuptiBackend*>(this)->samplingDataBuffer_ = {};
        const_cast<CuptiBackend*>(this)->samplingDataBuffer_.size = sizeof(CUpti_PCSamplingData);
        const_cast<CuptiBackend*>(this)->samplingDataBuffer_.collectNumPcs = maxPcs;
        const_cast<CuptiBackend*>(this)->samplingDataBuffer_.pPcData = const_cast<CuptiBackend*>(this)->pcDataBuffer_.data();

        CUpti_PCSamplingConfigurationInfo dataBufferInfo = {};
        dataBufferInfo.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_DATA_BUFFER;
        dataBufferInfo.attributeData.samplingDataBufferData.samplingDataBuffer = &const_cast<CuptiBackend*>(this)->samplingDataBuffer_;

        configParams.numAttributes = 1;
        configParams.pPCSamplingConfigurationInfo = &dataBufferInfo;
        GFL_LOG_DEBUG("[GPUFL] Configuring data buffer: address=",
                     (void*)&const_cast<CuptiBackend*>(this)->samplingDataBuffer_,
                     " pPcData=", (void*)const_cast<CuptiBackend*>(this)->samplingDataBuffer_.pPcData,
                     " collectNumPcs=", maxPcs);

        CUptiResult bufferRes = cuptiPCSamplingSetConfigurationAttribute(&configParams);
        if (bufferRes != CUPTI_SUCCESS) {
            const char* err; cuptiGetResultString(bufferRes, &err);
            GFL_LOG_ERROR("Data Buffer Config Failed: ", err, " (Code: ", bufferRes, ")");
        } else {
            GFL_LOG_DEBUG("[GPUFL] PC Sampling data buffer configured successfully");
        }

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
            GFL_LOG_DEBUG("[GPUFL] PC Sampling ENABLED successfully for ctx=", static_cast<void *>(this->ctx_));
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

                        {
                            const uint64_t corr = k->correlationId;
                            std::lock_guard<std::mutex> lk(backend->deviceMu_);
                            if (auto it = backend->deviceByCorr_.find(corr); it != backend->deviceByCorr_.end()) {
                                backend->deviceOrder_.erase(it->second.second);
                                backend->deviceOrder_.push_front(corr);
                                it->second = {k->deviceId, backend->deviceOrder_.begin()};
                            } else {
                                backend->deviceOrder_.push_front(corr);
                                backend->deviceByCorr_.emplace(corr, std::make_pair(k->deviceId, backend->deviceOrder_.begin()));
                                if (backend->deviceByCorr_.size() > CuptiBackend::kDeviceCorrMax) {
                                    const uint64_t old = backend->deviceOrder_.back();
                                    backend->deviceOrder_.pop_back();
                                    backend->deviceByCorr_.erase(old);
                                }
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
                        {
                            std::lock_guard<std::mutex> lk(backend->deviceMu_);
                            if (auto it = backend->deviceByCorr_.find(out.corrId); it != backend->deviceByCorr_.end()) {
                                out.deviceId = it->second.first;
                                backend->deviceOrder_.erase(it->second.second);
                                backend->deviceOrder_.push_front(out.corrId);
                                it->second.second = backend->deviceOrder_.begin();
                            } else {
                                out.deviceId = 0;
                            }
                        }

                        // Look up source file from sourceLocatorId
                        {
                            std::lock_guard<std::mutex> lk(sourceMapMu_);
                            if (auto it = sourceMap_.find(pc->sourceLocatorId); it != sourceMap_.end()) {
                                std::snprintf(out.sourceFile, sizeof(out.sourceFile), "%s",
                                            it->second.fileName.c_str());
                                out.sourceLine = it->second.lineNumber;
                                GFL_LOG_DEBUG("[PC_SAMPLING] Got sample: sourceFile=", out.sourceFile,
                                             ":", out.sourceLine, " samples=", out.samplesCount,
                                             " stallReason=", out.stallReason, " corrId=", out.corrId);
                            } else {
                                // Fallback to PC offset if source not found
                                uint64_t pcOffset = pc->pcOffset; // Copy to avoid packed field binding issue
                                uint32_t sourceLocId = pc->sourceLocatorId; // Copy to avoid packed field binding issue
                                std::snprintf(out.sourceFile, sizeof(out.sourceFile), "PC:0x%llx",
                                            (unsigned long long)pcOffset);
                                out.sourceLine = 0;
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
                // Driver API param structs differ from runtime API; avoid unsafe casts here.
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

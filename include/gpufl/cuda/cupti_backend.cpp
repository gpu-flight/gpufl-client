#include "gpufl/cuda/cupti_backend.hpp"
#include "gpufl/core/ring_buffer.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/trace_type.hpp"
#include "gpufl/core/debug_logger.hpp"

#include <iostream>
#include <cstring>

#define CUPTI_CHECK(call, failMsg) \
    do { \
        CUptiResult res = (call); \
        if (res != CUPTI_SUCCESS) { \
            ::gpufl::DebugLogger::error("[GPUFL Monitor] ", (failMsg)); \
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

    void CuptiBackend::Initialize(const MonitorOptions &opts) {
        opts_ = opts;
        DebugLogger::setEnabled(opts_.enable_debug_output);

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

    void CuptiBackend::Shutdown() {
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

    void CuptiBackend::Start() {
        if (!initialized_) return;
        active_.store(true);

        if (const CUptiResult res = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL); res != CUPTI_SUCCESS) {
            // Fallback to legacy if concurrent fails
            cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
        }

        GFL_LOG_DEBUG("Start complete");
    }

    void CuptiBackend::Stop() {
        if (!initialized_) return;
        active_.store(false);
        cuptiActivityFlushAll(1);
        cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL);
        cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
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
                    if (record->kind == CUPTI_ACTIVITY_KIND_KERNEL ||
                        record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {

                        const auto *k = reinterpret_cast<const
                            CUpti_ActivityKernel9 *>(record);

                        ActivityRecord out{};
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
        if (!backend || !backend->IsActive()) return;

        const char* funcName = cbInfo->functionName ? cbInfo->functionName : "unknown";
        const char* symbName = cbInfo->symbolName ? cbInfo->symbolName : "unknown";

        if (domain == CUPTI_CB_DOMAIN_RUNTIME_API || domain == CUPTI_CB_DOMAIN_DRIVER_API) {
            GFL_LOG_DEBUG("[DEBUG-CALLBACK] Domain=", (int)domain,
                          " CBID=", cbid,
                          " Name=", funcName,
                          " Symb=", symbName,
                          " CorrID=", cbInfo->correlationId);
        }
        if (domain == CUPTI_CB_DOMAIN_RESOURCE && cbid == CUPTI_CBID_RESOURCE_CONTEXT_CREATED) {
            GFL_LOG_DEBUG("[DEBUG-CALLBACK] Context Created! Enabling Runtime/Driver domains...");
            cuptiEnableDomain(1, backend->GetSubscriber(), CUPTI_CB_DOMAIN_RUNTIME_API);
            cuptiEnableDomain(1, backend->GetSubscriber(), CUPTI_CB_DOMAIN_DRIVER_API);
            return;
        }

        if (!backend->IsActive()) {
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

            if (backend->GetOptions().collect_kernel_details &&
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

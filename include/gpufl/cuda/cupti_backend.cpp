#include "gpufl/cuda/cupti_backend.hpp"
#include "gpufl/core/ring_buffer.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/trace_type.hpp"
#include "gpufl/cuda/cuda.hpp"

#include <iostream>
#include <cstring>

#define CUPTI_CHECK(call, failMsg) \
    do { \
        CUptiResult res = (call); \
        if (res != CUPTI_SUCCESS) { \
            std::cerr << (failMsg) << std::endl; \
        } \
    } while(0)

#define CUPTI_CHECK_RETURN(call, failMsg) \
    do { \
        CUptiResult res = (call); \
        if (res != CUPTI_SUCCESS) { \
            std::cerr << (failMsg) << std::endl; \
            return; \
        } \
    } while(0)

namespace gpufl {

    std::atomic<gpufl::CuptiBackend*> g_activeBackend{nullptr};

    // External ring buffer (defined in monitor.cpp)
    extern RingBuffer<ActivityRecord, 1024> g_monitorBuffer;

    void CuptiBackend::Initialize(const MonitorOptions &opts) {
        opts_ = opts;

        g_activeBackend.store(this, std::memory_order_release);
        if (opts_.enable_debug_output) {
            std::cout << "[GPUFL Monitor] Subscribing to CUPTI..." << std::endl;
        }
        CUPTI_CHECK_RETURN(
            cuptiSubscribe(&subscriber_, reinterpret_cast<CUpti_CallbackFunc>(GflCallback), this),
            "[GPUFL Monitor] ERROR: Failed to subscribe to CUPTI\n"
            "[GPUFL Monitor] This may indicate:\n"
            "  - CUPTI library not found or incompatible\n"
            "  - Insufficient permissions\n"
            "  - CUDA driver issues"
        );
        if (opts_.enable_debug_output) {
            std::cout << "[GPUFL Monitor] CUPTI subscription successful" << std::endl;
        }

        // Enable resource domain immediately to catch context creation
        cuptiEnableDomain(1, subscriber_, CUPTI_CB_DOMAIN_RESOURCE);
        cuptiEnableDomain(1, subscriber_, CUPTI_CB_DOMAIN_RUNTIME_API);
        cuptiEnableDomain(1, subscriber_, CUPTI_CB_DOMAIN_DRIVER_API);

        CUptiResult resCb = cuptiActivityRegisterCallbacks(BufferRequested, BufferCompleted);
        if (resCb != CUPTI_SUCCESS) {
            const char* errStr = nullptr;
            cuptiGetResultString(resCb, &errStr);
            std::cerr << "[GPUFL Monitor] FATAL: Failed to register activity callbacks." << std::endl;
            std::cerr << "[GPUFL Monitor] Error: " << (errStr ? errStr : "unknown")
                      << " (Code " << resCb << ")" << std::endl;

            initialized_ = false;
            return;
        }

        initialized_ = true;
        if (opts_.enable_debug_output) {
            std::cout << "[GPUFL Monitor] Callbacks registered successfully." << std::endl;
        }
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

        if (CUptiResult res = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL); res != CUPTI_SUCCESS) {
            // Fallback to legacy if concurrent fails
            cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
        }

        if (opts_.enable_debug_output) {
            std::cout << "[GPUFL Monitor] Start complete" << std::endl;
        }
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
            std::cerr << "[CUPTI] BufferCompleted: No active backend!" << std::endl;
            if (buffer) free(buffer);
            return;
        }

        if (backend->GetOptions().enable_debug_output) {
            std::cout << "[CUPTI] BufferCompleted validSize=" << validSize << std::endl;
        }
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
                        out.hasDetails = false;


                        // JOIN callback metadata by correlationId
                        std::cout << "[BufferCompleted] Processing kernel record with CorrID " << k->correlationId << std::endl;
                        const uint64_t corr = k->correlationId;
                        {
                            std::lock_guard lk(backend->metaMu_);
                            if (auto it = backend->metaByCorr_.find(corr); it != backend->metaByCorr_.end()) {
                                const LaunchMeta &m = it->second;

                                if (m.hasDetails) {
                                    out.hasDetails = true;
                                    out.gridX = m.gridX; out.gridY = m.gridY; out.gridZ = m.gridZ;
                                    out.blockX = m.blockX; out.blockY = m.blockY; out.blockZ = m.blockZ;
                                    out.dynShared = m.dynShared;
                                    out.staticShared = m.staticShared;
                                    out.localBytes = m.localBytes;
                                    out.constBytes = m.constBytes;
                                    out.numRegs = m.numRegs;
                                    out.occupancy = m.occupancy;
                                    out.maxActiveBlocks = m.maxActiveBlocks;
                                    std::cout << "[BufferCompleted] Found metadata for CorrID " << corr
                                              << " with occupancy=" << out.occupancy << std::endl;
                                } else {
                                    std::cout << "[BufferCompleted] Found metadata for CorrID " << corr
                                              << " but hasDetails=false" << std::endl;
                                }

                                backend->metaByCorr_.erase(it);
                            } else {
                                std::cout << "[BufferCompleted] No metadata found for CorrID " << corr << std::endl;
                            }
                        }

                        g_monitorBuffer.Push(out);
                    }
                } else if (st == CUPTI_ERROR_MAX_LIMIT_REACHED) {
                    // No more records in this buffer
                    break;
                } else {
                    std::cerr << "[CUPTI] Error parsing buffer: " << st << std::endl;
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
            std::cout << "[DEBUG-CALLBACK] Domain=" << (int)domain
                      << " CBID=" << cbid
                      << " Name=" << funcName
                      << " Symb=" << symbName
                      << " CorrID=" << cbInfo->correlationId
                      << std::endl;
        }
        if (domain == CUPTI_CB_DOMAIN_RESOURCE && cbid == CUPTI_CBID_RESOURCE_CONTEXT_CREATED) {
            std::cout << "[DEBUG-CALLBACK] Context Created! Enabling Runtime/Driver domains..." << std::endl;
            cuptiEnableDomain(1, backend->GetSubscriber(), CUPTI_CB_DOMAIN_RUNTIME_API);
            cuptiEnableDomain(1, backend->GetSubscriber(), CUPTI_CB_DOMAIN_DRIVER_API);
            return;
        }

        if (!backend->IsActive()) {
            std::cout << "[DEBUG-CALLBACK] Backend not active, skipping callback." << std::endl;
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
            std::cout << "[DEBUG-CALLBACK] >>> KERNEL LAUNCH DETECTED <<< (CorrID "
                      << cbInfo->correlationId << ")" << std::endl;
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

                cudaFuncAttributes attrs{};
                if (cudaFuncGetAttributes(&attrs, params->func) ==
                    cudaSuccess) {
                    meta.numRegs = attrs.numRegs;
                    meta.staticShared = static_cast<int>(attrs.sharedSizeBytes);
                    meta.localBytes = static_cast<int>(attrs.localSizeBytes);
                    meta.constBytes = static_cast<int>(attrs.constSizeBytes);

                    int dev = 0;
                    cudaGetDevice(&dev);
                    const auto &prop = cuda::getDevicePropsCached(dev);

                    const int blockSize =
                            meta.blockX * meta.blockY * meta.blockZ;

                    std::cout << "Block Size = " << blockSize << std::endl;
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                        &meta.maxActiveBlocks, params->func, blockSize,
                        meta.dynShared);
                    std::cout << "prop.maxThreadsPerMultiProcessor = " << prop.maxThreadsPerMultiProcessor << std::endl;

                    std::cout << "prop.warpSize = " << prop.warpSize << std::endl;
                    std::cout << "cbInfo->correlationId = " << cbInfo->correlationId << ", hasDetails = " << meta.hasDetails << std::endl;

                    if (prop.maxThreadsPerMultiProcessor > 0 && prop.warpSize >
                        0 && blockSize > 0) {
                        const int activeWarps =
                                meta.maxActiveBlocks * (
                                    blockSize / prop.warpSize);
                        const int maxWarps =
                                prop.maxThreadsPerMultiProcessor / prop.
                                warpSize;
                        meta.occupancy = (maxWarps > 0)
                                             ? static_cast<float>(activeWarps) /
                                               static_cast<float>(maxWarps)
                                             : 0.0f;
                        std::cout << "Calculated occupancy = " << meta.occupancy << std::endl;
                    }
                }
            }

            // Store by correlationId - atomically insert to prevent race with BufferCompleted
            {
                std::lock_guard<std::mutex> lk(backend->metaMu_);
                auto& existing = backend->metaByCorr_[cbInfo->correlationId];

                // If the existing entry has details, but the new one (e.g. from Driver API) does not,
                // KEEP the existing one. Do not overwrite it.
                if (existing.hasDetails && !meta.hasDetails) {
                    if (backend->GetOptions().enable_debug_output) {
                        std::cout << "[DEBUG-CALLBACK] Skipping overwrite of rich metadata for CorrID "
                                  << cbInfo->correlationId << " by Driver API." << std::endl;
                    }
                } else {
                    // Otherwise (it's new, or the new one has details and the old one didn't), update it.
                    existing = meta;

                    if (meta.hasDetails) {
                        std::cout << "[ENTER] Stored metadata for CorrID " << cbInfo->correlationId
                                  << " with occupancy=" << meta.occupancy << std::endl;
                    }
                }
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

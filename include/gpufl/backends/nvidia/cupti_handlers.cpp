#include "gpufl/backends/nvidia/cupti_handlers.hpp"
#include "gpufl/backends/nvidia/cupti_utils.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/stack_registry.hpp"
#include "gpufl/core/stack_trace.hpp"
#include "gpufl/core/scope_registry.hpp"
#include <cstdio>

namespace gpufl {

    // --- ResourceHandler ---

    ResourceHandler::ResourceHandler(CuptiBackend* backend) : backend_(backend) {}

    bool ResourceHandler::shouldHandle(CUpti_CallbackDomain domain, CUpti_CallbackId cbid) const {
        return domain == CUPTI_CB_DOMAIN_RESOURCE;
    }

    void ResourceHandler::handle(CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void* cbdata) {
        if (cbid == CUPTI_CBID_RESOURCE_MODULE_PROFILED) {
            auto *modData = static_cast<const CUpti_ModuleResourceData *>(cbdata);
            if (modData->cubinSize > 0) {
                GFL_LOG_DEBUG("[DEBUG-CALLBACK] cubin = ", modData->pCubin);
                GFL_LOG_DEBUG("[DEBUG-CALLBACK] cubinSize = ", modData->cubinSize);
            }
        }

        if (cbid == CUPTI_CBID_RESOURCE_MODULE_LOADED || cbid == CUPTI_CBID_RESOURCE_MODULE_PROFILED) {
            auto *modData = static_cast<const CUpti_ModuleResourceData *>(cbdata);
            const void* cubinPtr = nullptr;
            size_t cubinSize = 0;

            if (modData && modData->pCubin && modData->cubinSize > 0) {
                CUpti_GetCubinCrcParams params = {CUpti_GetCubinCrcParamsSize};
                cubinPtr = modData->pCubin;
                cubinSize = modData->cubinSize;
                params.cubinSize = cubinSize;
                params.cubin = cubinPtr;
                GFL_LOG_DEBUG("Attempting CRC for Cubin at ", cubinPtr, " Size: ", cubinSize);
                if (cuptiGetCubinCrc(&params) == CUPTI_SUCCESS) {
                    std::lock_guard<std::mutex> lk(backend_->cubinMu_);
                    auto& info = backend_->cubinByCrc_[params.cubinCrc];
                    info.crc = params.cubinCrc;
                    info.data.assign(reinterpret_cast<const uint8_t *>(cubinPtr),
                                   reinterpret_cast<const uint8_t *>(cubinPtr) + cubinSize);
                    GFL_LOG_DEBUG("[DEBUG-CALLBACK] Cubin SUCCESSFULLY stored: CRC=", params.cubinCrc, " Size=", cubinSize, " bytes ✓✓✓");
                } else {
                    GFL_LOG_ERROR("[DEBUG-CALLBACK] Failed to compute CRC for cubin");
                }
            }
        }
    }

    // --- KernelLaunchHandler ---

    KernelLaunchHandler::KernelLaunchHandler(CuptiBackend* backend) : backend_(backend) {}

    bool KernelLaunchHandler::shouldHandle(CUpti_CallbackDomain domain, CUpti_CallbackId cbid) const {
        return domain == CUPTI_CB_DOMAIN_RUNTIME_API || domain == CUPTI_CB_DOMAIN_DRIVER_API;
    }

    void KernelLaunchHandler::handle(CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void* cbdata) {
        if (!backend_->isActive()) return;

        auto *cbInfo = static_cast<const CUpti_CallbackData *>(cbdata);
        bool isKernelLaunch = false;

        if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
            if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020 ||
                cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) {
                isKernelLaunch = true;
            }
        } else if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
            if (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunch ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchGrid ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchGridAsync ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz) {
                isKernelLaunch = true;
            }
        }

        if (!isKernelLaunch) return;

        if (cbInfo->callbackSite == CUPTI_API_ENTER) {
            LaunchMeta meta{};
            meta.apiEnterNs = detail::getTimestampNs();

            const char *nm = cbInfo->symbolName ? cbInfo->symbolName : cbInfo->functionName;
            if (!nm) nm = "kernel_launch";
            std::snprintf(meta.name, sizeof(meta.name), "%s", nm);

            if (backend_->getOptions().enableStackTrace) {
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

            if (backend_->getOptions().collectKernelDetails && cbInfo->functionParams != nullptr) {
                if ((domain == CUPTI_CB_DOMAIN_RUNTIME_API && cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) ||
                    (domain == CUPTI_CB_DOMAIN_DRIVER_API && cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel)) {
                    meta.hasDetails = true;
                    const auto *params = (cudaLaunchKernel_v7000_params *) (cbInfo->functionParams);
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

            std::lock_guard<std::mutex> lk(backend_->metaMu_);
            auto& existing = backend_->metaByCorr_[cbInfo->correlationId];
            if (existing.hasDetails && !meta.hasDetails) {
                GFL_LOG_DEBUG("[DEBUG-CALLBACK] Skipping overwrite of rich metadata for CorrID ",
                              cbInfo->correlationId, " by Driver API.");
            } else {
                existing = meta;
            }
        } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
            const int64_t t = detail::getTimestampNs();
            std::lock_guard<std::mutex> lk(backend_->metaMu_);
            auto it = backend_->metaByCorr_.find(cbInfo->correlationId);
            if (it != backend_->metaByCorr_.end()) {
                it->second.apiExitNs = t;
            }
        }
    }

} // namespace gpufl

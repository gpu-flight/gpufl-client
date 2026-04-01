#include "gpufl/backends/nvidia/resource_handler.hpp"

#include <cupti_pcsampling.h>

#include "gpufl/backends/nvidia/cupti_utils.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/monitor.hpp"

namespace gpufl {

ResourceHandler::ResourceHandler(CuptiBackend *backend) : backend_(backend) {}

bool ResourceHandler::shouldHandle(CUpti_CallbackDomain domain,
                                   CUpti_CallbackId cbid) const {
    return domain == CUPTI_CB_DOMAIN_RESOURCE;
}

std::vector<CUpti_CallbackDomain> ResourceHandler::requiredDomains() const {
    return {CUPTI_CB_DOMAIN_RESOURCE};
}

std::vector<std::pair<CUpti_CallbackDomain, CUpti_CallbackId>>
ResourceHandler::requiredCallbacks() const {
    return {
        {CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_MODULE_LOADED},
        {CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_MODULE_PROFILED},
    };
}

void ResourceHandler::handle(CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
                             const void *cbdata) {
    auto resourceData = static_cast<const CUpti_ResourceData *>(cbdata);

    if (cbid == CUPTI_CBID_RESOURCE_MODULE_LOADED ||
        cbid == CUPTI_CBID_RESOURCE_MODULE_PROFILED) {
        auto *modData = static_cast<CUpti_ModuleResourceData *>(
            resourceData->resourceDescriptor);
        if (modData && modData->pCubin && modData->cubinSize > 0) {
            const void *cubinPtr = modData->pCubin;
            const size_t cubinSize = modData->cubinSize;

            {
                std::lock_guard<std::mutex> lk(backend_->cubin_mu_);
                if (backend_->seen_cubin_ptrs_.count(cubinPtr)) return;
                // CUPTI_CBID_RESOURCE_MODULE_PROFILED fires on every kernel
                // launch when PC sampling is active, including for
                // SASS-patched cubin variants that have a different pointer
                // than the original. Calling cuptiGetCubinCrc() from within
                // this callback deadlocks when SASS lazy patching holds
                // CUPTI-internal locks. Mark the pointer seen and bail out —
                // the original cubin was already processed by MODULE_LOADED.
                if (cbid == CUPTI_CBID_RESOURCE_MODULE_PROFILED) {
                    backend_->seen_cubin_ptrs_.insert(cubinPtr);
                    return;
                }
            }

            CUpti_GetCubinCrcParams params = {CUpti_GetCubinCrcParamsSize};
            params.cubinSize = cubinSize;
            params.cubin = cubinPtr;
            GFL_LOG_DEBUG("Attempting CRC for Cubin at ", cubinPtr,
                          " Size: ", cubinSize);
            if (cuptiGetCubinCrc(&params) == CUPTI_SUCCESS) {
                bool isNew = false;
                {
                    std::lock_guard<std::mutex> lk(backend_->cubin_mu_);
                    backend_->seen_cubin_ptrs_.insert(cubinPtr);
                    if (backend_->cubin_by_crc_.find(params.cubinCrc) ==
                        backend_->cubin_by_crc_.end()) {
                        auto &info = backend_->cubin_by_crc_[params.cubinCrc];
                        info.crc = params.cubinCrc;
                        info.data.assign(
                            static_cast<const uint8_t *>(cubinPtr),
                            static_cast<const uint8_t *>(cubinPtr) + cubinSize);
                        isNew = true;
                    }
                }
                if (isNew) {
                    Monitor::EnqueueCubinForDisassembly(
                        params.cubinCrc,
                        static_cast<const uint8_t *>(cubinPtr), cubinSize);
                }
            } else {
                GFL_LOG_ERROR(
                    "[DEBUG-CALLBACK] Failed to compute CRC for cubin");
            }
        }
    }
}

}  // namespace gpufl

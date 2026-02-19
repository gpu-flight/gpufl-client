#include "gpufl/backends/nvidia/resource_handler.hpp"
#include "gpufl/backends/nvidia/cupti_utils.hpp"
#include "gpufl/core/debug_logger.hpp"

namespace gpufl {

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

} // namespace gpufl

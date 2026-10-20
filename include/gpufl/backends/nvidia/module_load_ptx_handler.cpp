#include "gpufl/backends/nvidia/module_load_ptx_handler.hpp"

#include <cstddef>

#include "gpufl/core/monitor.hpp"

namespace gpufl {
namespace {

// cuModuleLoadData and cuLibraryLoadData take a NUL-terminated PTX string or a
// binary with no size; DictionaryManager stops at the NUL, at the first
// non-PTX byte, or here.
constexpr size_t kUnsizedImageLimit = 64u << 20;

}  // namespace

std::vector<std::pair<CUpti_CallbackDomain, CUpti_CallbackId>>
ModuleLoadPtxHandler::requiredCallbacks() const {
    return {
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuModuleLoadData},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuModuleLoadDataEx},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadData},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLinkAddData_v2},
    };
}

bool ModuleLoadPtxHandler::shouldHandle(CUpti_CallbackDomain domain,
                                        CUpti_CallbackId cbid) const {
    for (const auto& cb : requiredCallbacks()) {
        if (cb.first == domain && cb.second == cbid) return true;
    }
    return false;
}

void ModuleLoadPtxHandler::handle(CUpti_CallbackDomain domain,
                                  CUpti_CallbackId cbid, const void* cbdata) {
    if (!backend_ || domain != CUPTI_CB_DOMAIN_DRIVER_API) return;
    // PTX is shown beside SASS, so it is kept only where cubins are.
    if (!backend_->NeedsCubinCapture()) return;

    auto* cbInfo = static_cast<const CUpti_CallbackData*>(cbdata);
    if (!cbInfo || cbInfo->callbackSite != CUPTI_API_ENTER ||
        !cbInfo->functionParams)
        return;

    // Only the bytes are copied here; parsing waits for the PTX flush.
    switch (cbid) {
        case CUPTI_DRIVER_TRACE_CBID_cuModuleLoadData: {
            const auto* p = static_cast<const cuModuleLoadData_params*>(
                cbInfo->functionParams);
            Monitor::EnqueueLoadedPtx(p->image, kUnsizedImageLimit);
            break;
        }
        case CUPTI_DRIVER_TRACE_CBID_cuModuleLoadDataEx: {
            const auto* p = static_cast<const cuModuleLoadDataEx_params*>(
                cbInfo->functionParams);
            Monitor::EnqueueLoadedPtx(p->image, kUnsizedImageLimit);
            break;
        }
        case CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadData: {
            const auto* p = static_cast<const cuLibraryLoadData_params*>(
                cbInfo->functionParams);
            Monitor::EnqueueLoadedPtx(p->code, kUnsizedImageLimit);
            break;
        }
        case CUPTI_DRIVER_TRACE_CBID_cuLinkAddData_v2: {
            const auto* p = static_cast<const cuLinkAddData_v2_params*>(
                cbInfo->functionParams);
            if (p->type == CU_JIT_INPUT_PTX) Monitor::EnqueueLoadedPtx(p->data, p->size);
            break;
        }
        default:
            break;
    }
}

}  // namespace gpufl

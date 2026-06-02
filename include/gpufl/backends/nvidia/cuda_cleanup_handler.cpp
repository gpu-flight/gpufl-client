#include "gpufl/backends/nvidia/cuda_cleanup_handler.hpp"

#include "gpufl/core/scope_registry.hpp"

namespace gpufl {

std::vector<std::pair<CUpti_CallbackDomain, CUpti_CallbackId>>
CudaCleanupHandler::requiredCallbacks() const {
    return {
        {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020},
        {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaFreeArray_v3020},
        {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaFreeHost_v3020},
        {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaFreeMipmappedArray_v5000},
        {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020},
#if defined(CUPTI_RUNTIME_TRACE_CBID_cudaFreeAsync_v11020)
        {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaFreeAsync_v11020},
#endif
#if defined(CUPTI_RUNTIME_TRACE_CBID_cudaFreeAsync_ptsz_v11020)
        {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaFreeAsync_ptsz_v11020},
#endif
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemFree},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemFree_v2},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemFreeHost},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuCtxDestroy},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuCtxDestroy_v2},
#if defined(CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync)
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync},
#endif
#if defined(CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync_ptsz)
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync_ptsz},
#endif
    };
}

bool CudaCleanupHandler::shouldHandle(CUpti_CallbackDomain domain,
                                      CUpti_CallbackId cbid) const {
    for (const auto& cb : requiredCallbacks()) {
        if (cb.first == domain && cb.second == cbid) return true;
    }
    return false;
}

const char* CudaCleanupHandler::CleanupReason(CUpti_CallbackDomain domain,
                                              CUpti_CallbackId cbid) {
    if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
        switch (cbid) {
            case CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020: return "cudaFree";
            case CUPTI_RUNTIME_TRACE_CBID_cudaFreeArray_v3020: return "cudaFreeArray";
            case CUPTI_RUNTIME_TRACE_CBID_cudaFreeHost_v3020: return "cudaFreeHost";
            case CUPTI_RUNTIME_TRACE_CBID_cudaFreeMipmappedArray_v5000: return "cudaFreeMipmappedArray";
            case CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020: return "cudaDeviceReset";
#if defined(CUPTI_RUNTIME_TRACE_CBID_cudaFreeAsync_v11020)
            case CUPTI_RUNTIME_TRACE_CBID_cudaFreeAsync_v11020: return "cudaFreeAsync";
#endif
#if defined(CUPTI_RUNTIME_TRACE_CBID_cudaFreeAsync_ptsz_v11020)
            case CUPTI_RUNTIME_TRACE_CBID_cudaFreeAsync_ptsz_v11020: return "cudaFreeAsync_ptsz";
#endif
            default: return "runtime_cleanup";
        }
    }

    switch (cbid) {
        case CUPTI_DRIVER_TRACE_CBID_cuMemFree: return "cuMemFree";
        case CUPTI_DRIVER_TRACE_CBID_cuMemFree_v2: return "cuMemFree_v2";
        case CUPTI_DRIVER_TRACE_CBID_cuMemFreeHost: return "cuMemFreeHost";
        case CUPTI_DRIVER_TRACE_CBID_cuCtxDestroy: return "cuCtxDestroy";
        case CUPTI_DRIVER_TRACE_CBID_cuCtxDestroy_v2: return "cuCtxDestroy_v2";
#if defined(CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync)
        case CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync: return "cuMemFreeAsync";
#endif
#if defined(CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync_ptsz)
        case CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync_ptsz: return "cuMemFreeAsync_ptsz";
#endif
        default: return "driver_cleanup";
    }
}

void CudaCleanupHandler::handle(CUpti_CallbackDomain domain,
                                CUpti_CallbackId cbid, const void* cbdata) {
    if (!backend_) return;

    auto* cbInfo = static_cast<const CUpti_CallbackData*>(cbdata);
    if (!cbInfo || cbInfo->callbackSite != CUPTI_API_ENTER) return;

    // Only use cleanup APIs as an automatic final-flush boundary outside a
    // measured scope. Frees inside an active scope are part of user work and
    // should not trigger the mid-run SASS flush path we avoid for PyTorch.
    if (!getThreadScopeStack().empty()) return;

    backend_->FlushProfilingDataBeforeCudaTeardown(CleanupReason(domain, cbid));
}

}  // namespace gpufl

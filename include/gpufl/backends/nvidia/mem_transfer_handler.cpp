#include "gpufl/backends/nvidia/mem_transfer_handler.hpp"
#include "gpufl/backends/nvidia/cupti_utils.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/stack_registry.hpp"
#include "gpufl/core/stack_trace.hpp"
#include "gpufl/core/scope_registry.hpp"
#include <cstdio>

namespace gpufl {

    MemTransferHandler::MemTransferHandler(CuptiBackend* backend) : backend_(backend) {}

    bool MemTransferHandler::shouldHandle(CUpti_CallbackDomain domain, CUpti_CallbackId cbid) const {
        if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
            switch (cbid) {
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020:
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_ptds_v7000:
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020:
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_ptsz_v7000:
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2D_v3020:
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2D_ptds_v7000:
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DAsync_v3020:
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DAsync_ptsz_v7000:
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3D_v3020:
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3D_ptds_v7000:
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3DAsync_v3020:
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3DAsync_ptsz_v7000:
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyPeer_v4000:
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyPeerAsync_v4000:
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemset_v3020:
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemset_ptds_v7000:
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemsetAsync_v3020:
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemsetAsync_ptsz_v7000:
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemset2D_v3020:
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemset2D_ptds_v7000:
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemset2DAsync_v3020:
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemset2DAsync_ptsz_v7000:
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemset3D_v3020:
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemset3D_ptds_v7000:
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemset3DAsync_v3020:
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemset3DAsync_ptsz_v7000:
                    return true;
                default: return false;
            }
        } else if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
            switch (cbid) {
                case CUPTI_DRIVER_TRACE_CBID_cuMemcpy:
                case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2:
                case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2:
                case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2:
                case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoA_v2:
                case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoD_v2:
                case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoH_v2:
                case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoA_v2:
                case CUPTI_DRIVER_TRACE_CBID_cuMemcpy2D_v2:
                case CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DUnaligned_v2:
                case CUPTI_DRIVER_TRACE_CBID_cuMemcpy3D_v2:
                case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2:
                case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2:
                case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2:
                case CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DAsync_v2:
                case CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DAsync_v2:
                case CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeer:
                case CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeerAsync:
                case CUPTI_DRIVER_TRACE_CBID_cuMemsetD8_v2:
                case CUPTI_DRIVER_TRACE_CBID_cuMemsetD16_v2:
                case CUPTI_DRIVER_TRACE_CBID_cuMemsetD32_v2:
                case CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D8_v2:
                case CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D16_v2:
                case CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D32_v2:
                case CUPTI_DRIVER_TRACE_CBID_cuMemsetD8Async:
                case CUPTI_DRIVER_TRACE_CBID_cuMemsetD16Async:
                case CUPTI_DRIVER_TRACE_CBID_cuMemsetD32Async:
                case CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D8Async:
                case CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D16Async:
                case CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D32Async:
                    return true;
                default: return false;
            }
        }
        return false;
    }

    void MemTransferHandler::handle(CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void* cbdata) {

        GFL_LOG_DEBUG("Entering MEM event1");
        if (!backend_->isActive()) return;

        GFL_LOG_DEBUG("Entering MEM event2");

        auto *cbInfo = static_cast<const CUpti_CallbackData *>(cbdata);

        if (cbInfo->callbackSite == CUPTI_API_ENTER) {
            GFL_LOG_DEBUG("Entering MEM event");
            LaunchMeta meta{};
            meta.apiEnterNs = detail::getTimestampNs();

            const char *nm = cbInfo->functionName;
            if (!nm) nm = "mem_transfer";
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

            std::lock_guard<std::mutex> lk(backend_->metaMu_);
            backend_->metaByCorr_[cbInfo->correlationId] = meta;
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

#include "gpufl/backends/nvidia/mem_transfer_handler.hpp"

#include <cstdio>

#include "gpufl/backends/nvidia/cupti_utils.hpp"
#include "gpufl/core/activity_record.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/monitor.hpp"
#include "gpufl/core/ring_buffer.hpp"
#include "gpufl/core/scope_registry.hpp"
#include "gpufl/core/stack_registry.hpp"
#include "gpufl/core/stack_trace.hpp"

namespace gpufl {

MemTransferHandler::MemTransferHandler(CuptiBackend* backend)
    : backend_(backend) {}

std::vector<std::pair<CUpti_CallbackDomain, CUpti_CallbackId>>
MemTransferHandler::requiredCallbacks() const {
    return {
        // Runtime API memcpy
        {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020},
        {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_ptds_v7000},
        {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020},
        {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_ptsz_v7000},
        {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2D_v3020},
        {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2D_ptds_v7000},
        {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DAsync_v3020},
        {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DAsync_ptsz_v7000},
        {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3D_v3020},
        {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3D_ptds_v7000},
        {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3DAsync_v3020},
        {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3DAsync_ptsz_v7000},
        {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyPeer_v4000},
        {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyPeerAsync_v4000},
        // Runtime API memset
        {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMemset_v3020},
        {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMemset_ptds_v7000},
        {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMemsetAsync_v3020},
        {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMemsetAsync_ptsz_v7000},
        {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMemset2D_v3020},
        {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMemset2D_ptds_v7000},
        {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMemset2DAsync_v3020},
        {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMemset2DAsync_ptsz_v7000},
        {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMemset3D_v3020},
        {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMemset3D_ptds_v7000},
        {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMemset3DAsync_v3020},
        {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMemset3DAsync_ptsz_v7000},
        // Driver API memcpy
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemcpy},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoA_v2},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoD_v2},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoH_v2},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoA_v2},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemcpy2D_v2},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DUnaligned_v2},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemcpy3D_v2},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DAsync_v2},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DAsync_v2},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeer},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeerAsync},
        // Driver API memset
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemsetD8_v2},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemsetD16_v2},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemsetD32_v2},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D8_v2},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D16_v2},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D32_v2},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemsetD8Async},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemsetD16Async},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemsetD32Async},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D8Async},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D16Async},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D32Async},
    };
}

std::vector<CUpti_ActivityKind> MemTransferHandler::requiredActivityKinds()
    const {
    return {CUPTI_ACTIVITY_KIND_MEMCPY, CUPTI_ACTIVITY_KIND_MEMCPY2,
            CUPTI_ACTIVITY_KIND_MEMSET};
}

bool MemTransferHandler::shouldHandle(CUpti_CallbackDomain domain,
                                      CUpti_CallbackId cbid) const {
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
            default:
                return false;
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
            default:
                return false;
        }
    }
    return false;
}

void MemTransferHandler::handle(CUpti_CallbackDomain domain,
                                CUpti_CallbackId cbid, const void* cbdata) {
    GFL_LOG_DEBUG("Entering MEM event1");
    if (!backend_->IsActive()) return;

    GFL_LOG_DEBUG("Entering MEM event2");

    auto* cbInfo = static_cast<const CUpti_CallbackData*>(cbdata);

    if (cbInfo->callbackSite == CUPTI_API_ENTER) {
        GFL_LOG_DEBUG("Entering MEM event");
        LaunchMeta meta{};
        meta.api_enter_ns = detail::GetTimestampNs();

        const char* nm = cbInfo->functionName;
        if (!nm) nm = "mem_transfer";
        std::snprintf(meta.name, sizeof(meta.name), "%s", nm);

        if (backend_->GetOptions().enable_stack_trace) {
            const std::string trace = gpufl::core::GetCallStack(2);
            const std::string cleanTrace = detail::SanitizeStackTrace(trace);
            meta.stack_id =
                gpufl::StackRegistry::instance().getOrRegister(cleanTrace);
        } else {
            meta.stack_id = 0;
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
            std::snprintf(meta.user_scope, sizeof(meta.user_scope), "%s",
                          fullPath.c_str());
            meta.scope_depth = stack.size();
        } else {
            std::string fullPath = "global|";
            fullPath += meta.name;
            std::snprintf(meta.user_scope, sizeof(meta.user_scope), "%s",
                          fullPath.c_str());
            meta.scope_depth = 0;
        }

        std::lock_guard<std::mutex> lk(backend_->meta_mu_);
        backend_->meta_by_corr_[cbInfo->correlationId] = meta;
    } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
        const int64_t t = detail::GetTimestampNs();
        std::lock_guard<std::mutex> lk(backend_->meta_mu_);
        auto it = backend_->meta_by_corr_.find(cbInfo->correlationId);
        if (it != backend_->meta_by_corr_.end()) {
            it->second.api_exit_ns = t;
        }
    }
}

bool MemTransferHandler::handleActivityRecord(const CUpti_Activity* record,
                                              int64_t baseCpuNs,
                                              uint64_t baseCuptiTs) {
    if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY ||
        record->kind == CUPTI_ACTIVITY_KIND_MEMCPY2) {
        const auto* m = reinterpret_cast<const CUpti_ActivityMemcpy*>(record);
        ActivityRecord out{};
        out.device_id = m->deviceId;
        out.stream = static_cast<StreamHandle>(m->streamId);
        out.type = TraceType::MEMCPY;
        out.corr_id = m->correlationId;
        out.cpu_start_ns =
            baseCpuNs + static_cast<int64_t>(m->start - baseCuptiTs);
        out.duration_ns = static_cast<int64_t>(m->end - m->start);
        out.bytes = m->bytes;
        out.copy_kind = m->copyKind;
        out.src_kind = m->srcKind;
        out.dst_kind = m->dstKind;
        std::snprintf(out.name, sizeof(out.name), "memcpy");
        {
            std::lock_guard lk(backend_->meta_mu_);
            if (auto it = backend_->meta_by_corr_.find(out.corr_id);
                it != backend_->meta_by_corr_.end()) {
                const LaunchMeta& meta = it->second;
                out.scope_depth = meta.scope_depth;
                out.stack_id = meta.stack_id;
                std::copy(std::begin(meta.user_scope),
                          std::end(meta.user_scope),
                          std::begin(out.user_scope));
                out.api_start_ns = meta.api_enter_ns;
                out.api_exit_ns = meta.api_exit_ns;
                backend_->meta_by_corr_.erase(it);
            }
        }
        g_monitorBuffer.Push(out);
        return true;
    }

    if (record->kind == CUPTI_ACTIVITY_KIND_MEMSET) {
        const auto* m = reinterpret_cast<const CUpti_ActivityMemset*>(record);
        ActivityRecord out{};
        out.device_id = m->deviceId;
        out.stream = static_cast<StreamHandle>(m->streamId);
        out.type = TraceType::MEMSET;
        out.corr_id = m->correlationId;
        out.cpu_start_ns =
            baseCpuNs + static_cast<int64_t>(m->start - baseCuptiTs);
        out.duration_ns = static_cast<int64_t>(m->end - m->start);
        out.bytes = m->bytes;
        std::snprintf(out.name, sizeof(out.name), "memset");
        {
            std::lock_guard lk(backend_->meta_mu_);
            if (auto it = backend_->meta_by_corr_.find(out.corr_id);
                it != backend_->meta_by_corr_.end()) {
                const LaunchMeta& meta = it->second;
                out.scope_depth = meta.scope_depth;
                out.stack_id = meta.stack_id;
                std::copy(std::begin(meta.user_scope),
                          std::end(meta.user_scope),
                          std::begin(out.user_scope));
                out.api_start_ns = meta.api_enter_ns;
                out.api_exit_ns = meta.api_exit_ns;
                backend_->meta_by_corr_.erase(it);
            }
        }
        g_monitorBuffer.Push(out);
        return true;
    }

    return false;
}

}  // namespace gpufl

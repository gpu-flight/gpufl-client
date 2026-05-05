#include "gpufl/backends/nvidia/synchronization_handler.hpp"

#include "gpufl/core/common.hpp"   // detail::GetTimestampNs, detail::SanitizeStackTrace
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/stack_registry.hpp"
#include "gpufl/core/stack_trace.hpp"

namespace gpufl {

SynchronizationHandler::SynchronizationHandler(CuptiBackend* backend)
    : backend_(backend) {}

std::vector<std::pair<CUpti_CallbackDomain, CUpti_CallbackId>>
SynchronizationHandler::requiredCallbacks() const {
    // CUDA synchronization API CBIDs we capture stacks for. Add new
    // CBIDs here as CUPTI exposes them (e.g. cuStreamWaitValue32/64
    // are semaphore-style waits that can also produce SYNCHRONIZATION
    // activity records on newer CUPTI versions — confirm empirically
    // before adding).
    //
    // CUDART_VERSION guards keep the agent buildable against older
    // CUDA toolkits. The per-thread-default-stream (_ptsz) variants
    // were already in the public API at CUDA 7.0 so no guard needed.
    std::vector<std::pair<CUpti_CallbackDomain, CUpti_CallbackId>> cbs = {
        // ── Runtime API ───────────────────────────────────────────
        {CUPTI_CB_DOMAIN_RUNTIME_API,
         CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020},
        {CUPTI_CB_DOMAIN_RUNTIME_API,
         CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_ptsz_v7000},
        {CUPTI_CB_DOMAIN_RUNTIME_API,
         CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020},
        {CUPTI_CB_DOMAIN_RUNTIME_API,
         CUPTI_RUNTIME_TRACE_CBID_cudaEventSynchronize_v3020},
        {CUPTI_CB_DOMAIN_RUNTIME_API,
         CUPTI_RUNTIME_TRACE_CBID_cudaStreamWaitEvent_v3020},
        {CUPTI_CB_DOMAIN_RUNTIME_API,
         CUPTI_RUNTIME_TRACE_CBID_cudaStreamWaitEvent_ptsz_v7000},
        // ── Driver API ────────────────────────────────────────────
        {CUPTI_CB_DOMAIN_DRIVER_API,
         CUPTI_DRIVER_TRACE_CBID_cuStreamSynchronize},
        {CUPTI_CB_DOMAIN_DRIVER_API,
         CUPTI_DRIVER_TRACE_CBID_cuStreamSynchronize_ptsz},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuCtxSynchronize},
        {CUPTI_CB_DOMAIN_DRIVER_API,
         CUPTI_DRIVER_TRACE_CBID_cuEventSynchronize},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuStreamWaitEvent},
        {CUPTI_CB_DOMAIN_DRIVER_API,
         CUPTI_DRIVER_TRACE_CBID_cuStreamWaitEvent_ptsz},
    };
    return cbs;
}

std::vector<CUpti_ActivityKind>
SynchronizationHandler::requiredActivityKinds() const {
    // Sync activity records are enabled in cupti_backend.cpp directly
    // (CUPTI_ACTIVITY_KIND_SYNCHRONIZATION) — no per-handler activity
    // enable needed here. Returning an empty list is the established
    // convention for handlers that subscribe to callbacks only.
    return {};
}

bool SynchronizationHandler::shouldHandle(CUpti_CallbackDomain domain,
                                          CUpti_CallbackId cbid) const {
    // Mirror the CBID set registered in requiredCallbacks() — anything
    // missing here gets filtered before handle() runs and the
    // corresponding sync events arrive without a stack_id.
    if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
        switch (cbid) {
            case CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020:
            case CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_ptsz_v7000:
            case CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020:
            case CUPTI_RUNTIME_TRACE_CBID_cudaEventSynchronize_v3020:
            case CUPTI_RUNTIME_TRACE_CBID_cudaStreamWaitEvent_v3020:
            case CUPTI_RUNTIME_TRACE_CBID_cudaStreamWaitEvent_ptsz_v7000:
                return true;
            default:
                return false;
        }
    }
    if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
        switch (cbid) {
            case CUPTI_DRIVER_TRACE_CBID_cuStreamSynchronize:
            case CUPTI_DRIVER_TRACE_CBID_cuStreamSynchronize_ptsz:
            case CUPTI_DRIVER_TRACE_CBID_cuCtxSynchronize:
            case CUPTI_DRIVER_TRACE_CBID_cuEventSynchronize:
            case CUPTI_DRIVER_TRACE_CBID_cuStreamWaitEvent:
            case CUPTI_DRIVER_TRACE_CBID_cuStreamWaitEvent_ptsz:
                return true;
            default:
                return false;
        }
    }
    return false;
}

void SynchronizationHandler::handle(CUpti_CallbackDomain /*domain*/,
                                    CUpti_CallbackId cbid,
                                    const void* cbdata) {
    if (!backend_->IsActive()) return;

    auto* cbInfo = static_cast<const CUpti_CallbackData*>(cbdata);
    if (!cbInfo) {
        GFL_LOG_ERROR("[SynchronizationHandler] cbInfo is null");
        return;
    }

    if (cbInfo->callbackSite == CUPTI_API_ENTER) {
        // Capture the user's call stack now — by the time the matching
        // SYNCHRONIZATION activity record arrives on the buffer-flush
        // thread, this thread's stack is long gone.
        CuptiBackend::SyncMeta meta{};
        meta.api_enter_ns = detail::GetTimestampNs();

        if (backend_->GetOptions().enable_stack_trace) {
            const std::string trace = gpufl::core::GetCallStack(2);
            const std::string cleanTrace = detail::SanitizeStackTrace(trace);
            meta.stack_id =
                gpufl::StackRegistry::instance().getOrRegister(cleanTrace);
        } else {
            meta.stack_id = 0;
        }

        {
            std::lock_guard<std::mutex> lk(backend_->sync_meta_mu_);
            backend_->sync_meta_by_corr_[cbInfo->correlationId] = meta;
        }
        GFL_LOG_DEBUG("[SynchronizationHandler] API_ENTER corr=",
                      cbInfo->correlationId, " stack_id=", meta.stack_id,
                      " cbid=", cbid);
    }
    // Unlike kernels we don't need an API_EXIT recorder — sync wall
    // time is measured by CUPTI directly on the SYNCHRONIZATION
    // activity record (start/end fields), not derived from API
    // enter/exit timestamps. We still capture API_ENTER only because
    // that's where we have the user's stack.
}

bool SynchronizationHandler::handleActivityRecord(
        const CUpti_Activity* /*record*/, int64_t /*baseCpuNs*/,
        uint64_t /*baseCuptiTs*/) {
    // Sync activity records are processed inline in cupti_backend.cpp's
    // BufferCompleted (where the stack_id lookup against
    // sync_meta_by_corr_ also lives). Returning false here means the
    // dispatcher keeps walking the handler list and lets the inline
    // path handle the record.
    return false;
}

}  // namespace gpufl

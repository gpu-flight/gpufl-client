#pragma once

#include "gpufl/backends/nvidia/cupti_backend.hpp"
#include "gpufl/backends/nvidia/cupti_common.hpp"

namespace gpufl {

/**
 * Captures user call stacks for CUDA synchronization APIs
 * (cudaStreamSynchronize, cudaDeviceSynchronize, cudaEventSynchronize,
 * cuStreamWaitEvent and their variants).
 *
 * Why this exists:
 *   CUPTI's CUPTI_ACTIVITY_KIND_SYNCHRONIZATION records are delivered
 *   on the buffer-flush thread (background), so the user's call stack
 *   is no longer reachable at activity-record time. To attribute syncs
 *   to source code, we have to grab the stack on the API_ENTER
 *   callback (fired on the user's CPU thread) and join it back to the
 *   activity record by correlationId.
 *
 *   Mirrors KernelLaunchHandler's approach exactly — same callback /
 *   activity-join split, same StackRegistry interning, same
 *   {@code enable_stack_trace} opt-in. Only the CBID set differs.
 *
 * Storage:
 *   The captured stack_id (interned via StackRegistry) is stashed
 *   into CuptiBackend's `sync_meta_by_corr_` map under its own mutex
 *   (independent of the kernel-launch metadata map), and read out by
 *   the SYNCHRONIZATION branch of cupti_backend.cpp's BufferCompleted
 *   activity processor — which copies it into ActivityRecord.stack_id
 *   on its way to the monitor ring buffer. CollectorLoop later
 *   resolves the stack_id back into a real string via
 *   StackRegistry::get() and emits it as SynchronizationEvent.stack_trace.
 *
 * Activity records:
 *   This handler does NOT own a handleActivityRecord() implementation —
 *   sync activity processing stays in cupti_backend.cpp's
 *   BufferCompleted. The handler interface requires the override, so
 *   it returns false (= not handled) for any record.
 */
class SynchronizationHandler : public ICuptiHandler {
   public:
    explicit SynchronizationHandler(CuptiBackend* backend);

    const char* getName() const override { return "SynchronizationHandler"; }
    bool shouldHandle(CUpti_CallbackDomain domain,
                      CUpti_CallbackId cbid) const override;
    void handle(CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
                const void* cbdata) override;
    std::vector<std::pair<CUpti_CallbackDomain, CUpti_CallbackId>>
    requiredCallbacks() const override;
    std::vector<CUpti_ActivityKind> requiredActivityKinds() const override;
    bool handleActivityRecord(const CUpti_Activity* record, int64_t baseCpuNs,
                              uint64_t baseCuptiTs) override;

   private:
    CuptiBackend* backend_;
};

}  // namespace gpufl

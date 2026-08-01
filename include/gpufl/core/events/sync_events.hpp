#pragma once
#include <cstdint>
#include <string>

namespace gpufl {

// ── Batch row types (used by BatchBuffer, no heap strings) ────────────────

// One synchronization API call - `cudaStreamSynchronize` /
// `cudaDeviceSynchronize` / `cudaEventSynchronize` / `cuStreamWaitEvent`.
// Replaces the per-event `SynchronizationEvent` JSON with a packed row
// inside `synchronization_event_batch`. Cuts wire bytes ~14× on real
// workloads where the same call site fires repeatedly:
//   - The per-event envelope (type/pid/app/session_id) amortizes across
//     up to kMaxRows rows in the batch.
//   - `stack_trace` (typically 250+ bytes of nearly-identical text per
//     event in a hot loop) becomes a `function_id` interned via
//     `DictionaryManager::internFunction` and shipped exactly once per
//     unique stack via the existing `dictionary_update` flush.
struct SynchronizationEventBatchRow {
    int64_t  start_ns    = 0;   // absolute wall clock
    int64_t  duration_ns = 0;
    uint8_t  sync_type   = 0;   // CUpti_ActivitySynchronizationType (1..4)
    uint32_t stream_id   = 0;   // 0 = device-wide / context sync
    uint32_t event_id    = 0;   // 0 = no event handle
    uint32_t context_id  = 0;
    uint32_t corr_id     = 0;
    uint32_t function_id = 0;   // DictionaryManager::internFunction(stack_trace); 0 = no stack
};

/**
 * CUDA synchronization event captured by CUPTI.
 *
 * One event per cudaStreamSynchronize / cudaDeviceSynchronize /
 * cudaEventSynchronize / cuStreamWaitEvent call (and their driver-API
 * cousins). The wall-clock duration here is the CPU-side time the
 * thread was blocked - which is the exact metric that explains GPU
 * underutilization on workloads that interleave host-side python with
 * synchronous waits (PyTorch's `torch.cuda.synchronize()` between
 * forward / backward; eager-mode TF; manual debugging code).
 *
 * Per-event JSON (not batched). Volume is hundreds-to-thousands per
 * session in typical workloads - well within per-event capacity. If a
 * user runs a stress test that produces millions of syncs, switching
 * to a batched columnar format is a one-file change (mirrors the
 * KernelEventBatch pattern).
 *
 * `sync_type` is the integer from CUPTI's CUpti_ActivitySynchronizationType
 * enum; the dashboard renders it as a human label
 * (EventSynchronize / StreamWaitEvent / StreamSynchronize / ContextSynchronize).
 *
 * `corr_id` joins to KernelEvent.corr_id, letting the dashboard
 * answer questions like "this matmul kernel finished at T1; the
 * `cudaStreamSynchronize` waiting for it returned at T2 - that
 * (T2 - kernel_end) gap is host-side overhead, not GPU work."
 */
struct SynchronizationEvent {
    int pid = 0;
    std::string app;
    std::string session_id;
    int64_t start_ns = 0;
    int64_t end_ns = 0;
    int64_t duration_ns = 0;
    uint8_t  sync_type = 0;       // CUpti_ActivitySynchronizationType
    uint32_t stream_id = 0;       // 0 for context-wide / device sync
    uint32_t event_id = 0;        // 0 for non-event syncs
    uint32_t corr_id = 0;         // links to KernelEvent.corr_id
    uint32_t context_id = 0;
    // User call stack at the moment cudaStreamSynchronize / etc. fired.
    // Captured by SynchronizationHandler on the API_ENTER callback when
    // opts.enable_stack_trace is on, joined to the activity record by
    // correlationId. Mirrors KernelEvent.stack_trace - same string
    // format, same downstream wiring (backend stores as inline VARCHAR).
    // Empty when stack capture is disabled OR the launch API isn't in
    // SynchronizationHandler's CBID set.
    std::string stack_trace;
};

}  // namespace gpufl

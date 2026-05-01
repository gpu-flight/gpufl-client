#pragma once
#include <cstdint>

namespace gpufl {
enum class TraceType : uint8_t {
    KERNEL,
    PC_SAMPLE,
    SASS_METRIC,
    RANGE,
    MEMCPY,
    MEMSET,
    // NVTX range captured via CUPTI_ACTIVITY_KIND_MARKER. Fields used on
    // ActivityRecord: name (range name), cpu_start_ns, duration_ns.
    // Emitted as `nvtx_marker_event` in the NDJSON log.
    NVTX_MARKER,
    // Framework-emitted external correlation. Captured via
    // CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION; produced when a framework
    // (PyTorch / TF / JAX / XLA) brackets a code region with
    // cuptiActivityPushExternalCorrelationId / Pop. The record carries
    // the framework's op id + the CUPTI correlationId of the kernel(s)
    // launched inside that bracket. We DO NOT emit this as its own
    // NDJSON event — instead the (kind, id) is stamped onto the
    // matching kernel's ActivityRecord in BufferCompleted, and rides
    // along inside the existing `kernel_event_batch`. This trace type
    // exists so the dispatch loop has somewhere to file the record
    // type-wise even though it never reaches CollectorLoop.
    EXTERNAL_CORRELATION,
    // CUDA synchronization event captured via
    // CUPTI_ACTIVITY_KIND_SYNCHRONIZATION. One record per
    // cudaStreamSynchronize / cudaDeviceSynchronize / cudaEventSynchronize
    // / cuEventRecord / cuStreamWaitEvent. Volume is mid-scale —
    // hundreds-to-thousands per session, never the millions kernels
    // produce — so we emit per-event JSON (no batching).
    //
    // Fields used on ActivityRecord:
    //   cpu_start_ns  — sync wall start
    //   duration_ns   — wall duration of the sync op
    //   corr_id       — CUPTI correlationId, links to the kernel(s) the
    //                   sync was waiting on
    //   stream        — CUpti_ActivitySynchronization::streamId
    //   sync_type     — kind of sync (stream / device / event-record /
    //                   event-wait); see ActivityRecord::sync_type
    //   sync_event_id — for event-based syncs; 0 for stream/device
    SYNCHRONIZATION,
    // CUDA memory allocation event captured via
    // CUPTI_ACTIVITY_KIND_MEMORY2. One record per cudaMalloc / cudaFree
    // / cudaMallocAsync / cudaFreeAsync / cudaMallocManaged / cudaMallocHost.
    // Volume per session is mid-low (typically <1k events even for big
    // workloads — PyTorch's caching allocator absorbs most fine-grained
    // allocations into a few large CUDA-level blocks), so we emit per-event
    // JSON (no batching).
    //
    // Fields used on ActivityRecord:
    //   cpu_start_ns  — host call wall time
    //   bytes         — size of the allocation
    //   device_id     — target device (0 for host allocs)
    //   corr_id       — CUPTI correlationId, useful for joining to the
    //                   API call that triggered the alloc
    //   stream        — stream id for cudaMallocAsync; 0 otherwise
    //   memory_op     — alloc / free (uint8_t enum)
    //   memory_kind   — device / host / managed / pinned (uint8_t enum)
    //   address       — VA address of the allocation; for free, the
    //                   address being freed (lets us pair alloc/free
    //                   in a future v2 backend pass)
    MEMORY_ALLOC,
    // F4: CUDA graph launch event captured via
    // CUPTI_ACTIVITY_KIND_GRAPH_TRACE. One record per cudaGraphLaunch
    // call; CUPTI returns aggregate timing for the whole graph rather
    // than per-node, which is the point — graphs are *the* mechanism
    // CUDA uses to amortize launch overhead, so per-node tracing is
    // antithetical to their purpose.
    //
    // Volume per session is **very low** (typically tens to hundreds
    // even for a heavy training loop, since each graph captures many
    // kernels into one launch). Per-event JSON is fine.
    //
    // Fields used on ActivityRecord:
    //   cpu_start_ns  — graph execution start (wall clock)
    //   duration_ns   — end - start; 0 if CUPTI couldn't collect timing
    //   device_id     — device where the first node executes
    //   stream        — stream on which the graph was launched
    //   corr_id       — CUPTI correlationId; matches the driver API
    //                   record (cuGraphLaunch) but NOT the per-node
    //                   kernel records — those keep their own ids
    //   graph_id      — unique id of the graph (so repeated launches
    //                   of the same compiled graph share an id)
    GRAPH_LAUNCH,
};
}
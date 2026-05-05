#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "gpufl/core/stream_handle.hpp"
#include "gpufl/core/trace_type.hpp"

namespace gpufl {

struct ActivityRecord {
    uint32_t device_id = 0;
    char name[128]{};
    TraceType type = TraceType::KERNEL;
    StreamHandle stream = 0;
    int64_t cpu_start_ns = 0;
    int64_t api_start_ns = 0;
    int64_t api_exit_ns = 0;
    int64_t duration_ns = 0;

    // Detailed metrics (optional)
    bool has_details = false;
    int grid_x = 0, grid_y = 0, grid_z = 0;
    int block_x = 0, block_y = 0, block_z = 0;
    int dyn_shared = 0;
    int static_shared = 0;
    int local_bytes = 0;
    int const_bytes = 0;
    int num_regs = 0;
    int arch_vgpr_count = 0;  // AMD: architecture VGPRs only (for occupancy calc)
    float occupancy = 0.0f;

    // Per-resource occupancy breakdown
    float reg_occupancy = 0.0f;    // occupancy limited only by register file
    float smem_occupancy = 0.0f;   // occupancy limited only by shared memory
    float warp_occupancy = 0.0f;   // occupancy limited only by warp count
    float block_occupancy = 0.0f;  // occupancy limited only by block count
    char limiting_resource[16]{};  // "warps"|"registers"|"shared_mem"|"blocks"

    int max_active_blocks = 0;
    unsigned int corr_id = 0;

    // External-correlation stamp from CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION.
    // Populated in `KernelLaunchHandler::handleActivityRecord` by looking
    // up the kernel's `corr_id` in the global external-correlation map
    // (see g_extCorrMap in cupti_backend.cpp). external_id == 0 means no
    // framework was tracking this kernel.
    //
    // CUpti_ExternalCorrelationKind values (from <cupti_activity.h>):
    //   0 = INVALID, 1 = UNKNOWN, 2 = OPENACC, 3 = CUSTOM0..CUSTOM3,
    //   8 = CUSTOM_RESERVED1, ... (frameworks pick one).
    // We store as uint8_t to keep ActivityRecord small.
    uint8_t  external_kind = 0;
    uint64_t external_id = 0;

    char source_file[256]{};
    uint32_t source_line = 0;
    char function_name[256]{};
    char sample_kind[32]{};
    uint32_t samples_count = 0;
    uint32_t stall_reason = 0;
    std::string reason_name;
    char device_name[64]{};

    // SASS Metrics support
    uint32_t pc_offset = 0;
    uint64_t metric_value = 0;
    char metric_name[64]{};

    char user_scope[256]{};
    int scope_depth = 0;

    size_t stack_id = 0;

    // Phase 1a: additional CUpti_ActivityKernel11 fields
    uint32_t local_mem_total = 0;  // bytes across all threads
    uint32_t local_mem_per_thread = 0;  // bytes spilled per thread (0 = no spill)
    uint8_t cache_config_requested = 0;
    uint8_t cache_config_executed = 0;
    uint32_t shared_mem_executed = 0;

    // Memcpy / Memset specific
    uint64_t bytes = 0;
    uint32_t copy_kind = 0;  // backend-specific memcpy kind
    uint32_t src_kind = 0;   // backend-specific memory kind
    uint32_t dst_kind = 0;   // backend-specific memory kind

    // Graph-launch specific. graph_id is the CUPTI graphId field
    // (uint32 unique id of the captured graph). Same graph re-launched
    // multiple times in a training loop shares the same graph_id;
    // backend can aggregate by it to surface "this graph launched 50x,
    // total 4.2s" insights.
    uint32_t graph_id = 0;

    // Memory-allocation specific.
    //
    // memory_op: 1 = ALLOC, 2 = FREE. Mirrors CUpti_ActivityMemoryOperationType
    // (we collapse the variants we don't surface in v1 — release-async
    // and so on — into the two top-level buckets; the dashboard only
    // distinguishes alloc-vs-free in its first iteration).
    //
    // memory_kind values mirror CUpti_ActivityMemoryKind:
    //   0 = UNKNOWN, 1 = PAGEABLE_HOST, 2 = PINNED_HOST, 3 = DEVICE,
    //   4 = ARRAY, 5 = MANAGED, 6 = DEVICE_STATIC, 7 = MANAGED_STATIC.
    // Backend stores the raw integer; frontend maps to a label.
    //
    // address is the GPU virtual address (uint64) returned by cudaMalloc
    // (or freed by cudaFree). Stored as uint64 because Blackwell-class
    // addresses can be large; the dashboard renders it as 0x-prefixed
    // hex.
    uint8_t  memory_op = 0;
    uint8_t  memory_kind = 0;
    uint64_t address = 0;

    // Synchronization-event specific.
    //
    // sync_type encodes which CUPTI synchronization variant fired —
    // stored as uint8_t to keep ActivityRecord small. Values mirror
    // CUpti_ActivitySynchronizationType (cupti_activity.h):
    //   0 = UNKNOWN, 1 = EVENT_SYNCHRONIZE, 2 = STREAM_WAIT_EVENT,
    //   3 = STREAM_SYNCHRONIZE, 4 = CONTEXT_SYNCHRONIZE.
    // Mapping to user-readable names happens in SynchronizationEventModel
    // — backend stores the integer, frontend renders the label.
    //
    // sync_event_id is the CUPTI eventId (cudaEvent_t handle) for
    // EVENT_SYNCHRONIZE / STREAM_WAIT_EVENT records; zero for stream-
    // and context-wide syncs. Useful for grouping back-to-back waits
    // on the same event in the dashboard.
    uint8_t  sync_type = 0;
    uint32_t sync_event_id = 0;
    // CUDA context handle the sync executed on. Captured from
    // CUpti_ActivitySynchronization::contextId in the BufferCompleted
    // handler and forwarded to SynchronizationEvent.context_id by
    // CollectorLoop. (Was previously stashed in scope_depth — that
    // worked but blocked the legitimate use of scope_depth on
    // SYNCHRONIZATION records and obscured intent for readers.)
    uint32_t context_id = 0;
};

}  // namespace gpufl

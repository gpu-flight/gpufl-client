#pragma once
#include <cstdint>
#include <string>

namespace gpufl {

struct MemcpyEvent {
    int pid = 0;
    std::string app;
    std::string name;
    std::string platform;
    std::string session_id;
    uint32_t device_id = 0;
    uint32_t stream_id = 0;

    int64_t start_ns = 0;
    int64_t end_ns = 0;
    int64_t api_start_ns = 0;
    int64_t api_exit_ns = 0;

    unsigned int corr_id = 0;
    std::string user_scope;
    int scope_depth = 0;
    std::string stack_trace;

    uint64_t bytes = 0;
    std::string copy_kind;
    std::string src_kind;
    std::string dst_kind;
};

struct MemsetEvent {
    int pid = 0;
    std::string app;
    std::string name;
    std::string platform;
    std::string session_id;
    uint32_t device_id = 0;
    uint32_t stream_id = 0;

    int64_t start_ns = 0;
    int64_t end_ns = 0;
    int64_t api_start_ns = 0;
    int64_t api_exit_ns = 0;

    unsigned int corr_id = 0;
    std::string user_scope;
    int scope_depth = 0;
    std::string stack_trace;

    uint64_t bytes = 0;
};

// One CUPTI MEMORY2 record - `cudaMalloc` / `cudaFree` / `cudaMallocAsync` /
// etc. Replaces per-event `memory_alloc_event` JSON with a packed row
// inside `memory_alloc_event_batch`. Pure-numeric fields → no dictionary
// encoding, just envelope amortization. Saves ~85% on alloc-heavy
// workloads.
struct MemoryAllocEventBatchRow {
    int64_t  start_ns    = 0;
    int64_t  duration_ns = 0;   // 0 in v1 - CUPTI doesn't emit alloc duration
    uint8_t  memory_op   = 0;   // 1=ALLOC, 2=FREE
    uint8_t  memory_kind = 0;   // CUpti_ActivityMemoryKind
    uint64_t address     = 0;   // GPU virtual address
    uint64_t bytes       = 0;
    uint32_t device_id   = 0;
    uint32_t stream_id   = 0;
    uint32_t corr_id     = 0;
};

struct MemcpyBatchRow {
    int64_t  start_ns    = 0;
    uint32_t stream_id   = 0;
    int64_t  duration_ns = 0;
    uint64_t bytes       = 0;
    uint32_t copy_kind   = 0;  // numeric CUPTI/ROCprofiler kind value
    unsigned corr_id     = 0;
    uint32_t device_id   = 0;  // GPU that executed the transfer
};

/**
 * One CUDA memory-management event captured by CUPTI's
 * CUPTI_ACTIVITY_KIND_MEMORY2 stream.
 *
 * Covers cudaMalloc / cudaFree / cudaMallocAsync / cudaFreeAsync /
 * cudaMallocManaged / cudaMallocHost (and their driver-API cousins).
 * One event per call. Note that cudaMallocAsync is associated with a
 * stream and the reported {@code start_ns} is the host call time
 * (not the GPU completion time) - the host-side cost is what users
 * actually pay for in their python/c++ code.
 *
 * Per-event JSON. Volume in PyTorch workloads is typically <1k events
 * per session because torch's caching allocator absorbs most python-
 * level allocations; only large-block CUDA-level mallocs reach this
 * stream. TensorFlow eager mode is the high-volume edge case - if it
 * becomes a problem the gating flag {@code enable_memory_tracking}
 * lets users opt out without losing other CUPTI streams.
 *
 * The {@code address} field is the VA returned by cudaMalloc (or
 * being freed by cudaFree). Pairing alloc → free across the session
 * for leak / fragmentation analysis is a v2 follow-up; v1 just
 * stores raw events.
 */
struct MemoryAllocEvent {
    int pid = 0;
    std::string app;
    std::string session_id;
    int64_t start_ns = 0;
    int64_t duration_ns = 0;     // host-side; usually tiny but non-zero
    uint8_t  memory_op = 0;       // 1 = ALLOC, 2 = FREE
    uint8_t  memory_kind = 0;     // CUpti_ActivityMemoryKind
    uint64_t address = 0;
    uint64_t bytes = 0;
    uint32_t device_id = 0;
    uint32_t stream_id = 0;       // for cudaMallocAsync; 0 otherwise
    uint32_t corr_id = 0;
};

}  // namespace gpufl

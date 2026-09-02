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

// One CUPTI MEMORY2 or ROCprofiler memory-allocation record. Packed rows
// replace per-event `memory_alloc_event` JSON inside
// `memory_alloc_event_batch`. Pure-numeric fields need no dictionary
// encoding, which keeps allocation-heavy workloads compact.
struct MemoryAllocEventBatchRow {
    int64_t  start_ns    = 0;
    int64_t  duration_ns = 0;   // host-call duration when available
    uint8_t  memory_op   = 0;   // 1=ALLOC, 2=FREE
    uint8_t  memory_kind = 0;   // portable values; see ActivityRecord
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
 * One GPU memory-management event captured by CUPTI or ROCprofiler.
 *
 * CUDA covers allocation variants such as cudaMalloc, cudaFree, and their
 * asynchronous or managed-memory forms. AMD ROCprofiler reports ALLOCATE,
 * VMEM_ALLOCATE, FREE, and VMEM_FREE; the backend normalizes those to the
 * same portable alloc/free operation values.
 *
 * The reported timestamps describe the host call. The address and byte
 * count allow alloc/free pairing and future leak or fragmentation analysis.
 * Allocation tracking can be disabled with enable_memory_tracking without
 * disabling the other activity streams.
 */
struct MemoryAllocEvent {
    int pid = 0;
    std::string app;
    std::string session_id;
    int64_t start_ns = 0;
    int64_t duration_ns = 0;     // host-side; usually tiny but non-zero
    uint8_t  memory_op = 0;       // 1 = ALLOC, 2 = FREE
    uint8_t  memory_kind = 0;     // portable CUPTI-compatible kind value
    uint64_t address = 0;
    uint64_t bytes = 0;
    uint32_t device_id = 0;
    uint32_t stream_id = 0;       // for cudaMallocAsync; 0 otherwise
    uint32_t corr_id = 0;
};

}  // namespace gpufl

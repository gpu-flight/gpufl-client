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
};

}  // namespace gpufl

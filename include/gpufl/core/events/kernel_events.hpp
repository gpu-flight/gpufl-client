#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace gpufl {

struct KernelEvent {
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

    std::string grid;
    std::string block;
    bool has_details = false;
    int dyn_shared_bytes = 0;
    int num_regs = 0;
    std::size_t static_shared_bytes = 0;
    std::size_t local_bytes = 0;
    std::size_t const_bytes = 0;
    float occupancy = 0.0f;
    float reg_occupancy = 0.0f;
    float smem_occupancy = 0.0f;
    float warp_occupancy = 0.0f;
    float block_occupancy = 0.0f;
    std::string limiting_resource;
    int max_active_blocks = 0;
    unsigned int corr_id = 0;

    uint32_t local_mem_total = 0;        // total local mem across all threads (bytes)
    uint32_t local_mem_per_thread = 0;  // bytes spilled per thread (0 = no spill)

    uint8_t cache_config_requested = 0;
    uint8_t cache_config_executed = 0;
    uint32_t shared_mem_executed = 0;

    std::string user_scope;
    int scope_depth = 0;

    std::string stack_trace;

    // External correlation stamped onto this kernel by the framework
    // (PyTorch / TF / JAX). external_id == 0 means no framework tracked
    // this launch; kernel_event_model.cpp omits the columns when zero.
    uint8_t  external_kind = 0;
    uint64_t external_id   = 0;
};

struct KernelBatchRow {
    int64_t  start_ns    = 0;  // absolute GPU execution start
    uint32_t kernel_id   = 0;  // name dictionary ID
    uint32_t stream_id   = 0;  // raw CUDA stream ID
    int64_t  duration_ns = 0;
    unsigned corr_id     = 0;
    int      dyn_shared  = 0;
    int      num_regs    = 0;
    uint8_t  has_details = 0;  // 1 → a kernel_detail event follows with same corr_id

    // Framework-emitted external correlation, sourced from
    // CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION records. Stamped onto
    // the kernel by KernelLaunchHandler::handleActivityRecord; ferried
    // through the ActivityRecord into this row by CollectorLoop.
    // external_id == 0 means "no framework was tracking this kernel"
    // and the column is omitted from the JSON to keep the wire compact.
    uint8_t  external_kind = 0;
    uint64_t external_id   = 0;
};

struct KernelDetailRow {
    unsigned     corr_id = 0;
    std::string  session_id;
    int          pid = 0;
    std::string  app;
    int grid_x = 0, grid_y = 0, grid_z = 0;
    int block_x = 0, block_y = 0, block_z = 0;
    int  static_shared = 0;
    int  local_bytes   = 0;
    int  const_bytes   = 0;
    float occupancy       = 0.0f;
    float reg_occupancy   = 0.0f;
    float smem_occupancy  = 0.0f;
    float warp_occupancy  = 0.0f;
    float block_occupancy = 0.0f;
    char  limiting_resource[16]{};
    int   max_active_blocks      = 0;
    uint32_t local_mem_total     = 0;
    uint32_t local_mem_per_thread = 0;
    uint8_t  cache_config_requested = 0;
    uint8_t  cache_config_executed  = 0;
    uint32_t shared_mem_executed    = 0;
    std::string user_scope;
    std::string stack_trace;
};

}  // namespace gpufl

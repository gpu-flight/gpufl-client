#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gpufl {

struct ProfileSampleEvent {
    int pid = 0;
    std::string app;
    std::string session_id;

    int64_t ts_ns = 0;
    uint32_t device_id = 0;
    uint32_t corr_id = 0;
    uint32_t samples_count = 0;
    uint32_t stall_reason = 0;
    std::string reason_name;
    std::string sample_kind;  // "pc_sampling" | "sass_metric"

    std::string source_file;
    std::string function_name;
    uint32_t source_line = 0;

    // SASS Metrics
    std::string metric_name;
    uint64_t metric_value = 0;
    uint32_t pc_offset = 0;
};

struct ProfileSampleBatchRow {
    int64_t  ts_ns           = 0;
    uint32_t corr_id         = 0;
    uint32_t device_id       = 0;
    uint32_t function_id     = 0;   // function_dict ID
    uint32_t pc_offset       = 0;
    uint32_t metric_id       = 0;   // metric_dict ID (0 for pc_sampling)
    uint64_t metric_value    = 0;   // metric value (sass) or sample_count (pc)
    uint32_t stall_reason    = 0;   // pc_sampling only (0 for sass)
    uint8_t  sample_kind     = 0;   // 0 = pc_sampling, 1 = sass_metric
    uint32_t scope_name_id   = 0;   // scope_name_dict ID (0 = no scope)
    uint32_t source_file_id  = 0;   // source_file_dict ID (0 = unknown)
    uint32_t source_line     = 0;   // source line number (0 = unknown)
};

struct PmSampleBatchRow {
    uint32_t sample_index  = 0;
    int64_t  ts_ns         = 0;
    uint32_t device_id     = 0;
    uint32_t metric_id     = 0;   // metric_dict ID
    double   value         = 0.0;
    uint32_t scope_name_id = 0;   // scope_name_dict ID (0 = no scope)
};

struct PmSamplingConfigEvent {
    std::string session_id;
    int64_t ts_ns = 0;
    uint32_t device_id = 0;
    uint32_t interval_us = 0;
    uint32_t max_samples = 0;
    std::string preset;
    std::vector<std::string> metrics;
};

struct PerfMetricEvent {
    int pid = 0;
    std::string app;
    std::string session_id;
    std::string name;      // scope name
    int64_t start_ns = 0;
    int64_t end_ns = 0;
    int device_id = 0;

    // Hardware counters (-1/-1.0 = not available for this GPU/metric)
    double sm_throughput_pct = -1.0;   // SM active % of peak
    double l1_hit_rate_pct = -1.0;     // L1 global load hit rate
    double l2_hit_rate_pct = -1.0;     // L2 read hit rate
    int64_t dram_read_bytes = -1;      // DRAM read bytes
    int64_t dram_write_bytes = -1;     // DRAM write bytes
    double tensor_active_pct = -1.0;   // Tensor core active % (-1 if N/A)

    std::string user_scope;
    int scope_depth = 0;
};

struct KernelPerfMetricEvent {
    int pid = 0;
    std::string app;
    std::string session_id;
    int device_id = 0;
    size_t range_index = 0;
    std::string range_name;

    // Candidate join fields. KernelReplay auto-ranges usually expose a kernel
    // range name, but not the CUPTI activity correlation id.
    std::string kernel_name;
    uint32_t launch_ordinal = 0;

    double sm_throughput_pct = -1.0;
    double l1_hit_rate_pct = -1.0;
    double l2_hit_rate_pct = -1.0;
    int64_t dram_read_bytes = -1;
    int64_t dram_write_bytes = -1;
    double tensor_active_pct = -1.0;
    // Achieved (measured) occupancy as a percent 0-100
    // (sm__warps_active.avg.pct_of_peak_sustained_active) — the runtime
    // counterpart to the theoretical KernelEvent.occupancy computed from
    // launch config. -1.0 when not collected (only RangeProfilerKernelReplay
    // measures it). Note the scale: this is 0-100, KernelEvent.occupancy is 0-1.
    double achieved_occupancy_pct = -1.0;

    // Shared-memory bank-conflict counters from RangeProfilerKernelReplay.
    // Raw counts are -1 when unsupported. `shared_bank_conflict_overhead_pct`
    // is the fraction of shared wavefront work attributable to conflicts.
    // `shared_bank_conflict_nway` is the average serialization factor, where
    // 1.0 means conflict-free and 2.0 means two wavefronts per ideal request.
    int64_t shared_load_bank_conflicts = -1;
    int64_t shared_store_bank_conflicts = -1;
    int64_t shared_bank_conflicts = -1;
    int64_t shared_wavefronts = -1;
    double shared_bank_conflict_overhead_pct = -1.0;
    double shared_bank_conflict_nway = -1.0;
};

}  // namespace gpufl

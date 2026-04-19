#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace gpufl {
struct HostSample {
    double cpu_util_percent = 0.0;  // System-wide CPU usage (0.0 - 100.0)
    uint64_t ram_used_mib = 0;
    uint64_t ram_total_mib = 0;
};

struct GpuStaticDeviceInfo {
    int id = 0;
    std::string name;
    std::string uuid;
    std::string vendor;
    std::string architecture;
    int compute_major = 0;
    int compute_minor = 0;
    int l2_cache_size = 0;
    int shared_mem_per_block = 0;
    int regs_per_block = 0;
    int multi_processor_count = 0;
    int warp_size = 0;
};

struct DeviceSample {
    int device_id = 0;
    std::string name;
    std::string uuid;
    std::string vendor;
    int pci_bus_id = 0;

    size_t free_mib = 0;
    size_t total_mib = 0;
    size_t used_mib = 0;

    unsigned int gpu_util = 0;   // %
    unsigned int mem_util = 0;   // %
    unsigned int temp_c = 0;     // Celsius
    unsigned int power_mw = 0;   // Milliwatts
    unsigned int clock_gfx = 0;  // MHz
    unsigned int clock_sm = 0;   // MHz
    unsigned int clock_mem = 0;  // MHz

    // Extended metrics (AMD ROCm SMI)
    unsigned int fan_speed_pct = 0;    // Fan speed 0-100%
    unsigned int temp_mem_c = 0;       // Memory temperature, Celsius
    unsigned int temp_junction_c = 0;  // Junction temperature, Celsius
    unsigned int voltage_mv = 0;       // GFX voltage, millivolts
    uint64_t energy_uj = 0;            // Cumulative energy, microjoules
    uint64_t ecc_corrected = 0;        // Correctable ECC error count
    uint64_t ecc_uncorrected = 0;      // Uncorrectable ECC error count

    bool throttle_power;    // True if hitting Power CAp
    bool throttle_thermal;  // True if slowing down due to Heat

    unsigned long long nvlink_rx_bps;  // Receive Speed
    unsigned long long nvlink_tx_bps;  // Transmit Speed

    unsigned long long pcie_rx_bps;  // Host -> Device (Upload)
    unsigned long long pcie_tx_bps;  // Device -> Host (Download)
};

struct InitEvent {
    int pid = 0;
    std::string app;
    std::string session_id;
    std::string log_path;
    int64_t ts_ns = 0;
    HostSample host;
    std::vector<DeviceSample> devices;
    std::vector<GpuStaticDeviceInfo> gpu_static_device_infos;
};

struct ShutdownEvent {
    int pid = 0;
    std::string app;
    std::string session_id;
    int64_t ts_ns = 0;
};

struct SassConfigEvent {
    std::string session_id;
    int64_t ts_ns = 0;
    uint32_t device_id = 0;
    std::vector<std::string> configured_metrics;  // metrics successfully enabled
    std::vector<std::string> skipped_metrics;     // metrics CUPTI rejected for this GPU
};

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
};

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

struct ScopeBeginEvent {
    uint64_t scope_id = 0;
    int pid = 0;
    std::string app;
    std::string session_id;
    std::string name;
    std::string tag;
    int64_t ts_ns = 0;

    HostSample host;
    std::vector<DeviceSample> devices;

    std::string user_scope;
    int scope_depth = 0;
};

struct ScopeEndEvent {
    uint64_t scope_id = 0;
    int pid = 0;
    std::string app;
    std::string session_id;
    std::string name;
    std::string tag;
    int64_t ts_ns = 0;

    HostSample host;
    std::vector<DeviceSample> devices;

    std::string user_scope;
    int scope_depth = 0;
};

struct SystemStartEvent {
    int pid{};
    std::string app;
    std::string name;
    std::string session_id;
    int64_t ts_ns{};

    HostSample host;
    std::vector<DeviceSample> devices;
};

struct SystemSampleEvent {
    int pid = 0;
    std::string app;
    std::string session_id;
    std::string name;
    int64_t ts_ns = 0;

    HostSample host;
    std::vector<DeviceSample> devices;
};

struct SystemStopEvent {
    int pid{};
    std::string app;
    std::string session_id;
    std::string name;
    int64_t ts_ns{};

    HostSample host;
    std::vector<DeviceSample> devices;
};

// ── Batch row types (used by BatchBuffer, no heap strings) ────────────────

struct KernelBatchRow {
    int64_t  start_ns    = 0;  // absolute GPU execution start
    uint32_t kernel_id   = 0;  // name dictionary ID
    uint32_t stream_id   = 0;  // raw CUDA stream ID
    int64_t  duration_ns = 0;
    unsigned corr_id     = 0;
    int      dyn_shared  = 0;
    int      num_regs    = 0;
    uint8_t  has_details = 0;  // 1 → a kernel_detail event follows with same corr_id
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

struct MemcpyBatchRow {
    int64_t  start_ns    = 0;
    uint32_t stream_id   = 0;
    int64_t  duration_ns = 0;
    uint64_t bytes       = 0;
    uint32_t copy_kind   = 0;  // numeric CUpti kind value
    unsigned corr_id     = 0;
};

struct DeviceMetricBatchRow {
    int64_t  ts_ns     = 0;  // absolute timestamp
    int      device_id = 0;
    unsigned gpu_util  = 0;  // %
    unsigned mem_util  = 0;  // %
    unsigned temp_c    = 0;
    unsigned power_mw  = 0;
    uint64_t used_mib  = 0;
    uint64_t total_mib = 0;
    unsigned clock_sm  = 0;  // MHz
    // Extended metrics
    unsigned fan_speed_pct   = 0;  // %
    unsigned temp_mem_c      = 0;  // Celsius
    unsigned temp_junction_c = 0;  // Celsius
    unsigned voltage_mv      = 0;  // millivolts
    uint64_t energy_uj       = 0;  // cumulative microjoules
    unsigned clock_mem       = 0;  // MHz
    uint64_t pcie_bw_bps     = 0;  // bytes/sec (rx+tx combined)
    uint64_t ecc_corrected   = 0;
    uint64_t ecc_uncorrected = 0;
};

struct HostMetricBatchRow {
    int64_t  ts_ns         = 0;   // absolute timestamp
    uint32_t cpu_pct_x100  = 0;   // cpu_util_percent × 100 (2 decimal places)
    uint64_t ram_used_mib  = 0;
    uint64_t ram_total_mib = 0;
};

struct ScopeBatchRow {
    int64_t  ts_ns             = 0;  // absolute timestamp
    uint64_t scope_instance_id = 0;  // monotonic ID shared by begin/end pair
    uint32_t name_id           = 0;  // scope name dictionary ID
    uint8_t  event_type        = 0;  // 0 = begin, 1 = end
    int      depth             = 0;
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

struct PerfMetricEvent {
    int pid = 0;
    std::string app;
    std::string session_id;
    std::string name;      // scope name
    int64_t start_ns = 0;
    int64_t end_ns = 0;
    int device_id = 0;

    // Hardware counters (-1.0 = not available for this GPU/metric)
    double sm_throughput_pct = -1.0;   // SM active % of peak
    double l1_hit_rate_pct = -1.0;     // L1 global load hit rate
    double l2_hit_rate_pct = -1.0;     // L2 read hit rate
    uint64_t dram_read_bytes = 0;      // DRAM read bytes
    uint64_t dram_write_bytes = 0;     // DRAM write bytes
    double tensor_active_pct = -1.0;   // Tensor core active % (-1 if N/A)

    std::string user_scope;
    int scope_depth = 0;
};

/**
 * NVTX range captured via CUPTI_ACTIVITY_KIND_MARKER. Sources include:
 *   - GFL_SCOPE (which auto-emits nvtxRangePushA/Pop as of the PyTorch
 *     integration work)
 *   - PyTorch's automatic NVTX annotations (via torch.cuda.nvtx or our
 *     gpufl.torch.TorchDispatchMode wrapping)
 *   - cuDNN / cuBLAS / NCCL / TensorRT which emit NVTX internally
 *   - User-emitted nvtxRangePush / nvtxMarkA calls
 *
 * Paired START/END records from CUPTI are merged in the client before
 * emitting; one NvtxMarkerEvent per completed range.
 */
struct NvtxMarkerEvent {
    int pid = 0;
    std::string app;
    std::string session_id;
    std::string name;           // Range name (NVTX push argument)
    std::string domain;         // NVTX domain, "" for default
    int64_t start_ns = 0;
    int64_t end_ns = 0;
    int64_t duration_ns = 0;    // Redundant with end-start; kept for convenience
    uint32_t marker_id = 0;     // CUPTI marker ID (for debug / dedup)
};
}  // namespace gpufl

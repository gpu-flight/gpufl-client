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

struct CudaStaticDeviceInfo {
    int id;
    std::string name;
    std::string uuid;
    int compute_major;
    int compute_minor;
    int l2_cache_size;
    int shared_mem_per_block;
    int regs_per_block;
    int multi_processor_count;
    int warp_size;
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
    std::vector<CudaStaticDeviceInfo> cuda_static_device_infos;
};

struct ShutdownEvent {
    int pid = 0;
    std::string app;
    std::string session_id;
    int64_t ts_ns = 0;
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
}  // namespace gpufl

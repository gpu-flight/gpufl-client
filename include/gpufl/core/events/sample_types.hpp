#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

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

    // Extended device capabilities used by `gpufl info`. These remain out of
    // the job_start serializer until the backend adopts the expanded schema,
    // so adding them does not change the existing telemetry wire contract.
    uint64_t total_global_mem = 0;
    uint64_t total_const_mem = 0;
    int shared_mem_per_block_optin = 0;
    int shared_mem_per_multiprocessor = 0;
    int regs_per_multiprocessor = 0;
    int max_threads_per_block = 0;
    int max_threads_per_multiprocessor = 0;
    int max_blocks_per_multiprocessor = 0;
    std::array<int, 3> max_threads_dim{};
    std::array<int, 3> max_grid_size{};
    int clock_rate_khz = 0;
    int memory_clock_rate_khz = 0;
    int memory_bus_width_bits = 0;
    int async_engine_count = 0;
    bool concurrent_kernels = false;
    bool cooperative_launch = false;
    bool unified_addressing = false;
    bool managed_memory = false;
    bool memory_pools_supported = false;
    bool cluster_launch = false;
    bool tensor_map_access_supported = false;
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

}  // namespace gpufl

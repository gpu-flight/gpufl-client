#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "gpufl/core/events/sample_types.hpp"

namespace gpufl {

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

}  // namespace gpufl

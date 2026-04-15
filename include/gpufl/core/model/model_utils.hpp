#pragma once

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "gpufl/core/events.hpp"
#include "gpufl/core/json/json.hpp"

namespace gpufl::model {

/// Alias for backward compatibility — delegates to gpufl::json::escape().
inline std::string jsonEscape(const std::string& s) {
    return gpufl::json::escape(s);
}

inline std::string hostToJson(const HostSample& h) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);
    oss << "{\"cpu_pct\":" << h.cpu_util_percent
        << ",\"ram_used_mib\":" << h.ram_used_mib
        << ",\"ram_total_mib\":" << h.ram_total_mib << "}";
    return oss.str();
}

inline std::string devicesToJson(const std::vector<DeviceSample>& devs) {
    std::ostringstream oss;
    oss << "[";
    bool first = true;
    for (const auto& d : devs) {
        if (!first) oss << ",";
        first = false;
        oss << "{\"id\":" << d.device_id
            << ",\"name\":\""   << jsonEscape(d.name)   << "\""
            << ",\"uuid\":\""   << jsonEscape(d.uuid)   << "\""
            << ",\"vendor\":\"" << jsonEscape(d.vendor) << "\""
            << ",\"pci_bus\":"  << d.pci_bus_id
            << ",\"used_mib\":"  << d.used_mib
            << ",\"free_mib\":"  << d.free_mib
            << ",\"total_mib\":" << d.total_mib
            << ",\"util_gpu_pct\":"  << d.gpu_util
            << ",\"util_mem_pct\":"  << d.mem_util
            << ",\"temp_c\":"    << d.temp_c
            << ",\"power_mw\":"  << d.power_mw
            << ",\"clk_gfx_mhz\":"   << d.clock_gfx
            << ",\"clk_sm_mhz\":"    << d.clock_sm
            << ",\"clk_mem_mhz\":"   << d.clock_mem
            << ",\"throttle_pwr\":"   << (d.throttle_power   ? 1 : 0)
            << ",\"throttle_therm\":" << (d.throttle_thermal ? 1 : 0)
            << ",\"pcie_rx_bw_bps\":" << d.pcie_rx_bps
            << ",\"pcie_tx_bw_bps\":" << d.pcie_tx_bps << "}";
    }
    oss << "]";
    return oss.str();
}

inline std::string staticDevicesToJson(
    const std::vector<GpuStaticDeviceInfo>& devs) {
    std::ostringstream oss;
    oss << "[";
    bool first = true;
    for (const auto& d : devs) {
        if (!first) oss << ",";
        first = false;
        oss << "{\"id\":" << d.id
            << ",\"name\":\""          << jsonEscape(d.name) << "\""
            << ",\"uuid\":\""          << jsonEscape(d.uuid) << "\""
            << ",\"vendor\":\""        << jsonEscape(d.vendor) << "\""
            << ",\"architecture\":\""  << jsonEscape(d.architecture) << "\""
            << ",\"compute_major\":"   << d.compute_major
            << ",\"compute_minor\":"   << d.compute_minor
            << ",\"l2_cache_size_bytes\":"        << d.l2_cache_size
            << ",\"shared_mem_per_block_bytes\":"  << d.shared_mem_per_block
            << ",\"regs_per_block\":"              << d.regs_per_block
            << ",\"multi_processor_count\":"       << d.multi_processor_count
            << ",\"warp_size\":"       << d.warp_size << "}";
    }
    oss << "]";
    return oss.str();
}

inline std::string staticDevicesToJsonForVendor(
    const std::vector<GpuStaticDeviceInfo>& devs, const std::string& vendor) {
    std::ostringstream oss;
    oss << "[";
    bool first = true;
    for (const auto& d : devs) {
        if (d.vendor != vendor) continue;
        if (!first) oss << ",";
        first = false;
        oss << "{\"id\":" << d.id
            << ",\"name\":\""          << jsonEscape(d.name) << "\""
            << ",\"uuid\":\""          << jsonEscape(d.uuid) << "\""
            << ",\"vendor\":\""        << jsonEscape(d.vendor) << "\""
            << ",\"architecture\":\""  << jsonEscape(d.architecture) << "\""
            << ",\"compute_major\":"   << d.compute_major
            << ",\"compute_minor\":"   << d.compute_minor
            << ",\"l2_cache_size_bytes\":"        << d.l2_cache_size
            << ",\"shared_mem_per_block_bytes\":"  << d.shared_mem_per_block
            << ",\"regs_per_block\":"              << d.regs_per_block
            << ",\"multi_processor_count\":"       << d.multi_processor_count
            << ",\"warp_size\":"       << d.warp_size << "}";
    }
    oss << "]";
    return oss.str();
}

}  // namespace gpufl::model

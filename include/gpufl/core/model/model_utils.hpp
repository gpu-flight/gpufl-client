#pragma once

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "gpufl/core/events.hpp"

namespace gpufl::model {

inline std::string jsonEscape(const std::string& s) {
    std::ostringstream oss;
    for (unsigned char c : s) {
        switch (c) {
            case '\\': oss << "\\\\"; break;
            case '"':  oss << "\\\""; break;
            case '\n': oss << "\\n";  break;
            case '\r': oss << "\\r";  break;
            case '\t': oss << "\\t";  break;
            default:
                if (c < 0x20) oss << "\\u" << std::hex << std::setw(4)
                                  << std::setfill('0') << static_cast<int>(c)
                                  << std::dec;
                else          oss << c;
        }
    }
    return oss.str();
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
            << ",\"util_gpu\":"  << d.gpu_util
            << ",\"util_mem\":"  << d.mem_util
            << ",\"temp_c\":"    << d.temp_c
            << ",\"power_mw\":"  << d.power_mw
            << ",\"clk_gfx\":"   << d.clock_gfx
            << ",\"clk_sm\":"    << d.clock_sm
            << ",\"clk_mem\":"   << d.clock_mem
            << ",\"throttle_pwr\":"   << (d.throttle_power   ? 1 : 0)
            << ",\"throttle_therm\":" << (d.throttle_thermal ? 1 : 0)
            << ",\"pcie_rx_bw\":" << d.pcie_rx_bps
            << ",\"pcie_tx_bw\":" << d.pcie_tx_bps << "}";
    }
    oss << "]";
    return oss.str();
}

inline std::string cudaStaticDevicesToJson(
    const std::vector<CudaStaticDeviceInfo>& devs) {
    std::ostringstream oss;
    oss << "[";
    bool first = true;
    for (const auto& d : devs) {
        if (!first) oss << ",";
        first = false;
        oss << "{\"id\":" << d.id
            << ",\"name\":\""          << jsonEscape(d.name) << "\""
            << ",\"uuid\":\""          << jsonEscape(d.uuid) << "\""
            << ",\"compute_major\":\"" << d.compute_major    << "\""
            << ",\"compute_minor\":"   << d.compute_minor
            << ",\"l2_cache_size\":"   << d.l2_cache_size
            << ",\"shared_mem_per_block\":" << d.shared_mem_per_block
            << ",\"regs_per_block\":"       << d.regs_per_block
            << ",\"multi_processor_count\":" << d.multi_processor_count
            << ",\"warp_size\":"       << d.warp_size << "}";
    }
    oss << "]";
    return oss.str();
}

}  // namespace gpufl::model

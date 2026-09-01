#pragma once

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "gpufl/core/events.hpp"
#include "gpufl/core/json/json.hpp"

namespace gpufl::model {

/// Alias for backward compatibility - delegates to gpufl::json::escape().
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

inline void appendStringArray(std::ostringstream& oss,
                              const std::vector<std::string>& values) {
    oss << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) oss << ',';
        oss << '"' << jsonEscape(values[i]) << '"';
    }
    oss << ']';
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
            << ",\"pcie_tx_bw_bps\":" << d.pcie_tx_bps;
        const auto& capabilities = d.telemetry_capabilities;
        if (!capabilities.available.empty() ||
            !capabilities.unavailable.empty() ||
            !capabilities.memory_model.empty() ||
            !capabilities.allocation_scope.empty()) {
            oss << ",\"telemetry_capabilities\":{\"available\":";
            appendStringArray(oss, capabilities.available);
            oss << ",\"unavailable\":";
            appendStringArray(oss, capabilities.unavailable);
            oss << ",\"memory_model\":\""
                << jsonEscape(capabilities.memory_model) << "\""
                << ",\"allocation_scope\":\""
                << jsonEscape(capabilities.allocation_scope) << "\""
                << ",\"process_allocated_mib\":"
                << capabilities.process_allocated_mib
                << ",\"recommended_max_working_set_mib\":"
                << capabilities.recommended_max_working_set_mib << '}';
        }
        oss << '}';
    }
    oss << "]";
    return oss.str();
}

inline void appendStaticDeviceJson(std::ostringstream& oss,
                                   const GpuStaticDeviceInfo& d) {
    oss << "{\"id\":" << d.id << ",\"name\":\"" << jsonEscape(d.name) << "\""
        << ",\"uuid\":\"" << jsonEscape(d.uuid) << "\""
        << ",\"vendor\":\"" << jsonEscape(d.vendor) << "\""
        << ",\"architecture\":\"" << jsonEscape(d.architecture) << "\""
        << ",\"compute_major\":" << d.compute_major
        << ",\"compute_minor\":" << d.compute_minor
        << ",\"l2_cache_size_bytes\":" << d.l2_cache_size
        << ",\"shared_mem_per_block_bytes\":" << d.shared_mem_per_block
        << ",\"regs_per_block\":" << d.regs_per_block
        << ",\"multi_processor_count\":" << d.multi_processor_count
        << ",\"warp_size\":" << d.warp_size;

    if (d.metal.available) {
        oss << ",\"metal\":{\"registry_id\":\""
            << jsonEscape(d.metal.registry_id) << "\""
            << ",\"architecture_name\":\""
            << jsonEscape(d.metal.architecture_name) << "\""
            << ",\"low_power\":" << (d.metal.low_power ? "true" : "false")
            << ",\"headless\":" << (d.metal.headless ? "true" : "false")
            << ",\"removable\":" << (d.metal.removable ? "true" : "false")
            << ",\"unified_memory\":"
            << (d.metal.unified_memory ? "true" : "false") << ",\"location\":\""
            << jsonEscape(d.metal.location) << "\""
            << ",\"location_number\":" << d.metal.location_number
            << ",\"recommended_max_working_set_bytes\":"
            << d.metal.recommended_max_working_set_bytes
            << ",\"max_transfer_rate_bps\":" << d.metal.max_transfer_rate_bps
            << ",\"max_buffer_length_bytes\":"
            << d.metal.max_buffer_length_bytes
            << ",\"max_threads_per_threadgroup\":["
            << d.metal.max_threads_per_threadgroup[0] << ','
            << d.metal.max_threads_per_threadgroup[1] << ','
            << d.metal.max_threads_per_threadgroup[2] << "],\"gpu_families\":";
        appendStringArray(oss, d.metal.gpu_families);
        oss << ",\"counter_sets\":";
        appendStringArray(oss, d.metal.counter_sets);
        oss << '}';
    }
    oss << '}';
}

inline std::string staticDevicesToJson(
    const std::vector<GpuStaticDeviceInfo>& devs) {
    std::ostringstream oss;
    oss << "[";
    bool first = true;
    for (const auto& d : devs) {
        if (!first) oss << ",";
        first = false;
        appendStaticDeviceJson(oss, d);
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
        appendStaticDeviceJson(oss, d);
    }
    oss << "]";
    return oss.str();
}

}  // namespace gpufl::model

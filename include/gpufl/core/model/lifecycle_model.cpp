#include "gpufl/core/model/lifecycle_model.hpp"

#include <sstream>

#include "gpufl/core/host_info.hpp"
#include "gpufl/core/model/model_utils.hpp"

namespace gpufl::model {

std::string InitEventModel::buildJson() const {
    // Resolve once per call. Both fields are session-level metadata —
    // included at the top of `job_start` so a file-tailing agent can
    // associate every subsequent batch with the right host without
    // having to resolve the hostname itself (which would be wrong if
    // the agent runs on a different machine than the workload).
    const std::string hostname = gpufl::getLocalHostname();
    const std::string ipAddr   = gpufl::getLocalIpAddr();

    std::ostringstream oss;
    oss << "{\"version\":1,\"type\":\"job_start\""
        << ",\"pid\":"         << e_.pid
        << ",\"app\":\""       << jsonEscape(e_.app)        << "\""
        << ",\"session_id\":\"" << jsonEscape(e_.session_id) << "\""
        << ",\"log_path\":\""  << jsonEscape(e_.log_path)   << "\""
        << ",\"ts_ns\":"       << e_.ts_ns
        << ",\"hostname\":\""  << jsonEscape(hostname)      << "\""
        << ",\"ip_addr\":\""   << jsonEscape(ipAddr)        << "\""
        << ",\"host\":"        << hostToJson(e_.host)
        << ",\"devices\":"     << devicesToJson(e_.devices)
        << ",\"gpu_static_devices\":" << staticDevicesToJson(e_.gpu_static_device_infos)
        << ",\"cuda_static_devices\":"
        << staticDevicesToJsonForVendor(e_.gpu_static_device_infos, "NVIDIA")
        << ",\"rocm_static_devices\":"
        << staticDevicesToJsonForVendor(e_.gpu_static_device_infos, "AMD")
        << "}";
    return oss.str();
}

std::string ShutdownEventModel::buildJson() const {
    std::ostringstream oss;
    oss << "{\"type\":\"shutdown\""
        << ",\"pid\":"          << e_.pid
        << ",\"app\":\""        << jsonEscape(e_.app)        << "\""
        << ",\"session_id\":\"" << jsonEscape(e_.session_id) << "\""
        << ",\"ts_ns\":"        << e_.ts_ns << "}";
    return oss.str();
}

std::string SassConfigModel::buildJson() const {
    std::ostringstream oss;
    oss << "{\"version\":1,\"type\":\"sass_config\""
        << ",\"session_id\":\"" << jsonEscape(e_.session_id) << "\""
        << ",\"ts_ns\":"        << e_.ts_ns
        << ",\"device_id\":"    << e_.device_id
        << ",\"configured_metrics\":[";
    for (size_t i = 0; i < e_.configured_metrics.size(); ++i) {
        if (i) oss << ',';
        oss << "\"" << jsonEscape(e_.configured_metrics[i]) << "\"";
    }
    oss << "],\"skipped_metrics\":[";
    for (size_t i = 0; i < e_.skipped_metrics.size(); ++i) {
        if (i) oss << ',';
        oss << "\"" << jsonEscape(e_.skipped_metrics[i]) << "\"";
    }
    oss << "]}";
    return oss.str();
}

}  // namespace gpufl::model

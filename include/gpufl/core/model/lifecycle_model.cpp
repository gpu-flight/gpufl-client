#include "gpufl/core/model/lifecycle_model.hpp"

#include <sstream>

#include "gpufl/core/host_info.hpp"
#include "gpufl/core/model/model_utils.hpp"

namespace gpufl::model {

std::string InitEventModel::buildJson() const {
    // Resolve once per call. Both fields are session-level metadata -
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
        << staticDevicesToJsonForVendor(e_.gpu_static_device_infos, "AMD");

    oss << ",\"session_kind\":\"" << jsonEscape(e_.session_kind) << "\"";
    if (!e_.profiling_engine.empty()) {
        oss << ",\"profiling_engine\":\"" << jsonEscape(e_.profiling_engine) << "\"";
    }

    // Multi-pass grouping - emitted together, only for multi-pass runs.
    // A single-pass run leaves analysis_id empty and the job_start wire is
    // byte-identical to pre-P1 (so pass_index==0 is never confused with unset).
    if (!e_.analysis_id.empty()) {
        oss << ",\"analysis_id\":\"" << jsonEscape(e_.analysis_id) << "\""
            << ",\"pass_index\":" << e_.pass_index
            << ",\"pass_count\":" << e_.pass_count;
    }

    oss << "}";
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

std::string ExecutionSignatureModel::buildJson() const {
    std::ostringstream oss;
    oss << "{\"version\":1,\"type\":\"execution_signature\""
        << ",\"session_id\":\"" << jsonEscape(e_.session_id) << "\""
        << ",\"ts_ns\":"        << e_.ts_ns
        << ",\"scope_name\":\"" << jsonEscape(e_.scope_name) << "\""
        // signature is a full-width uint64 hash - emit as a STRING so a JSON
        // number consumer (JS doubles lose precision above 2^53) can't corrupt
        // it. The backend parses it back to an unsigned 64-bit value.
        << ",\"signature\":\""  << e_.signature << "\""
        << ",\"launch_count\":" << e_.launch_count
        << ",\"distinct_kernels\":" << e_.distinct_kernels
        << "}";
    return oss.str();
}

std::string CaptureCapabilitiesModel::buildJson() const {
    std::ostringstream oss;
    oss << "{\"version\":1,\"type\":\"capture_capabilities\""
        << ",\"session_id\":\"" << jsonEscape(e_.session_id) << "\""
        << ",\"ts_ns\":" << e_.ts_ns
        << ",\"requested_engine\":\"" << jsonEscape(e_.requested_engine) << "\""
        << ",\"selected_engine\":\"" << jsonEscape(e_.selected_engine) << "\""
        << ",\"capabilities\":[";
    for (size_t i = 0; i < e_.capabilities.size(); ++i) {
        const auto& c = e_.capabilities[i];
        if (i) oss << ',';
        oss << "{\"feature\":\"" << jsonEscape(c.feature) << "\""
            << ",\"requested\":" << (c.requested ? "true" : "false")
            << ",\"status\":\"" << jsonEscape(c.status) << "\""
            << ",\"mode\":\"" << jsonEscape(c.mode) << "\""
            << ",\"reason_code\":\"" << jsonEscape(c.reason_code) << "\""
            << ",\"message\":\"" << jsonEscape(c.message) << "\"}";
    }
    oss << "]}";
    return oss.str();
}

}  // namespace gpufl::model

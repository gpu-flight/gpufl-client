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
    if (!e_.telemetry_backend.empty()) {
        oss << ",\"telemetry_backend\":\"" << jsonEscape(e_.telemetry_backend)
            << "\"";
    }
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

    // Segmentation grouping is also all-or-nothing. run_id is the presence
    // discriminator so segment_index==0 remains a valid first segment rather
    // than being confused with an unset value.
    if (!e_.run_id.empty()) {
        oss << ",\"run_id\":\"" << jsonEscape(e_.run_id) << "\""
            << ",\"segment_index\":" << e_.segment_index;
    }

    // Roll-chain identity is gated on its own key, not run_id: a segmented run
    // that never rolled has run_id set but no chain, and must not grow the wire.
    // part_index rides with roll_chain_id (1-based, so it is never a false 0);
    // previous_run_id is omitted for the first part rather than sent as null.
    if (!e_.roll_chain_id.empty()) {
        oss << ",\"roll_chain_id\":\"" << jsonEscape(e_.roll_chain_id) << "\""
            << ",\"part_index\":" << e_.part_index;
        if (!e_.previous_run_id.empty()) {
            oss << ",\"previous_run_id\":\""
                << jsonEscape(e_.previous_run_id) << "\"";
        }
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

std::string SegmentStartEventModel::buildJson() const {
    std::ostringstream oss;
    oss << "{\"version\":1,\"type\":\"segment_start\""
        << ",\"session_id\":\"" << jsonEscape(e_.session_id) << "\""
        << ",\"run_id\":\"" << jsonEscape(e_.run_id) << "\""
        << ",\"segment_index\":" << e_.segment_index
        << ",\"ts_ns\":" << e_.ts_ns
        << ",\"actual_start_ns\":" << e_.actual_start_ns
        << ",\"previous_session_id\":";
    if (e_.previous_session_id.empty()) {
        oss << "null";
    } else {
        oss << "\"" << jsonEscape(e_.previous_session_id) << "\"";
    }
    oss << ",\"requested_boundary_ns\":";
    if (e_.has_requested_boundary) {
        oss << e_.requested_boundary_ns;
    } else {
        oss << "null";
    }
    oss << ",\"boundary_delay_ns\":" << e_.boundary_delay_ns
        << ",\"deferred_by\":";
    if (e_.deferred_by.empty()) {
        oss << "null";
    } else {
        oss << "\"" << jsonEscape(e_.deferred_by) << "\"";
    }
    oss << "}";
    return oss.str();
}

std::string SegmentEndEventModel::buildJson() const {
    std::ostringstream oss;
    oss << "{\"version\":1,\"type\":\"segment_end\""
        << ",\"session_id\":\"" << jsonEscape(e_.session_id) << "\""
        << ",\"run_id\":\"" << jsonEscape(e_.run_id) << "\""
        << ",\"segment_index\":" << e_.segment_index
        << ",\"ts_ns\":" << e_.ts_ns
        << ",\"actual_end_ns\":" << e_.actual_end_ns
        << ",\"requested_boundary_ns\":";
    if (e_.has_requested_boundary) {
        oss << e_.requested_boundary_ns;
    } else {
        oss << "null";
    }
    oss << ",\"boundary_delay_ns\":" << e_.boundary_delay_ns
        << ",\"end_reason\":\"" << jsonEscape(e_.end_reason) << "\""
        << ",\"deferred_by\":";
    if (e_.deferred_by.empty()) {
        oss << "null";
    } else {
        oss << "\"" << jsonEscape(e_.deferred_by) << "\"";
    }
    oss << ",\"records_outside_segment_window\":"
        << e_.records_outside_segment_window
        << "}";
    return oss.str();
}

std::string RunEndEventModel::buildJson() const {
    std::ostringstream oss;
    oss << "{\"version\":1,\"type\":\"run_end\""
        << ",\"session_id\":\"" << jsonEscape(e_.session_id) << "\""
        << ",\"run_id\":\"" << jsonEscape(e_.run_id) << "\""
        << ",\"final_segment_index\":" << e_.final_segment_index
        << ",\"ts_ns\":" << e_.ts_ns
        << ",\"ended_ns\":" << e_.ended_ns;
    // Only a rolled run carries rollover provenance; a shutdown run_end is
    // byte-identical to the pre-rollover wire.
    if (e_.end_reason == "rolled") {
        oss << ",\"end_reason\":\"" << jsonEscape(e_.end_reason) << "\""
            << ",\"rollover_reason\":\""
            << jsonEscape(e_.rollover_reason) << "\""
            << ",\"requested_rollover_ns\":" << e_.requested_rollover_ns
            << ",\"actual_rollover_ns\":" << e_.actual_rollover_ns;
    }
    oss << "}";
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

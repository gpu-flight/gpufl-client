#include "gpufl/core/model/memcpy_event_model.hpp"

#include <sstream>

#include "gpufl/core/model/model_utils.hpp"

namespace gpufl::model {

std::string MemcpyEventModel::buildJson() const {
    std::ostringstream oss;
    oss << "{\"type\":\"memcpy_event\""
        << ",\"pid\":"           << e_.pid
        << ",\"app\":\""         << jsonEscape(e_.app)        << "\""
        << ",\"session_id\":\"" << jsonEscape(e_.session_id) << "\""
        << ",\"name\":\""        << jsonEscape(e_.name)       << "\""
        << ",\"platform\":\""    << jsonEscape(e_.platform)   << "\""
        << ",\"device_id\":\""   << e_.device_id              << "\""
        << ",\"stream_id\":\""   << e_.stream_id              << "\""
        << ",\"stack_trace\":\"" << jsonEscape(e_.stack_trace) << "\""
        << ",\"user_scope\":\""  << jsonEscape(e_.user_scope)  << "\""
        << ",\"scope_depth\":"   << e_.scope_depth
        << ",\"start_ns\":"      << e_.start_ns
        << ",\"end_ns\":"        << e_.end_ns
        << ",\"api_start_ns\":"  << e_.api_start_ns
        << ",\"api_exit_ns\":"   << e_.api_exit_ns
        << ",\"corr_id\":"       << e_.corr_id
        << ",\"bytes\":"         << e_.bytes
        << ",\"copy_kind\":\"" << jsonEscape(e_.copy_kind) << "\""
        << ",\"src_kind\":\""  << jsonEscape(e_.src_kind)  << "\""
        << ",\"dst_kind\":\""  << jsonEscape(e_.dst_kind)  << "\""
        << "}";
    return oss.str();
}

std::string MemsetEventModel::buildJson() const {
    std::ostringstream oss;
    oss << "{\"type\":\"memset_event\""
        << ",\"pid\":"           << e_.pid
        << ",\"app\":\""         << jsonEscape(e_.app)        << "\""
        << ",\"session_id\":\"" << jsonEscape(e_.session_id) << "\""
        << ",\"name\":\""        << jsonEscape(e_.name)       << "\""
        << ",\"platform\":\""    << jsonEscape(e_.platform)   << "\""
        << ",\"device_id\":\""   << e_.device_id              << "\""
        << ",\"stream_id\":\""   << e_.stream_id              << "\""
        << ",\"stack_trace\":\"" << jsonEscape(e_.stack_trace) << "\""
        << ",\"user_scope\":\""  << jsonEscape(e_.user_scope)  << "\""
        << ",\"scope_depth\":"   << e_.scope_depth
        << ",\"start_ns\":"      << e_.start_ns
        << ",\"end_ns\":"        << e_.end_ns
        << ",\"api_start_ns\":"  << e_.api_start_ns
        << ",\"api_exit_ns\":"   << e_.api_exit_ns
        << ",\"corr_id\":"       << e_.corr_id
        << ",\"bytes\":"         << e_.bytes
        << "}";
    return oss.str();
}

}  // namespace gpufl::model

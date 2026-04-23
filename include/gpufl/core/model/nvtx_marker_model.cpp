#include "gpufl/core/model/nvtx_marker_model.hpp"

#include <sstream>

#include "gpufl/core/model/model_utils.hpp"

namespace gpufl::model {

std::string NvtxMarkerModel::buildJson() const {
    std::ostringstream oss;
    oss << "{\"type\":\"nvtx_marker_event\""
        << ",\"pid\":"         << e_.pid
        << ",\"app\":\""       << jsonEscape(e_.app)        << "\""
        << ",\"session_id\":\"" << jsonEscape(e_.session_id) << "\""
        << ",\"name\":\""      << jsonEscape(e_.name)       << "\""
        << ",\"domain\":\""    << jsonEscape(e_.domain)     << "\""
        << ",\"start_ns\":"    << e_.start_ns
        << ",\"end_ns\":"      << e_.end_ns
        << ",\"duration_ns\":" << e_.duration_ns
        << ",\"marker_id\":"   << e_.marker_id
        << "}";
    return oss.str();
}

}  // namespace gpufl::model

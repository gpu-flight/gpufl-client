#include "gpufl/core/model/deep_window_model.hpp"

#include <sstream>

#include "gpufl/core/model/model_utils.hpp"

namespace gpufl::model {

std::string DeepWindowModel::buildJson() const {
    std::ostringstream engines;
    engines << '[';
    for (size_t i = 0; i < e_.engines.size(); ++i) {
        if (i) engines << ',';
        engines << '"' << jsonEscape(e_.engines[i]) << '"';
    }
    engines << ']';

    std::ostringstream oss;
    oss << "{\"type\":\"deep_window_event\""
        << ",\"pid\":"                     << e_.pid
        << ",\"app\":\""                   << jsonEscape(e_.app)          << "\""
        << ",\"session_id\":\""            << jsonEscape(e_.session_id)   << "\""
        << ",\"name\":\""                  << jsonEscape(e_.name)         << "\""
        << ",\"close_reason\":\""          << jsonEscape(e_.close_reason) << "\""
        << ",\"engines\":"                 << engines.str()
        << ",\"start_ns\":"                << e_.start_ns
        << ",\"end_ns\":"                  << e_.end_ns
        << ",\"duration_ns\":"             << e_.duration_ns
        << ",\"launches_covered\":"        << e_.launches_covered
        << ",\"requested_duration_ms\":"   << e_.requested_duration_ms
        << ",\"requested_max_launches\":"  << e_.requested_max_launches
        << "}";
    return oss.str();
}

}  // namespace gpufl::model

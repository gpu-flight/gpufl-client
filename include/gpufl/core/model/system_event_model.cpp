#include "gpufl/core/model/system_event_model.hpp"

#include <sstream>

#include "gpufl/core/model/model_utils.hpp"

namespace gpufl::model {

std::string SystemStartModel::buildJson() const {
    std::ostringstream oss;
    oss << "{\"type\":\"system_start\""
        << ",\"pid\":"           << e_.pid
        << ",\"app\":\""         << jsonEscape(e_.app)        << "\""
        << ",\"session_id\":\"" << jsonEscape(e_.session_id) << "\""
        << ",\"name\":\""        << jsonEscape(e_.name)       << "\""
        << ",\"ts_ns\":"         << e_.ts_ns
        << ",\"host\":"          << hostToJson(e_.host)
        << ",\"devices\":"       << devicesToJson(e_.devices)
        << "}";
    return oss.str();
}

std::string SystemStopModel::buildJson() const {
    std::ostringstream oss;
    oss << "{\"type\":\"system_stop\""
        << ",\"pid\":"           << e_.pid
        << ",\"app\":\""         << jsonEscape(e_.app)        << "\""
        << ",\"session_id\":\"" << jsonEscape(e_.session_id) << "\""
        << ",\"name\":\""        << jsonEscape(e_.name)       << "\""
        << ",\"ts_ns\":"         << e_.ts_ns
        << ",\"host\":"          << hostToJson(e_.host)
        << ",\"devices\":"       << devicesToJson(e_.devices)
        << "}";
    return oss.str();
}

std::string SystemSampleModel::buildJson() const {
    std::ostringstream oss;
    oss << "{\"type\":\"system_sample\""
        << ",\"pid\":"           << e_.pid
        << ",\"app\":\""         << jsonEscape(e_.app)        << "\""
        << ",\"session_id\":\"" << jsonEscape(e_.session_id) << "\""
        << ",\"name\":\""        << jsonEscape(e_.name)       << "\""
        << ",\"ts_ns\":"         << e_.ts_ns
        << ",\"host\":"          << hostToJson(e_.host)
        << ",\"devices\":"       << devicesToJson(e_.devices)
        << "}";
    return oss.str();
}

}  // namespace gpufl::model

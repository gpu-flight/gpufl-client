#include "gpufl/core/model/scope_event_model.hpp"

#include <sstream>

#include "gpufl/core/model/model_utils.hpp"

namespace gpufl::model {

std::string ScopeBeginModel::buildJson() const {
    std::ostringstream oss;
    oss << "{\"type\":\"scope_begin\""
        << ",\"pid\":"           << e_.pid
        << ",\"app\":\""         << jsonEscape(e_.app)        << "\""
        << ",\"session_id\":\"" << jsonEscape(e_.session_id) << "\""
        << ",\"name\":\""        << jsonEscape(e_.name)       << "\""
        << ",\"tag\":\""         << jsonEscape(e_.tag)        << "\""
        << ",\"ts_ns\":"         << e_.ts_ns
        << ",\"user_scope\":\""  << jsonEscape(e_.user_scope) << "\""
        << ",\"scope_depth\":"   << e_.scope_depth
        << ",\"host\":"          << hostToJson(e_.host)
        << ",\"devices\":"       << devicesToJson(e_.devices)
        << "}";
    return oss.str();
}

std::string ScopeEndModel::buildJson() const {
    std::ostringstream oss;
    oss << "{\"type\":\"scope_end\""
        << ",\"pid\":"           << e_.pid
        << ",\"app\":\""         << jsonEscape(e_.app)        << "\""
        << ",\"session_id\":\"" << jsonEscape(e_.session_id) << "\""
        << ",\"name\":\""        << jsonEscape(e_.name)       << "\""
        << ",\"tag\":\""         << jsonEscape(e_.tag)        << "\""
        << ",\"ts_ns\":"         << e_.ts_ns
        << ",\"user_scope\":\""  << jsonEscape(e_.user_scope) << "\""
        << ",\"scope_depth\":"   << e_.scope_depth
        << ",\"host\":"          << hostToJson(e_.host)
        << ",\"devices\":"       << devicesToJson(e_.devices)
        << "}";
    return oss.str();
}

}  // namespace gpufl::model

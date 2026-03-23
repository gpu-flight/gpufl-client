#include "gpufl/core/model/lifecycle_model.hpp"

#include <sstream>

#include "gpufl/core/model/model_utils.hpp"

namespace gpufl::model {

std::string InitEventModel::buildJson() const {
    std::ostringstream oss;
    oss << "{\"version\":1,\"type\":\"job_start\""
        << ",\"pid\":"         << e_.pid
        << ",\"app\":\""       << jsonEscape(e_.app)        << "\""
        << ",\"session_id\":\"" << jsonEscape(e_.session_id) << "\""
        << ",\"log_path\":\""  << jsonEscape(e_.log_path)   << "\""
        << ",\"ts_ns\":"       << e_.ts_ns
        << ",\"host\":"        << hostToJson(e_.host)
        << ",\"devices\":"     << devicesToJson(e_.devices)
        << ",\"cuda_static_devices\":" << cudaStaticDevicesToJson(e_.cuda_static_device_infos)
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

}  // namespace gpufl::model

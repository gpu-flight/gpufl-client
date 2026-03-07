#include "gpufl/core/model/profile_sample_model.hpp"

#include <iomanip>
#include <sstream>

#include "gpufl/core/model/model_utils.hpp"

namespace gpufl::model {

std::string ProfileSampleModel::buildJson() const {
    std::ostringstream oss;
    oss << "{\"type\":\"profile_sample\""
        << ",\"pid\":"           << e_.pid
        << ",\"app\":\""         << jsonEscape(e_.app)        << "\""
        << ",\"session_id\":\"" << jsonEscape(e_.session_id) << "\""
        << ",\"ts_ns\":"         << e_.ts_ns
        << ",\"device_id\":"     << e_.device_id
        << ",\"corr_id\":"       << e_.corr_id;

    if (e_.samples_count > 0) {
        oss << ",\"sample_count\":" << e_.samples_count
            << ",\"stall_reason\":" << e_.stall_reason
            << ",\"reason_name\":\"" << jsonEscape(e_.reason_name) << "\"";
    }

    if (!e_.metric_name.empty()) {
        oss << ",\"metric_name\":\""  << jsonEscape(e_.metric_name) << "\""
            << ",\"metric_value\":"   << e_.metric_value
            << ",\"pc_offset\":\"0x" << std::hex << e_.pc_offset << std::dec << "\"";
    }

    oss << ",\"source_file\":\""    << jsonEscape(e_.source_file)    << "\""
        << ",\"function_name\":\"" << jsonEscape(e_.function_name) << "\""
        << ",\"source_line\":"     << e_.source_line
        << "}";
    return oss.str();
}

}  // namespace gpufl::model

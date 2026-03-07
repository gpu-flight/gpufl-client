#include "gpufl/core/model/perf_metric_model.hpp"

#include <iomanip>
#include <sstream>

#include "gpufl/core/model/model_utils.hpp"

namespace gpufl::model {

std::string PerfMetricModel::buildJson() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "{\"type\":\"perf_metric_event\""
        << ",\"pid\":"           << e_.pid
        << ",\"app\":\""         << jsonEscape(e_.app)        << "\""
        << ",\"session_id\":\"" << jsonEscape(e_.session_id) << "\""
        << ",\"name\":\""        << jsonEscape(e_.name)       << "\""
        << ",\"start_ns\":"      << e_.start_ns
        << ",\"end_ns\":"        << e_.end_ns
        << ",\"device_id\":"     << e_.device_id
        << ",\"sm_throughput_pct\":" << e_.sm_throughput_pct
        << ",\"l1_hit_rate_pct\":"   << e_.l1_hit_rate_pct
        << ",\"l2_hit_rate_pct\":"   << e_.l2_hit_rate_pct
        << ",\"dram_read_bytes\":"   << e_.dram_read_bytes
        << ",\"dram_write_bytes\":"  << e_.dram_write_bytes
        << ",\"tensor_active_pct\":" << e_.tensor_active_pct
        << ",\"user_scope\":\""  << jsonEscape(e_.user_scope) << "\""
        << ",\"scope_depth\":"   << e_.scope_depth
        << "}";
    return oss.str();
}

}  // namespace gpufl::model

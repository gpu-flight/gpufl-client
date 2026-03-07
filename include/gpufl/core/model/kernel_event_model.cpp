#include "gpufl/core/model/kernel_event_model.hpp"

#include <sstream>

#include "gpufl/core/model/model_utils.hpp"

namespace gpufl::model {

std::string KernelEventModel::buildJson() const {
    std::ostringstream oss;
    oss << std::boolalpha;
    oss << "{\"type\":\"kernel_event\""
        << ",\"pid\":"          << e_.pid
        << ",\"app\":\""        << jsonEscape(e_.app)        << "\""
        << ",\"session_id\":\"" << jsonEscape(e_.session_id) << "\""
        << ",\"name\":\""       << jsonEscape(e_.name)       << "\""
        << ",\"platform\":\""   << jsonEscape(e_.platform)   << "\""
        << ",\"has_details\":"  << e_.has_details
        << ",\"device_id\":\""  << e_.device_id              << "\""
        << ",\"stream_id\":\""  << e_.stream_id              << "\""
        << ",\"stack_trace\":\"" << jsonEscape(e_.stack_trace) << "\""
        << ",\"user_scope\":\""  << jsonEscape(e_.user_scope)  << "\""
        << ",\"scope_depth\":"   << e_.scope_depth
        << ",\"start_ns\":"      << e_.start_ns
        << ",\"end_ns\":"        << e_.end_ns
        << ",\"api_start_ns\":"  << e_.api_start_ns
        << ",\"api_exit_ns\":"   << e_.api_exit_ns
        << ",\"grid\":\""        << jsonEscape(e_.grid)       << "\""
        << ",\"block\":\""       << jsonEscape(e_.block)      << "\""
        << ",\"dyn_shared_bytes\":"    << e_.dyn_shared_bytes
        << ",\"num_regs\":"            << e_.num_regs
        << ",\"static_shared_bytes\":" << e_.static_shared_bytes
        << ",\"local_bytes\":"         << e_.local_bytes
        << ",\"const_bytes\":"         << e_.const_bytes
        << ",\"occupancy\":"           << e_.occupancy
        << ",\"reg_occupancy\":"       << e_.reg_occupancy
        << ",\"smem_occupancy\":"      << e_.smem_occupancy
        << ",\"warp_occupancy\":"      << e_.warp_occupancy
        << ",\"block_occupancy\":"     << e_.block_occupancy
        << ",\"limiting_resource\":\"" << jsonEscape(e_.limiting_resource) << "\""
        << ",\"max_active_blocks\":"   << e_.max_active_blocks
        << ",\"corr_id\":"             << e_.corr_id
        << ",\"local_mem_total\":"     << e_.local_mem_total
        << ",\"cache_config_requested\":" << static_cast<int>(e_.cache_config_requested)
        << ",\"cache_config_executed\":"  << static_cast<int>(e_.cache_config_executed)
        << ",\"shared_mem_executed\":"    << e_.shared_mem_executed
        << "}";
    return oss.str();
}

}  // namespace gpufl::model

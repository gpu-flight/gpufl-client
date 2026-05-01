#include "gpufl/core/model/graph_launch_event_model.hpp"

#include <sstream>

#include "gpufl/core/model/model_utils.hpp"

namespace gpufl::model {

std::string GraphLaunchEventModel::buildJson() const {
    // Field set:
    //   pid, app, session_id   — universal envelope
    //   start_ns / end_ns      — wall-clock launch window; both 0 when
    //                            CUPTI couldn't collect timing (the
    //                            graph_id is still useful in that case)
    //   duration_ns            — derived; kept for backend convenience
    //   graph_id               — unique id of the captured graph;
    //                            repeated launches share an id
    //   device_id, stream_id   — context fields
    //   corr_id                — joins to the driver-API record that
    //                            triggered the launch (cuGraphLaunch)
    std::ostringstream oss;
    oss << "{\"type\":\"graph_launch_event\""
        << ",\"pid\":"          << e_.pid
        << ",\"app\":\""        << jsonEscape(e_.app)        << "\""
        << ",\"session_id\":\"" << jsonEscape(e_.session_id) << "\""
        << ",\"start_ns\":"     << e_.start_ns
        << ",\"end_ns\":"       << e_.end_ns
        << ",\"duration_ns\":"  << e_.duration_ns
        << ",\"graph_id\":"     << e_.graph_id
        << ",\"device_id\":"    << e_.device_id
        << ",\"stream_id\":"    << e_.stream_id
        << ",\"corr_id\":"      << e_.corr_id
        << "}";
    return oss.str();
}

}  // namespace gpufl::model

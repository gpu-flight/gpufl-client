#include "gpufl/core/model/synchronization_event_model.hpp"

#include <sstream>

#include "gpufl/core/model/model_utils.hpp"

namespace gpufl::model {

std::string SynchronizationEventModel::buildJson() const {
    // Field set, in order:
    //   pid, app, session_id  — universal envelope
    //   start_ns / end_ns     — wall clock; backend partitions on time
    //   duration_ns           — keep redundant copy (mirrors NvtxMarkerModel)
    //   sync_type             — integer; backend stores raw, frontend
    //                           maps to label (StreamSynchronize / etc.)
    //   stream_id, event_id, context_id, corr_id — join keys
    //
    // Numeric-only fields use no quotes; strings are escaped via
    // jsonEscape (handles backslashes / quotes / control chars).
    std::ostringstream oss;
    oss << "{\"type\":\"synchronization_event\""
        << ",\"pid\":"          << e_.pid
        << ",\"app\":\""        << jsonEscape(e_.app)        << "\""
        << ",\"session_id\":\"" << jsonEscape(e_.session_id) << "\""
        << ",\"start_ns\":"     << e_.start_ns
        << ",\"end_ns\":"       << e_.end_ns
        << ",\"duration_ns\":"  << e_.duration_ns
        << ",\"sync_type\":"    << static_cast<int>(e_.sync_type)
        << ",\"stream_id\":"    << e_.stream_id
        << ",\"event_id\":"     << e_.event_id
        << ",\"context_id\":"   << e_.context_id
        << ",\"corr_id\":"      << e_.corr_id
        << ",\"stack_trace\":\"" << jsonEscape(e_.stack_trace) << "\""
        << "}";
    return oss.str();
}

}  // namespace gpufl::model

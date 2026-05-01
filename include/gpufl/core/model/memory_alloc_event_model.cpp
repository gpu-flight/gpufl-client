#include "gpufl/core/model/memory_alloc_event_model.hpp"

#include <sstream>

#include "gpufl/core/model/model_utils.hpp"

namespace gpufl::model {

std::string MemoryAllocEventModel::buildJson() const {
    // Field set:
    //   pid, app, session_id  — universal envelope
    //   start_ns / duration_ns — host call timestamp; duration is 0
    //                            in v1 because CUpti_ActivityMemory4
    //                            carries no end timestamp
    //   memory_op, memory_kind — both stored as integers (uint8 on
    //                            the wire); frontend renders human
    //                            labels via lookup tables
    //   address, bytes         — VA + size of the allocation
    //   device_id, stream_id   — context fields; stream_id is set
    //                            for cudaMallocAsync, 0 otherwise
    //   corr_id                — joins to the matching API record
    //                            (we don't surface this in v1 but
    //                            it's there for future leak-pairing)
    std::ostringstream oss;
    oss << "{\"type\":\"memory_alloc_event\""
        << ",\"pid\":"          << e_.pid
        << ",\"app\":\""        << jsonEscape(e_.app)        << "\""
        << ",\"session_id\":\"" << jsonEscape(e_.session_id) << "\""
        << ",\"start_ns\":"     << e_.start_ns
        << ",\"duration_ns\":"  << e_.duration_ns
        << ",\"memory_op\":"    << static_cast<int>(e_.memory_op)
        << ",\"memory_kind\":"  << static_cast<int>(e_.memory_kind)
        << ",\"address\":"      << e_.address
        << ",\"bytes\":"        << e_.bytes
        << ",\"device_id\":"    << e_.device_id
        << ",\"stream_id\":"    << e_.stream_id
        << ",\"corr_id\":"      << e_.corr_id
        << "}";
    return oss.str();
}

}  // namespace gpufl::model

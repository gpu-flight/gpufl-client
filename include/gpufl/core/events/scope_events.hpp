#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "gpufl/core/events/sample_types.hpp"

namespace gpufl {

struct ScopeBeginEvent {
    uint64_t scope_id = 0;
    int pid = 0;
    std::string app;
    std::string session_id;
    std::string name;
    std::string tag;
    int64_t ts_ns = 0;

    HostSample host;
    std::vector<DeviceSample> devices;

    std::string user_scope;
    int scope_depth = 0;
};

struct ScopeEndEvent {
    uint64_t scope_id = 0;
    int pid = 0;
    std::string app;
    std::string session_id;
    std::string name;
    std::string tag;
    int64_t ts_ns = 0;

    HostSample host;
    std::vector<DeviceSample> devices;

    std::string user_scope;
    int scope_depth = 0;
};

struct ScopeBatchRow {
    int64_t  ts_ns             = 0;  // absolute timestamp
    uint64_t scope_instance_id = 0;  // monotonic ID shared by begin/end pair
    uint32_t name_id           = 0;  // scope name dictionary ID
    uint8_t  event_type        = 0;  // 0 begin, 1 end, 2 continuation-open,
                                     // 3 continuation-close
    int      depth             = 0;
    // Logical first-open timestamp. Equal to ts_ns for an ordinary begin/end
    // pair; preserved across every continuation row in a segmented run.
    int64_t  original_start_ns = 0;

    // Optional benchmark metadata set on the BEGIN row only (0 on END).
    // Populated when the scope was opened with iteration metadata -
    // e.g. Python's `for _ in gpufl.Scope(name, repeat=N, warmup=K)`.
    // 0 on either field means "not provided" and the row serializes
    // the same as before (analyzer / backend simply skip the metric).
    // Backend joins by scope_instance_id to read the begin-row values.
    uint32_t repeat            = 0;  // measured iterations bracketed by scope
    uint32_t warmup            = 0;  // iterations run BEFORE scope opened
};

}  // namespace gpufl

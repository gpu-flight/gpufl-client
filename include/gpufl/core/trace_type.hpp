#pragma once
#include <cstdint>

namespace gpufl {
enum class TraceType : uint8_t {
    KERNEL,
    PC_SAMPLE,
    SASS_METRIC,
    RANGE,
    MEMCPY,
    MEMSET,
    // NVTX range captured via CUPTI_ACTIVITY_KIND_MARKER. Fields used on
    // ActivityRecord: name (range name), cpu_start_ns, duration_ns.
    // Emitted as `nvtx_marker_event` in the NDJSON log.
    NVTX_MARKER,
};
}
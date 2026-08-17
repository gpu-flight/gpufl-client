#pragma once

#include <cstdint>

#include "gpufl/backends/amd/amd_profiling_policy.hpp"
#include "gpufl/core/events.hpp"

namespace gpufl::amd {

struct AmdCaptureCapabilityInput {
    std::string session_id;
    int64_t ts_ns = 0;
    AmdResolvedProfilingPlan plan;
    bool trace_configured = false;
    uint64_t kernel_rows = 0;
    uint64_t memcpy_rows = 0;
    uint64_t profiling_sample_rows = 0;
    uint64_t dropped_trace_records = 0;
};

CaptureCapabilitiesEvent BuildAmdCaptureCapabilitiesEvent(
    const AmdCaptureCapabilityInput& input);

}  // namespace gpufl::amd

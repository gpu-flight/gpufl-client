#pragma once

#include <string>

#include "gpufl/core/monitor.hpp"

namespace gpufl::amd {

enum class AmdProfilingPath {
    None,
    BufferTracing,
    DispatchCounting,
    PcSampling,
    DeviceCounting,
};

struct AmdProfilingSupport {
    bool buffer_tracing = true;
    bool dispatch_counting = false;
    bool pc_sampling = false;
    bool device_counting = false;
};

struct AmdResolvedProfilingPlan {
    ProfilingEngine requested_engine = ProfilingEngine::Monitor;
    AmdProfilingPath selected_path = AmdProfilingPath::None;
    bool degraded = false;
    std::string reason_code;
};

// Request names describe cross-vendor intent; selected paths below identify
// the concrete ROCprofiler service and therefore carry the AMD prefix.
const char* AmdRequestIntentWireName(ProfilingEngine engine);
const char* AmdSelectedPathWireName(AmdProfilingPath path);

AmdResolvedProfilingPlan ResolveAmdProfilingPlan(
    ProfilingEngine requested,
    const AmdProfilingSupport& support);

bool AmdRequestNeedsDispatchCounting(ProfilingEngine engine);
bool AmdRequestNeedsPcSampling(ProfilingEngine engine);
bool AmdRequestNeedsDeviceCounting(ProfilingEngine engine);

}  // namespace gpufl::amd

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "gpufl/backends/amd/amd_pc_sampling_policy.hpp"
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
    // True only when the run opted in and the agent configured a stochastic
    // sampler; never inferred from the architecture's name.
    bool pc_sampling = false;
    bool device_counting = false;
    // Why pc_sampling is false.
    AmdPcSamplingRefusal pc_sampling_refusal =
        AmdPcSamplingRefusal::NotOptedIn;
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

// GPUFL_ENGINE_COMBO lists a '+' pass's engines, while the AMD backend runs
// one service beside tracing. Every AMD session traces, so Trace+PcSampling
// is PC sampling. Any other group keeps its first engine: PC sampling can't
// share a context with counters.
ProfilingEngine ResolveAmdRequestedEngine(ProfilingEngine selected,
                                          std::string_view combo);

bool AmdRequestNeedsDispatchCounting(ProfilingEngine engine);
bool AmdRequestNeedsPcSampling(ProfilingEngine engine);
bool AmdRequestNeedsDeviceCounting(ProfilingEngine engine);

// AMD's device-counting service consumes native ROCprofiler counter names.
// Explicit user metrics win; the default is a portable device-utilization
// signal that is available as a derived counter across supported AMD GPUs.
std::vector<std::string> ResolveAmdDeviceCountingMetrics(
    const std::vector<std::string>& requested_metrics);

// Dispatch counting is currently configured for one GPU agent. Resolve only
// records from that agent so a secondary GPU can never be mislabeled as device
// zero (or as the configured primary device).
std::optional<uint32_t> ResolveAmdDispatchDeviceId(
    uint64_t configured_agent_handle,
    uint32_t configured_device_id,
    uint64_t dispatch_agent_handle);

}  // namespace gpufl::amd

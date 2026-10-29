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
    // The pass's '+' group named Trace beside PcSampling.
    bool trace_requested = false;
    // The pass's '+' group named PcSampling, and another engine ran.
    bool pc_sampling_dropped = false;
};

// Request names describe cross-vendor intent; selected paths below identify
// the concrete ROCprofiler service and therefore carry the AMD prefix.
const char* AmdRequestIntentWireName(ProfilingEngine engine);
const char* AmdSelectedPathWireName(AmdProfilingPath path);

// What a session reports as its engine: the session's profiling_engine and
// capture_capabilities' selected_engine. A Trace+PcSampling group traced
// every kernel and asked for that timing, so it reports amd.buffer_tracing;
// its pc_sampling row says what sampling did.
const char* AmdSessionEngineWireName(const AmdResolvedProfilingPlan& plan);

AmdResolvedProfilingPlan ResolveAmdProfilingPlan(
    ProfilingEngine requested,
    const AmdProfilingSupport& support);

// What a session asked the AMD backend for: GPUFL_PROFILING_ENGINE, read
// with a '+' pass's GPUFL_ENGINE_COMBO.
struct AmdProfilingRequest {
    ProfilingEngine engine = ProfilingEngine::Monitor;
    bool trace_requested = false;      // see AmdResolvedProfilingPlan
    bool pc_sampling_dropped = false;  // see AmdResolvedProfilingPlan
};

// The AMD backend runs one service beside tracing, so a group is resolved
// by ClassifyAmdPcSamplingGroup: PcSampling with Trace is PC sampling, and
// any other group keeps its first engine (`selected`).
AmdProfilingRequest ResolveAmdProfilingRequest(ProfilingEngine selected,
                                               std::string_view combo);

AmdResolvedProfilingPlan ResolveAmdProfilingPlan(
    const AmdProfilingRequest& request,
    const AmdProfilingSupport& support);

// Why a PcSampling request must not call rocprofiler-sdk's PC sampling API
// at all (no opt-in, or window-only arming), or nullopt to query the agent.
std::optional<AmdPcSamplingRefusal> AmdPcSamplingPreflight(
    const MonitorOptions& opts);

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

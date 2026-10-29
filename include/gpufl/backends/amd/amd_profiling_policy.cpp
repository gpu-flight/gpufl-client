#include "gpufl/backends/amd/amd_profiling_policy.hpp"

#include <algorithm>
#include <unordered_set>

namespace gpufl::amd {
namespace {

AmdResolvedProfilingPlan RetainBaselineTracing(
    const ProfilingEngine requested,
    const AmdProfilingSupport& support,
    const char* reason) {
    AmdResolvedProfilingPlan plan;
    plan.requested_engine = requested;
    plan.selected_path = support.buffer_tracing ? AmdProfilingPath::BufferTracing
                                       : AmdProfilingPath::None;
    plan.degraded = true;
    plan.reason_code = reason;
    return plan;
}

}  // namespace

const char* AmdRequestIntentWireName(const ProfilingEngine engine) {
    switch (engine) {
        case ProfilingEngine::Monitor: return "monitor";
        case ProfilingEngine::Trace: return "trace";
        case ProfilingEngine::PcSampling: return "pc_sampling";
        case ProfilingEngine::SassMetrics: return "sass_metrics";
        case ProfilingEngine::PmSampling: return "pm_sampling";
        case ProfilingEngine::RangeProfiler: return "range_profiler";
        case ProfilingEngine::RangeProfilerKernelReplay:
            return "range_profiler_kernel_replay";
        case ProfilingEngine::Deep: return "deep";
    }
    return "unknown";
}

const char* AmdSelectedPathWireName(const AmdProfilingPath path) {
    switch (path) {
        case AmdProfilingPath::None: return "amd.none";
        case AmdProfilingPath::BufferTracing: return "amd.buffer_tracing";
        case AmdProfilingPath::DispatchCounting: return "amd.dispatch_counting";
        case AmdProfilingPath::PcSampling: return "amd.pc_sampling";
        case AmdProfilingPath::DeviceCounting: return "amd.device_counting";
    }
    return "amd.unknown";
}

const char* AmdSessionEngineWireName(const AmdResolvedProfilingPlan& plan) {
    if (plan.selected_path == AmdProfilingPath::PcSampling &&
        plan.trace_requested) {
        return AmdSelectedPathWireName(AmdProfilingPath::BufferTracing);
    }
    return AmdSelectedPathWireName(plan.selected_path);
}

AmdProfilingRequest ResolveAmdProfilingRequest(const ProfilingEngine selected,
                                               const std::string_view combo) {
    AmdProfilingRequest request;
    request.engine = selected;
    switch (ClassifyAmdPcSamplingGroup(combo, ',')) {
        case AmdPcSamplingGroup::Absent:
            break;
        case AmdPcSamplingGroup::Samples: {
            request.engine = ProfilingEngine::PcSampling;
            const auto names = SplitAmdEngineGroup(combo, ',');
            request.trace_requested =
                std::find(names.begin(), names.end(), "Trace") != names.end();
            break;
        }
        case AmdPcSamplingGroup::Dropped:
            request.pc_sampling_dropped = true;
            break;
    }
    return request;
}

AmdResolvedProfilingPlan ResolveAmdProfilingPlan(
    const AmdProfilingRequest& request, const AmdProfilingSupport& support) {
    AmdResolvedProfilingPlan plan =
        ResolveAmdProfilingPlan(request.engine, support);
    plan.trace_requested = request.trace_requested;
    plan.pc_sampling_dropped = request.pc_sampling_dropped;
    return plan;
}

std::optional<AmdPcSamplingRefusal> AmdPcSamplingPreflight(
    const MonitorOptions& opts) {
    if (opts.amd_pc_sampling != AmdPcSamplingMethod::Stochastic) {
        return AmdPcSamplingRefusal::NotOptedIn;
    }
    // The sampler runs with the whole context, which tracing keeps running
    // between windows; the other engines honor windows, so refuse rather
    // than sample where no window asked.
    if (opts.deep_arm_mode == DeepArmMode::WindowOnly) {
        return AmdPcSamplingRefusal::WindowOnly;
    }
    return std::nullopt;
}

bool AmdRequestNeedsDispatchCounting(const ProfilingEngine engine) {
    return engine == ProfilingEngine::SassMetrics ||
           engine == ProfilingEngine::RangeProfiler ||
           engine == ProfilingEngine::RangeProfilerKernelReplay ||
           engine == ProfilingEngine::Deep;
}

bool AmdRequestNeedsPcSampling(const ProfilingEngine engine) {
    return engine == ProfilingEngine::PcSampling;
}

bool AmdRequestNeedsDeviceCounting(const ProfilingEngine engine) {
    return engine == ProfilingEngine::PmSampling ||
           engine == ProfilingEngine::Deep;
}

std::vector<std::string> ResolveAmdDeviceCountingMetrics(
    const std::vector<std::string>& requested_metrics) {
    if (requested_metrics.empty()) return {"GPUBusy"};

    std::vector<std::string> resolved;
    resolved.reserve(requested_metrics.size());
    std::unordered_set<std::string> seen;
    for (const auto& metric : requested_metrics) {
        if (!metric.empty() && seen.emplace(metric).second) {
            resolved.push_back(metric);
        }
    }
    return resolved.empty() ? std::vector<std::string>{"GPUBusy"}
                            : resolved;
}

std::optional<uint32_t> ResolveAmdDispatchDeviceId(
    const uint64_t configured_agent_handle,
    const uint32_t configured_device_id,
    const uint64_t dispatch_agent_handle) {
    if (configured_agent_handle == 0 || dispatch_agent_handle == 0 ||
        configured_agent_handle != dispatch_agent_handle) {
        return std::nullopt;
    }
    return configured_device_id;
}

AmdResolvedProfilingPlan ResolveAmdProfilingPlan(
    const ProfilingEngine requested,
    const AmdProfilingSupport& support) {
    AmdResolvedProfilingPlan plan;
    plan.requested_engine = requested;

    switch (requested) {
        case ProfilingEngine::Monitor:
            plan.selected_path = AmdProfilingPath::None;
            return plan;
        case ProfilingEngine::Trace:
            if (support.buffer_tracing) {
                plan.selected_path = AmdProfilingPath::BufferTracing;
                return plan;
            }
            return RetainBaselineTracing(requested, support,
                                   "rocprofiler_buffer_tracing_unavailable");
        case ProfilingEngine::PcSampling:
            if (support.pc_sampling) {
                plan.selected_path = AmdProfilingPath::PcSampling;
                return plan;
            }
            return RetainBaselineTracing(
                requested, support,
                AmdPcSamplingRefusalReason(support.pc_sampling_refusal));
        case ProfilingEngine::PmSampling:
            if (support.device_counting) {
                plan.selected_path = AmdProfilingPath::DeviceCounting;
                return plan;
            }
            return RetainBaselineTracing(requested, support,
                                   "amd_device_counting_unavailable_baseline_tracing_retained");
        case ProfilingEngine::SassMetrics:
        case ProfilingEngine::RangeProfiler:
        case ProfilingEngine::RangeProfilerKernelReplay:
            if (support.dispatch_counting) {
                plan.selected_path = AmdProfilingPath::DispatchCounting;
                plan.degraded = true;
                plan.reason_code = "requested_metric_model_unavailable_dispatch_counting_selected";
                return plan;
            }
            return RetainBaselineTracing(requested, support,
                                   "requested_metric_model_and_dispatch_counting_unavailable");
        case ProfilingEngine::Deep:
            if (support.dispatch_counting) {
                plan.selected_path = AmdProfilingPath::DispatchCounting;
                plan.degraded = true;
                plan.reason_code = "deep_services_unavailable_dispatch_counting_selected";
                return plan;
            }
            return RetainBaselineTracing(requested, support,
                                   "deep_services_unavailable_baseline_tracing_retained");
    }

    return RetainBaselineTracing(requested, support, "amd_engine_request_unknown");
}

}  // namespace gpufl::amd

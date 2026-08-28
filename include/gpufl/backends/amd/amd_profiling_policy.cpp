#include "gpufl/backends/amd/amd_profiling_policy.hpp"

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
            return RetainBaselineTracing(requested, support,
                                   "amd_pc_sampling_unavailable_baseline_tracing_retained");
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

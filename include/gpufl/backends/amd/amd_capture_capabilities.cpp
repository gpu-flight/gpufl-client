#include "gpufl/backends/amd/amd_capture_capabilities.hpp"

#include <string>
#include <utility>

namespace gpufl::amd {
namespace {

std::string WithThousands(const uint64_t value) {
    std::string digits = std::to_string(value);
    for (size_t at = digits.size(); at > 3; at -= 3) {
        digits.insert(at - 3, ",");
    }
    return digits;
}

std::string PcSamplingRefusalMessage(const std::string& reason,
                                     const std::string& detail) {
    if (reason == AmdPcSamplingRefusalReason(
                      AmdPcSamplingRefusal::NotOptedIn)) {
        return "AMD PC sampling is a beta feature of rocprofiler-sdk, and "
               "this run didn't enable it (gpufl trace "
               "--amd-pc-sampling=stochastic).";
    }
    if (reason == AmdPcSamplingRefusalReason(
                      AmdPcSamplingRefusal::MethodUnavailable)) {
        return "This GPU has no stochastic PC sampling. rocprofiler-sdk "
               "offers it on MI300- and MI350-series GPUs.";
    }
    if (reason == AmdPcSamplingRefusalReason(
                      AmdPcSamplingRefusal::DriverUnsupported)) {
        return "The amdgpu driver doesn't offer PC sampling. AMD's "
               "amdgpu-dkms driver does.";
    }
    if (reason ==
        AmdPcSamplingRefusalReason(AmdPcSamplingRefusal::Unavailable)) {
        return "AMD PC sampling is unavailable on this GPU: another process "
               "may be sampling it with a different configuration, or the "
               "process runs under ROCgdb.";
    }
    if (reason == AmdPcSamplingRefusalReason(
                      AmdPcSamplingRefusal::ConfigurationFailed)) {
        return "rocprofiler-sdk refused PC sampling: " +
               (detail.empty() ? std::string("no reason given") : detail) +
               ".";
    }
    if (reason ==
        AmdPcSamplingRefusalReason(AmdPcSamplingRefusal::WindowOnly)) {
        return "AMD PC sampling can't be limited to deep windows: "
               "rocprofiler-sdk samples for as long as the profiling context "
               "runs. This window-only run (GPUFL_DEEP_ARM=window) only "
               "traced.";
    }
    if (reason ==
        AmdPcSamplingRefusalReason(AmdPcSamplingRefusal::GroupUnsupported)) {
        return "On AMD GPUs, PC sampling runs alone or beside Trace, so this "
               "pass's group ran " +
               detail + " instead.";
    }
    return "AMD PC sampling did not run.";
}

void AddCapability(CaptureCapabilitiesEvent& event, std::string feature,
                   const bool requested, std::string status,
                   std::string mode, std::string reason,
                   std::string message) {
    event.capabilities.push_back(CaptureCapability{
        std::move(feature), requested, std::move(status), std::move(mode),
        std::move(reason), std::move(message)});
}

}  // namespace

CaptureCapabilitiesEvent BuildAmdCaptureCapabilitiesEvent(
    const AmdCaptureCapabilityInput& input) {
    CaptureCapabilitiesEvent event;
    event.session_id = input.session_id;
    event.ts_ns = input.ts_ns;
    event.requested_engine =
        AmdRequestIntentWireName(input.plan.requested_engine);
    event.selected_engine = AmdSessionEngineWireName(input.plan);

    const bool tracing_requested =
        input.plan.requested_engine != ProfilingEngine::Monitor;
    const bool dispatch_requested =
        AmdRequestNeedsDispatchCounting(input.plan.requested_engine);
    const bool pc_requested =
        AmdRequestNeedsPcSampling(input.plan.requested_engine) ||
        input.plan.pc_sampling_dropped;
    const bool device_counting_requested =
        AmdRequestNeedsDeviceCounting(input.plan.requested_engine);
    const bool dispatch_selected =
        input.plan.selected_path == AmdProfilingPath::DispatchCounting;
    const bool pc_selected =
        input.plan.selected_path == AmdProfilingPath::PcSampling;
    const bool device_counting_selected =
        input.plan.selected_path == AmdProfilingPath::DeviceCounting;

    AddCapability(
        event, "engine_selection", tracing_requested,
        !tracing_requested ? "not_requested"
                           : (input.plan.degraded ? "fallback" : "selected"),
        event.selected_engine, input.plan.reason_code,
        !tracing_requested
            ? "No ROCprofiler profiling service was requested."
            : (input.plan.degraded
                   ? "The requested feature is unavailable; the reported "
                     "ROCprofiler service was retained or selected explicitly."
                   : "The requested ROCprofiler service was selected."));

    AddCapability(
        event, "kernel_events", tracing_requested,
        !tracing_requested
            ? "not_requested"
            : (input.kernel_rows > 0
                   ? "collected"
                   : (input.trace_configured ? "enabled_no_data" : "skipped")),
        input.trace_configured ? "rocprofiler_kernel_dispatch" : "disabled",
        input.trace_configured
            ? (input.kernel_rows > 0 ? "" : "enabled_but_no_records")
            : (tracing_requested ? "rocprofiler_buffer_tracing_unavailable"
                                 : "not_selected"),
        input.kernel_rows > 0
            ? "Kernel dispatch records were collected through ROCprofiler SDK."
            : (input.trace_configured
                   ? "Kernel dispatch tracing was enabled but emitted no rows this session."
                   : "Kernel dispatch tracing was not active."));

    AddCapability(
        event, "memcpy_activity", tracing_requested,
        !tracing_requested
            ? "not_requested"
            : (input.memcpy_rows > 0
                   ? "collected"
                   : (input.trace_configured ? "enabled_no_data" : "skipped")),
        input.trace_configured ? "rocprofiler_memory_copy" : "disabled",
        input.trace_configured
            ? (input.memcpy_rows > 0 ? "" : "enabled_but_no_records")
            : (tracing_requested ? "rocprofiler_buffer_tracing_unavailable"
                                 : "not_selected"),
        input.memcpy_rows > 0
            ? "Memory-copy records were collected through ROCprofiler SDK."
            : (input.trace_configured
                   ? "Memory-copy tracing was enabled but emitted no rows this session."
                   : "Memory-copy tracing was not active."));

    AddCapability(
        event, "sync_activity", input.synchronization_requested,
        !input.synchronization_requested
            ? "not_requested"
            : input.synchronization_configured
            ? (input.synchronization_rows > 0 ? "collected" : "enabled_no_data")
            : "skipped",
        input.synchronization_configured
            ? "rocprofiler_hip_runtime_api_ext"
            : "disabled",
        input.synchronization_configured
            ? (input.synchronization_rows > 0 ? "" : "enabled_but_no_records")
            : input.synchronization_requested
            ? "rocprofiler_hip_runtime_api_unavailable"
            : "",
        input.synchronization_configured
            ? (input.synchronization_rows > 0
                   ? "HIP synchronization records were collected through ROCprofiler SDK."
                   : "HIP synchronization tracing was enabled but emitted no rows this session.")
            : input.synchronization_requested
            ? "HIP synchronization tracing was requested but unavailable."
            : "Synchronization tracing was not requested.");

    AddCapability(
        event, "memory_activity", input.memory_activity_requested,
        !input.memory_activity_requested
            ? "not_requested"
            : (input.memory_activity_configured
                   ? (input.memory_activity_rows > 0 ? "collected"
                                                     : "enabled_no_data")
                   : "skipped"),
        input.memory_activity_configured ? "rocprofiler_memory_allocation"
                                         : "disabled",
        input.memory_activity_configured
            ? (input.memory_activity_rows > 0 ? "" : "enabled_but_no_records")
            : (input.memory_activity_requested
                   ? "rocprofiler_memory_allocation_unavailable"
                   : ""),
        input.memory_activity_configured
            ? (input.memory_activity_rows > 0
                   ? "ROCprofiler memory-allocation records were collected."
                   : "ROCprofiler memory-allocation tracing was enabled but "
                     "emitted no rows this session.")
            : (input.memory_activity_requested
                   ? "ROCprofiler memory-allocation tracing was requested but unavailable."
                   : "Memory-allocation tracing was not requested."));

    AddCapability(
        event, "dispatch_counting", dispatch_requested,
        !dispatch_requested
            ? "not_requested"
            : (dispatch_selected
                   ? (input.profiling_sample_rows > 0 ? "collected"
                                                      : "enabled_no_data")
                   : "skipped"),
        dispatch_selected ? "rocprofiler_dispatch_counting_service"
                          : "disabled",
        dispatch_selected
            ? (input.profiling_sample_rows > 0 ? "" : "enabled_but_no_records")
            : (dispatch_requested ? input.plan.reason_code : "not_selected"),
        dispatch_selected
            ? (input.profiling_sample_rows > 0
                   ? "Per-dispatch AMD hardware-counter samples were collected."
                   : "AMD dispatch counting was selected but emitted no samples this session.")
            : "AMD dispatch counting was not selected.");

    {
        std::string status = "not_requested";
        std::string reason = "not_selected";
        std::string message = "AMD PC sampling was not requested.";
        if (input.plan.pc_sampling_dropped) {
            status = "skipped";
            reason = AmdPcSamplingRefusalReason(
                AmdPcSamplingRefusal::GroupUnsupported);
            message = PcSamplingRefusalMessage(
                reason, AmdRequestIntentWireName(input.plan.requested_engine));
        } else if (pc_requested && !pc_selected) {
            status = "skipped";
            reason = input.plan.reason_code;
            message =
                PcSamplingRefusalMessage(reason, input.pc_sampling_detail);
        } else if (pc_selected) {
            const std::string every =
                "Stochastic PC sampling every " +
                WithThousands(input.pc_sampling_interval) + " cycles";
            const std::string lost = WithThousands(input.pc_samples_lost);
            // Lost samples make the data partial even when none survived:
            // a quiet run and one that lost everything must not look alike.
            if (input.pc_samples_lost > 0) {
                status = "partial";
                reason = "amd_pc_sampling_samples_dropped";
            } else if (input.pc_samples > 0) {
                status = "collected";
                reason = "";
            } else {
                status = "enabled_no_data";
                reason = "enabled_but_no_records";
            }
            if (input.pc_samples > 0) {
                message = every + ": " + WithThousands(input.pc_samples) +
                          " samples, " +
                          WithThousands(input.pc_samples_issued) +
                          " issued, " +
                          WithThousands(input.pc_samples_not_issued) +
                          " waiting.";
                if (input.pc_samples_lost > 0) {
                    message += " " + lost +
                               " samples were dropped or matched no kernel.";
                }
            } else if (input.pc_samples_lost > 0) {
                message = every + " kept no samples: " + lost +
                          " were dropped or matched no kernel.";
            } else {
                message =
                    "AMD PC sampling was enabled but recorded no samples "
                    "this session.";
            }
        }
        AddCapability(event, "pc_sampling", pc_requested, std::move(status),
                      pc_selected ? kAmdPcSamplingModeWireName : "disabled",
                      std::move(reason), std::move(message));
    }

    AddCapability(
        event, "device_counting", device_counting_requested,
        !device_counting_requested
            ? "not_requested"
            : (device_counting_selected
                   ? (input.pm_sample_rows > 0 ? "collected"
                                               : "enabled_no_data")
                   : "skipped"),
        device_counting_selected ? "rocprofiler_device_counting_service"
                                 : "disabled",
        device_counting_selected
            ? (input.pm_sample_rows > 0 ? "" : "enabled_but_no_records")
            : (device_counting_requested ? input.plan.reason_code
                                         : "not_selected"),
        device_counting_selected
            ? (input.pm_sample_rows > 0
                   ? "Device-wide AMD hardware-counter samples were collected."
                   : "AMD device counting was selected but emitted no samples "
                     "this session.")
            : "AMD device counting is not available in the current "
              "implementation.");

    AddCapability(
        event, "device_attribution", tracing_requested,
        !tracing_requested
            ? "not_requested"
            : (!input.trace_configured
                   ? "skipped"
                   : (input.unattributed_trace_records > 0 ? "partial"
                                                          : "complete")),
        input.trace_configured ? "rocprofiler_agent_mapping" : "disabled",
        !input.trace_configured
            ? (tracing_requested ? "rocprofiler_buffer_tracing_unavailable"
                                 : "not_selected")
            : (input.unattributed_trace_records > 0
                   ? "rocprofiler_agent_unmapped"
                   : ""),
        !input.trace_configured
            ? "ROCprofiler device attribution was not active."
            : (input.unattributed_trace_records > 0
                   ? "Dropped " +
                         std::to_string(input.unattributed_trace_records) +
                         " trace record(s) whose GPU agent could not be mapped."
                   : "Every GPU trace record was attributed to a device."));

    AddCapability(
        event, "trace_buffer_delivery", tracing_requested,
        !tracing_requested
            ? "not_requested"
            : (!input.trace_configured
                   ? "skipped"
                   : (input.dropped_trace_records > 0 ||
                              input.trace_buffer_flush_failures > 0 ||
                              input.dropped_client_records > 0
                          ? "partial"
                          : "enabled")),
        input.trace_configured ? "rocprofiler_buffer_tracing" : "disabled",
        !input.trace_configured
            ? (tracing_requested ? "rocprofiler_buffer_tracing_unavailable"
                                 : "not_selected")
            : (input.trace_buffer_flush_failures > 0
                   ? "rocprofiler_buffer_flush_failed"
                   : (input.dropped_trace_records > 0
                          ? "rocprofiler_records_dropped"
                          : (input.dropped_client_records > 0
                                 ? "gpufl_activity_queue_full"
                                 : ""))),
        !input.trace_configured
            ? "ROCprofiler trace-buffer delivery was not active."
            : (input.trace_buffer_flush_failures > 0
                   ? "ROCprofiler trace-buffer flush failed " +
                         std::to_string(input.trace_buffer_flush_failures) +
                         " time(s); buffered activity may be incomplete or in the wrong segment."
                   : (input.dropped_trace_records > 0
                          ? "ROCprofiler reported " +
                                std::to_string(input.dropped_trace_records) +
                                " dropped trace record(s); this session is incomplete."
                          : (input.dropped_client_records > 0
                                 ? "GPUFlight dropped " +
                                       std::to_string(input.dropped_client_records) +
                                       " trace record(s) because its activity queue was full."
                                 : "ROCprofiler and GPUFlight reported complete trace delivery."))));

    AddCapability(
        event, "scope_correlation", tracing_requested,
        !tracing_requested
            ? "not_requested"
            : (!input.trace_configured
                   ? "skipped"
                   : (input.scope_correlation_failures > 0 ? "partial" : "enabled")),
        input.trace_configured ? "rocprofiler_external_correlation" : "disabled",
        !input.trace_configured
            ? (tracing_requested ? "rocprofiler_buffer_tracing_unavailable"
                                 : "not_selected")
            : (input.scope_correlation_failures > 0
                   ? "rocprofiler_scope_correlation_failed"
                   : ""),
        !input.trace_configured
            ? "ROCprofiler scope correlation was not active."
            : (input.scope_correlation_failures > 0
                   ? "ROCprofiler scope correlation failed " +
                         std::to_string(input.scope_correlation_failures) +
                         " time(s); some trace rows may have no user scope."
                   : "ROCprofiler accepted every GPUFlight scope-correlation push and pop."));

    return event;
}

}  // namespace gpufl::amd

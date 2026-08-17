#include "gpufl/backends/amd/amd_capture_capabilities.hpp"

#include <string>
#include <utility>

namespace gpufl::amd {
namespace {

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
    event.selected_engine = AmdSelectedPathWireName(input.plan.selected_path);

    const bool tracing_requested =
        input.plan.requested_engine != ProfilingEngine::Monitor;
    const bool dispatch_requested =
        AmdRequestNeedsDispatchCounting(input.plan.requested_engine);
    const bool pc_requested =
        AmdRequestNeedsPcSampling(input.plan.requested_engine);
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

    AddCapability(
        event, "pc_sampling", pc_requested,
        !pc_requested
            ? "not_requested"
            : (pc_selected
                   ? (input.profiling_sample_rows > 0 ? "collected"
                                                      : "enabled_no_data")
                   : "skipped"),
        pc_selected ? "rocprofiler_pc_sampling" : "disabled",
        pc_selected
            ? (input.profiling_sample_rows > 0 ? "" : "enabled_but_no_records")
            : (pc_requested ? input.plan.reason_code : "not_selected"),
        pc_selected
            ? "AMD PC sampling was selected."
            : "AMD PC sampling is not available in the current implementation.");

    AddCapability(
        event, "device_counting", device_counting_requested,
        !device_counting_requested
            ? "not_requested"
            : (device_counting_selected
                   ? (input.profiling_sample_rows > 0 ? "collected"
                                                      : "enabled_no_data")
                   : "skipped"),
        device_counting_selected ? "rocprofiler_device_counting_service"
                                 : "disabled",
        device_counting_selected
            ? (input.profiling_sample_rows > 0 ? "" : "enabled_but_no_records")
            : (device_counting_requested ? input.plan.reason_code
                                         : "not_selected"),
        device_counting_selected
            ? "AMD device counting was selected."
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
                   : (input.dropped_trace_records > 0 ? "partial" : "enabled")),
        input.trace_configured ? "rocprofiler_buffer_tracing" : "disabled",
        !input.trace_configured
            ? (tracing_requested ? "rocprofiler_buffer_tracing_unavailable"
                                 : "not_selected")
            : (input.dropped_trace_records > 0
                   ? "rocprofiler_records_dropped"
                   : ""),
        !input.trace_configured
            ? "ROCprofiler trace-buffer delivery was not active."
            : (input.dropped_trace_records > 0
                   ? "ROCprofiler reported " +
                         std::to_string(input.dropped_trace_records) +
                         " dropped trace record(s); this session is incomplete."
                   : "ROCprofiler reported no dropped trace records."));

    return event;
}

}  // namespace gpufl::amd

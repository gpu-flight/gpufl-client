#include "gpufl/backends/nvidia/capture_capability_resolver.hpp"

#include <string>
#include <utility>
#include <vector>

namespace gpufl {
namespace {

void AddCapability(CaptureCapabilitiesEvent& evt, std::string feature,
                   bool requested, std::string status, std::string mode,
                   std::string reason, std::string message) {
    evt.capabilities.push_back(CaptureCapability{
        std::move(feature), requested, std::move(status), std::move(mode),
        std::move(reason), std::move(message)});
}

}  // namespace

std::vector<std::string> BuildCaptureCapabilityWarnings(
    const CaptureCapabilityInput& input) {
    std::vector<std::string> warnings;
    if (input.requests.pc && input.engine_state.pc.active &&
        !input.engine_state.pc.has_data) {
        warnings.push_back(
            "[gpufl] PC sampling collected 0 stall samples - the profiled "
            "workload was too short for the sampling interval. Run a "
            "longer/heavier workload, or sample more frequently with "
            "`gpufl trace --pc-sample-period <N>` (lower N).");
    }
    if (input.requests.sass && input.engine_state.sass.active &&
        !input.engine_state.sass.has_data) {
        warnings.push_back(
            "[gpufl] SASS metrics collected 0 instruction samples - the "
            "profiled kernels were too short. Run more iterations / a "
            "longer-running kernel to collect instruction-level data.");
    }
    if (input.requests.pm && input.engine_state.pm.active &&
        !input.engine_state.pm.has_data) {
        warnings.push_back(
            "[gpufl] PM sampling collected 0 hardware samples - the profiled "
            "workload was too short for the sampling interval.");
    }
    return warnings;
}

CaptureCapabilitiesEvent BuildCaptureCapabilitiesEvent(
    const CaptureCapabilityInput& input) {
    const bool realKernelRows = input.counters.kernel_rows > 0;
    const bool syntheticKernels =
        (input.requests.sass || input.requests.pc) &&
        input.requested_engine != ProfilingEngine::SassMetrics &&
        input.counters.launch_count > 0 &&
        input.counters.kernel_rows * 2 < input.counters.launch_count;
    const bool kernelHasData = realKernelRows || syntheticKernels;
    const bool memoryHasData = input.counters.memory_rows > 0;
    const bool memTransferHasData = input.counters.mem_transfer_rows > 0;
    const bool syncHasData = input.counters.sync_rows > 0;
    const bool nvtxHasData = input.counters.nvtx_rows > 0;
    const bool graphHasData = input.counters.graph_rows > 0;
    const bool externalHasData = input.counters.external_rows > 0;
    const bool sourceHasData =
        input.counters.source_rows > 0 || input.counters.function_rows > 0;

    std::string selected = ProfilingEngineWireName(input.requested_engine);
    if (input.combo_active) {
        selected = "nvidia.composite";
    } else if (input.requested_engine == ProfilingEngine::Deep) {
        if (input.engine_state.sass.active) selected = "nvidia.sass_metrics";
        else if (input.engine_state.pc.active) selected = "nvidia.pc_sampling";
        else selected = "nvidia.none";
    }

    CaptureCapabilitiesEvent evt;
    evt.session_id = input.session_id;
    evt.ts_ns = input.ts_ns;
    evt.requested_engine = ProfilingEngineWireName(input.requested_engine);
    evt.selected_engine = selected;

    AddCapability(
        evt, "kernel_events", input.kernel_activity,
        kernelHasData
            ? (syntheticKernels ? "fallback" : "collected")
            : (input.kernel_activity
                   ? (input.sass_metrics_only ? "skipped" : "enabled_no_data")
                   : "not_requested"),
        input.sass_metrics_only
            ? "sass_metrics_only"
            : (syntheticKernels
                   ? "launch_callbacks_synthetic"
                   : ((realKernelRows || input.kernel_activity)
                          ? "cupti_activity"
                          : "disabled")),
        kernelHasData
            ? (syntheticKernels
                   ? (input.requests.pc
                          ? "cupti_kernel_activity_conflicts_with_pc_sampling"
                          : "cupti_kernel_activity_deadlock_risk")
                   : "")
            : (input.kernel_activity
                   ? (input.sass_metrics_only
                          ? "disabled_to_preserve_sass_counters"
                          : "enabled_but_no_records")
                   : "not_selected"),
        kernelHasData
            ? (syntheticKernels
                   ? (input.requests.pc
                          ? "Kernel rows were collected from launch callbacks; durations are estimated because CUPTI kernel activity is disabled while PC Sampling API is active."
                          : "Kernel rows were collected from launch callbacks; durations are estimated because CUPTI kernel activity is disabled in SASS safe mode.")
                   : "Kernel rows were collected from CUPTI kernel activity records.")
            : (!input.kernel_activity
                   ? "Kernel timeline activity was not requested by the selected engine domains."
                   : input.sass_metrics_only
                         ? "Kernel activity was intentionally disabled because CUPTI SASS Metrics requires metrics-only mode on this GPU/driver to produce non-zero counters."
                         : "Kernel tracing was enabled but emitted no kernel rows this session."));
    AddCapability(
        evt, "kernel_names", input.kernel_activity,
        kernelHasData
            ? (syntheticKernels ? "partial" : "collected")
            : (input.kernel_activity
                   ? (input.sass_metrics_only ? "skipped" : "enabled_no_data")
                   : "not_requested"),
        input.sass_metrics_only
            ? "sass_metrics_only"
            : (syntheticKernels
                   ? "callback_symbol_probe"
                   : ((realKernelRows || input.kernel_activity)
                          ? "cupti_activity_name"
                          : "disabled")),
        kernelHasData
            ? (syntheticKernels ? "symbol_name_may_be_unavailable" : "")
            : (input.kernel_activity
                   ? (input.sass_metrics_only
                          ? "disabled_to_preserve_sass_counters"
                          : "enabled_but_no_records")
                   : "not_selected"),
        kernelHasData
            ? (syntheticKernels
                   ? "Kernel names use CUPTI callback symbolName when safely readable, otherwise the CUDA launch API name."
                   : "Kernel names came from CUPTI activity records.")
            : (!input.kernel_activity
                   ? "Kernel name tracing was not requested by the selected engine domains."
                   : input.sass_metrics_only
                         ? "Kernel name tracing was intentionally disabled to keep SASS metric counters valid."
                         : "Kernel name capture was enabled but no kernel rows were emitted."));
    AddCapability(
        evt, "kernel_details", input.kernel_activity,
        kernelHasData
            ? (syntheticKernels ? "partial" : "collected")
            : (input.kernel_activity
                   ? (input.sass_metrics_only ? "skipped" : "enabled_no_data")
                   : "not_requested"),
        input.sass_metrics_only
            ? "sass_metrics_only"
            : (syntheticKernels
                   ? "launch_callback_params"
                   : ((realKernelRows || input.kernel_activity)
                          ? "cupti_activity_details"
                          : "disabled")),
        kernelHasData
            ? (syntheticKernels ? "activity_details_unavailable" : "")
            : (input.kernel_activity
                   ? (input.sass_metrics_only
                          ? "disabled_to_preserve_sass_counters"
                          : "enabled_but_no_records")
                   : "not_selected"),
        kernelHasData
            ? (syntheticKernels
                   ? "Grid/block parameters are captured from launch callbacks; register and occupancy details may be unavailable."
                   : "Kernel details came from CUPTI activity records and launch metadata.")
            : (!input.kernel_activity
                   ? "Kernel detail tracing was not requested by the selected engine domains."
                   : input.sass_metrics_only
                         ? "Kernel detail tracing was intentionally disabled to keep SASS metric counters valid."
                         : "Kernel detail capture was enabled but no kernel rows were emitted."));
    AddCapability(
        evt, "memcpy_activity", input.kernel_activity,
        input.kernel_activity
            ? (memTransferHasData ? "collected" : "enabled_no_data")
            : "not_requested",
        input.kernel_activity ? "cupti_memcpy_activity" : "disabled",
        input.kernel_activity
            ? (memTransferHasData ? "" : "enabled_but_no_records")
            : "not_selected",
        input.kernel_activity
            ? (memTransferHasData
                   ? "Memcpy/memset activity records were collected."
                   : "Memcpy/memset activity was enabled but emitted no rows this session.")
            : "Memcpy/memset timeline activity was not requested by the selected engine domains.");
    const bool syncRequested = input.kernel_activity &&
        input.options.enable_synchronization &&
        input.allow_sass_sync_activity;
    AddCapability(
        evt, "sync_activity", syncRequested,
        syncRequested
            ? (syncHasData ? "collected" : "enabled_no_data")
            : (input.options.enable_synchronization ? "skipped" : "not_requested"),
        syncRequested ? "cupti_synchronization" : "disabled",
        syncRequested
            ? (syncHasData ? "" : "enabled_but_no_records")
            : (input.kernel_activity ? "disabled_by_policy" : "not_selected"),
        syncRequested
            ? (syncHasData
                   ? "CUDA synchronization activity records were collected."
                   : "CUDA synchronization activity was enabled but emitted no rows this session.")
            : "CUDA synchronization timeline activity was not collected.");
    AddCapability(
        evt, "nvtx_markers", input.kernel_activity,
        input.kernel_activity
            ? (nvtxHasData ? "collected" : "enabled_no_data")
            : "not_requested",
        input.kernel_activity ? "cupti_marker_activity" : "disabled",
        input.kernel_activity
            ? (nvtxHasData ? "" : "enabled_but_no_records")
            : "not_selected",
        input.kernel_activity
            ? (nvtxHasData
                   ? "NVTX marker activity records were collected."
                   : "NVTX marker activity was enabled but emitted no completed ranges this session.")
            : "NVTX timeline activity was not requested by the selected engine domains.");
    const bool graphRequested = input.kernel_activity &&
        input.options.enable_cuda_graphs_tracking &&
        input.allow_sass_graph_activity;
    AddCapability(
        evt, "graph_activity", graphRequested,
        graphRequested
            ? (graphHasData ? "collected" : "enabled_no_data")
            : (input.options.enable_cuda_graphs_tracking ? "skipped" : "not_requested"),
        graphRequested ? "cupti_graph_trace" : "disabled",
        graphRequested
            ? (graphHasData ? "" : "enabled_but_no_records")
            : (input.kernel_activity ? "disabled_by_option" : "not_selected"),
        graphRequested
            ? (graphHasData
                   ? "CUDA graph launch activity records were collected."
                   : "CUDA graph launch activity was enabled but emitted no rows this session.")
            : "CUDA graph launch timeline activity was not collected.");
    AddCapability(
        evt, "cubin_disassembly", input.cubin_requested,
        input.cubin_capture
            ? "collected"
            : (input.cubin_requested ? "skipped" : "not_requested"),
        input.cubin_capture ? "module_resource_callbacks" : "disabled",
        input.cubin_requested && !input.cubin_capture
            ? "cubin_capture_disabled"
            : "",
        input.cubin_capture
            ? "CUBINs were captured for offline SASS disassembly."
            : (input.cubin_requested
                   ? "This profiling path requested CUBIN capture, but it was disabled by policy or environment."
                   : "This profiling engine does not request CUBIN capture."));
    AddCapability(
        evt, "sass_metrics", input.requests.sass,
        input.engine_state.sass.active
            ? (input.engine_state.sass.has_data ? "collected" : "enabled_no_data")
            : (input.requests.sass ? "skipped" : "not_requested"),
        input.engine_state.sass.active ? "cupti_sass_metrics" : "disabled",
        input.engine_state.sass.active
            ? (input.engine_state.sass.has_data ? "" : "enabled_but_no_samples")
            : "not_selected_or_not_operational",
        input.engine_state.sass.active
            ? (input.engine_state.sass.has_data
                   ? "SASS metrics were collected for this session."
                   : "SASS metrics were enabled but produced no instruction-level samples this session (e.g. kernels too short, or CUPTI replay returned no data).")
            : "SASS metrics were not collected for this session.");

    const char* pcInactiveReason =
        input.pc_insufficient_privileges ? "insufficient_privilege"
        : input.pc_stall_reasons_unavailable ? "no_stall_reasons"
        : input.pc_no_cuda_context ? "no_cuda_context"
                                  : "not_selected_or_not_operational";
    const char* pcInactiveMessage =
        input.pc_insufficient_privileges
            ? "PC sampling was blocked by GPU profiling permissions - enable "
              "\"GPU performance counters for all users\" in the NVIDIA "
              "Control Panel or run elevated."
        : input.pc_stall_reasons_unavailable
            ? "The driver exposed no PC sampling stall reasons "
              "(cuptiPCSamplingGetNumStallReasons returned 0). This usually "
              "means the CUPTI runtime bundled with gpufl is older than the "
              "installed display driver supports - update gpufl (or the "
              "CUDA toolkit it was built with) to match the driver "
              "generation."
        : input.pc_no_cuda_context
            ? "PC sampling never started because the target process did not "
              "create a CUDA context."
            : "PC sampling was not collected for this session.";
    AddCapability(
        evt, "pc_sampling", input.requests.pc,
        input.engine_state.pc.active
            ? (input.engine_state.pc.has_data ? "collected" : "enabled_no_data")
            : (input.requested_engine == ProfilingEngine::Deep &&
                       input.engine_state.sass.active
                   ? "skipped"
                   : (input.requests.pc ? "skipped" : "not_requested")),
        input.engine_state.pc.active ? "cupti_pc_sampling" : "disabled",
        input.requested_engine == ProfilingEngine::Deep &&
                input.engine_state.sass.active
            ? "mutually_exclusive_with_sass_metrics"
            : (input.engine_state.pc.active
                   ? (input.engine_state.pc.has_data
                          ? ""
                          : "enabled_but_no_samples")
                   : pcInactiveReason),
        input.requested_engine == ProfilingEngine::Deep &&
                input.engine_state.sass.active
            ? "Deep selected SASS metrics; PC sampling was skipped because SASS metrics and PC sampling are mutually exclusive in one run."
            : (input.engine_state.pc.active
                   ? (input.engine_state.pc.has_data
                          ? "PC sampling was collected for this session."
                          : "PC sampling was enabled but produced no stall samples this session (e.g. kernels too short for the sampling period).")
                   : pcInactiveMessage));
    AddCapability(
        evt, "pm_sampling", input.requests.pm,
        input.engine_state.pm.active
            ? (input.engine_state.pm.has_data ? "collected" : "enabled_no_data")
            : (input.requests.pm ? "skipped" : "not_requested"),
        input.engine_state.pm.active ? "cupti_pm_sampling" : "disabled",
        input.engine_state.pm.active
            ? (input.engine_state.pm.has_data ? "" : "enabled_but_no_samples")
            : "not_selected_or_not_operational",
        input.engine_state.pm.active
            ? (input.engine_state.pm.has_data
                   ? "PM sampling hardware metric samples were collected for this session."
                   : "PM sampling was enabled but produced no hardware samples this session.")
            : "PM sampling was not collected for this session.");
    AddCapability(
        evt, "range_counters", input.requests.range,
        input.engine_state.range.active
            ? (input.engine_state.range.has_data ? "collected" : "enabled_no_data")
            : (input.requests.range ? "skipped" : "not_requested"),
        input.engine_state.range.active ? "cupti_range_profiler" : "disabled",
        input.engine_state.range.active
            ? (input.engine_state.range.has_data ? "" : "enabled_but_no_ranges")
            : "not_selected_or_not_operational",
        input.engine_state.range.active
            ? (input.engine_state.range.has_data
                   ? "Range Profiler scope-level hardware counters were collected for this session."
                   : "Range Profiler was enabled but produced no decoded range counters this session.")
            : "Range Profiler counters were not collected for this session.");
    AddCapability(
        evt, "kernel_replay_counters", input.requests.range_kernel,
        input.engine_state.range_kernel.active
            ? (input.engine_state.range_kernel.has_data ? "collected" : "enabled_no_data")
            : (input.requests.range_kernel ? "skipped" : "not_requested"),
        input.engine_state.range_kernel.active
            ? "cupti_range_profiler_kernel_replay"
            : "disabled",
        input.engine_state.range_kernel.active
            ? (input.engine_state.range_kernel.has_data
                   ? ""
                   : "enabled_but_no_ranges")
            : "not_selected_or_not_operational",
        input.engine_state.range_kernel.active
            ? (input.engine_state.range_kernel.has_data
                   ? "Range Profiler kernel replay counters were collected for this session."
                   : "Range Profiler kernel replay was enabled but produced no decoded kernel ranges this session.")
            : "Range Profiler kernel replay counters were not collected for this session.");
    AddCapability(
        evt, "source_correlation", input.engine_state.pc.active,
        input.engine_state.pc.active
            ? (sourceHasData ? "collected" : "enabled_no_data")
            : (input.engine_state.sass.active ? "skipped" : "not_requested"),
        input.engine_state.pc.active ? "pc_sampling_source_locator" : "disabled",
        input.engine_state.pc.active
            ? (sourceHasData ? "" : "enabled_but_no_records")
            : (input.engine_state.sass.active
                   ? "sass_metrics_have_no_source_lines"
                   : "not_requested"),
        input.engine_state.pc.active
            ? (sourceHasData
                   ? "PC sampling source locator/function records were collected for CUDA source correlation."
                   : "PC sampling source correlation was enabled but emitted no source locator/function records.")
            : "CUDA source-line correlation was not collected in this session.");
    const bool memoryRequestedAndAllowed =
        input.options.enable_memory_tracking &&
        input.allow_sass_memory2_activity;
    AddCapability(
        evt, "memory_activity", input.options.enable_memory_tracking,
        memoryRequestedAndAllowed
            ? (memoryHasData ? "collected" : "enabled_no_data")
            : (input.options.enable_memory_tracking ? "skipped" : "not_requested"),
        memoryRequestedAndAllowed ? "cupti_memory" : "disabled",
        memoryRequestedAndAllowed
            ? (memoryHasData ? "" : "enabled_but_no_records")
            : (input.options.enable_memory_tracking &&
                       !input.allow_sass_memory2_activity
                   ? "sass_safe_mode_memory_activity_disabled"
                   : ""),
        memoryRequestedAndAllowed
            ? (memoryHasData
                   ? "CUPTI memory activity records were collected."
                   : "CUPTI memory activity was enabled but emitted no memory rows this session.")
            : "CUPTI memory activity was not collected.");
    const bool externalRequestedAndAllowed =
        input.kernel_activity && input.options.enable_external_correlation &&
        input.allow_sass_external_correlation;
    AddCapability(
        evt, "external_correlation",
        input.kernel_activity && input.options.enable_external_correlation,
        externalRequestedAndAllowed
            ? (externalHasData ? "collected" : "enabled_no_data")
            : (input.options.enable_external_correlation
                   ? (input.kernel_activity ? "skipped" : "not_requested")
                   : "not_requested"),
        externalRequestedAndAllowed ? "cupti_external_correlation" : "disabled",
        externalRequestedAndAllowed
            ? (externalHasData ? "" : "enabled_but_no_records")
            : (input.kernel_activity && input.options.enable_external_correlation &&
                       !input.allow_sass_external_correlation
                   ? "sass_safe_mode_external_correlation_disabled"
                   : ""),
        externalRequestedAndAllowed
            ? (externalHasData
                   ? "Framework external correlation records were collected."
                   : "Framework external correlation was enabled but emitted no records this session.")
            : "Framework external correlation was not collected.");

    return evt;
}

}  // namespace gpufl

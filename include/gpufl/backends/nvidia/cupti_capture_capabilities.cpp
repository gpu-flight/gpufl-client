#include "gpufl/backends/nvidia/cupti_backend.hpp"

#include <atomic>
#include <string>
#include <utility>

#include "gpufl/backends/nvidia/cupti_engine_selection.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/logger/logger.hpp"
#include "gpufl/core/model/lifecycle_model.hpp"
#include "gpufl/core/runtime.hpp"

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

void CuptiBackend::EmitCaptureCapabilities_() const {
    const Runtime* rt = runtime();
    if (!(rt && rt->logger)) return;
    if (capture_capabilities_emitted_.exchange(true, std::memory_order_acq_rel)) {
        return;
    }

    const EngineRequestSet requests =
        BuildEngineRequestSet(opts_.profiling_engine, combo_);
    const bool kernelActivity = collectsKernelEvents();
    const bool cubinRequested = requests.needsCubin();
    const bool cubinCapture = NeedsCubinCapture();
    // Capability emission happens after final engine shutdown so the report can
    // see late-flushed samples. Some engines drop their operational flag during
    // shutdown, so keep a path active if it was requested and produced data.
    const EngineRuntimeState engineState =
        InspectEngineRuntimeState(engine_.get(), opts_.profiling_engine,
                                  comboActive());

    // Did each path actually emit rows (vs merely arm)? Drives the
    // "enabled_no_data" status so a capability that was turned ON but produced
    // zero records is reported honestly instead of as "collected".
    const uint64_t kernelRows =
        kernel_activity_emitted_.load(std::memory_order_relaxed);
    const uint64_t memoryRows =
        memory_activity_emitted_.load(std::memory_order_relaxed);
    const uint64_t memTransferRows =
        mem_transfer_activity_emitted_.load(std::memory_order_relaxed);
    const uint64_t syncRows =
        sync_activity_emitted_.load(std::memory_order_relaxed);
    const uint64_t nvtxRows =
        nvtx_marker_emitted_.load(std::memory_order_relaxed);
    const uint64_t graphRows =
        graph_activity_emitted_.load(std::memory_order_relaxed);
    const uint64_t externalRows =
        external_correlation_seen_.load(std::memory_order_relaxed);
    const uint64_t sourceRows =
        source_locator_seen_.load(std::memory_order_relaxed);
    const uint64_t functionRows =
        function_record_seen_.load(std::memory_order_relaxed);
    // Real CUPTI kernel-activity records emitted this session (MEASURED GPU
    // durations). PcSampling enables CONCURRENT_KERNEL out-of-band in its engine
    // (bypassing collectsKernelEvents()), so SOME real records can flow even for
    // PC sessions - e.g. when the GPUFL_PC_KERNEL_COLLECT=all heavy drain's
    // Stop->flush->restart cycle succeeds for a handful of launches.
    const uint64_t launchCount =
        kernel_launch_callback_count_.load(std::memory_order_acquire);
    const bool realKernelRows = kernelRows > 0;
    // The rest of the launches get SYNTHETIC rows - launch-callback duration
    // ESTIMATES (host dispatch gaps, not measured) from drainSyntheticKernels.
    // Report "synthetic" when they're the MAJORITY (fewer than half the launches
    // got a real record): a binary "no real record at all" test under-reported -
    // when the PC heavy drain captured a handful of real records it flipped the
    // whole session to "collected" while the timeline was still ~90% host-gap
    // estimates. launchCount and kernelRows are both populated DURING the run, so
    // this is correct whether capabilities emit before or after the end-of-session
    // synthetic drain. (Pure SASS-metrics mode keeps no kernel rows at all.)
    const bool syntheticKernels =
        (requests.sass || requests.pc) &&
        opts_.profiling_engine != ProfilingEngine::SassMetrics &&
        launchCount > 0 &&
        kernelRows * 2 < launchCount;
    // "Has kernel data" = any kernel rows, real OR synthetic. The status then
    // splits collected (measured) vs fallback (mostly estimated) on syntheticKernels.
    const bool kernelHasData = realKernelRows || syntheticKernels;
    const bool memoryHasData = memoryRows > 0;
    const bool memTransferHasData = memTransferRows > 0;
    const bool syncHasData = syncRows > 0;
    const bool nvtxHasData = nvtxRows > 0;
    const bool graphHasData = graphRows > 0;
    const bool externalHasData = externalRows > 0;
    const bool sourceHasData = sourceRows > 0 || functionRows > 0;

    std::string selected = ProfilingEngineWireName(opts_.profiling_engine);
    if (comboActive()) {
        selected = "nvidia.composite";
    } else if (opts_.profiling_engine == ProfilingEngine::Deep) {
        if (engineState.sass.active) selected = "nvidia.sass_metrics";
        else if (engineState.pc.active) selected = "nvidia.pc_sampling";
        else selected = "nvidia.none";
    }

    CaptureCapabilitiesEvent evt;
    evt.session_id = rt->session_id;
    evt.ts_ns = detail::GetTimestampNs();
    evt.requested_engine = ProfilingEngineWireName(opts_.profiling_engine);
    evt.selected_engine = selected;

    const bool metricsOnly = SassMetricsOnlyMode();
    AddCapability(evt, "kernel_events", kernelActivity,
                  kernelHasData
                      ? (syntheticKernels ? "fallback" : "collected")
                      : (kernelActivity
                            ? (metricsOnly ? "skipped" : "enabled_no_data")
                            : "not_requested"),
                  metricsOnly
                      ? "sass_metrics_only"
                      : (syntheticKernels ? "launch_callbacks_synthetic"
                         : ((realKernelRows || kernelActivity) ? "cupti_activity"
                                                               : "disabled")),
                  kernelHasData
                      ? (syntheticKernels
                            ? (requests.pc ? "cupti_kernel_activity_conflicts_with_pc_sampling"
                                           : "cupti_kernel_activity_deadlock_risk")
                            : "")
                      : (kernelActivity
                            ? (metricsOnly ? "disabled_to_preserve_sass_counters"
                                           : "enabled_but_no_records")
                            : "not_selected"),
                  kernelHasData
                      ? (syntheticKernels
                            ? (requests.pc
                                  ? "Kernel rows were collected from launch callbacks; durations are estimated because CUPTI kernel activity is disabled while PC Sampling API is active."
                                  : "Kernel rows were collected from launch callbacks; durations are estimated because CUPTI kernel activity is disabled in SASS safe mode.")
                            : "Kernel rows were collected from CUPTI kernel activity records.")
                      : (!kernelActivity
                            ? "Kernel timeline activity was not requested by the selected engine domains."
                            : metricsOnly
                            ? "Kernel activity was intentionally disabled because CUPTI SASS Metrics requires metrics-only mode on this GPU/driver to produce non-zero counters."
                            : "Kernel tracing was enabled but emitted no kernel rows this session."));
    AddCapability(evt, "kernel_names", kernelActivity,
                  kernelHasData
                      ? (syntheticKernels ? "partial" : "collected")
                      : (kernelActivity
                            ? (metricsOnly ? "skipped" : "enabled_no_data")
                            : "not_requested"),
                  metricsOnly
                      ? "sass_metrics_only"
                      : (syntheticKernels ? "callback_symbol_probe"
                         : ((realKernelRows || kernelActivity) ? "cupti_activity_name"
                                                               : "disabled")),
                  kernelHasData
                      ? (syntheticKernels ? "symbol_name_may_be_unavailable" : "")
                      : (kernelActivity
                            ? (metricsOnly ? "disabled_to_preserve_sass_counters"
                                           : "enabled_but_no_records")
                            : "not_selected"),
                  kernelHasData
                      ? (syntheticKernels
                            ? "Kernel names use CUPTI callback symbolName when safely readable, otherwise the CUDA launch API name."
                            : "Kernel names came from CUPTI activity records.")
                      : (!kernelActivity
                            ? "Kernel name tracing was not requested by the selected engine domains."
                            : metricsOnly
                            ? "Kernel name tracing was intentionally disabled to keep SASS metric counters valid."
                            : "Kernel name capture was enabled but no kernel rows were emitted."));
    AddCapability(evt, "kernel_details", kernelActivity,
                  kernelHasData
                      ? (syntheticKernels ? "partial" : "collected")
                      : (kernelActivity
                            ? (metricsOnly ? "skipped" : "enabled_no_data")
                            : "not_requested"),
                  metricsOnly
                      ? "sass_metrics_only"
                      : (syntheticKernels ? "launch_callback_params"
                         : ((realKernelRows || kernelActivity) ? "cupti_activity_details"
                                                               : "disabled")),
                  kernelHasData
                      ? (syntheticKernels ? "activity_details_unavailable" : "")
                      : (kernelActivity
                            ? (metricsOnly ? "disabled_to_preserve_sass_counters"
                                           : "enabled_but_no_records")
                            : "not_selected"),
                  kernelHasData
                      ? (syntheticKernels
                            ? "Grid/block parameters are captured from launch callbacks; register and occupancy details may be unavailable."
                            : "Kernel details came from CUPTI activity records and launch metadata.")
                      : (!kernelActivity
                            ? "Kernel detail tracing was not requested by the selected engine domains."
                            : metricsOnly
                            ? "Kernel detail tracing was intentionally disabled to keep SASS metric counters valid."
                            : "Kernel detail capture was enabled but no kernel rows were emitted."));
    AddCapability(evt, "memcpy_activity", kernelActivity,
                  kernelActivity
                      ? (memTransferHasData ? "collected" : "enabled_no_data")
                      : "not_requested",
                  kernelActivity ? "cupti_memcpy_activity" : "disabled",
                  kernelActivity
                      ? (memTransferHasData ? "" : "enabled_but_no_records")
                      : "not_selected",
                  kernelActivity
                      ? (memTransferHasData
                            ? "Memcpy/memset activity records were collected."
                            : "Memcpy/memset activity was enabled but emitted no rows this session.")
                      : "Memcpy/memset timeline activity was not requested by the selected engine domains.");
    const bool syncRequested =
        kernelActivity && opts_.enable_synchronization && AllowSassSyncActivity();
    AddCapability(evt, "sync_activity", syncRequested,
                  syncRequested
                      ? (syncHasData ? "collected" : "enabled_no_data")
                      : (opts_.enable_synchronization ? "skipped" : "not_requested"),
                  syncRequested ? "cupti_synchronization" : "disabled",
                  syncRequested
                      ? (syncHasData ? "" : "enabled_but_no_records")
                      : (kernelActivity ? "disabled_by_policy" : "not_selected"),
                  syncRequested
                      ? (syncHasData
                            ? "CUDA synchronization activity records were collected."
                            : "CUDA synchronization activity was enabled but emitted no rows this session.")
                      : "CUDA synchronization timeline activity was not collected.");
    AddCapability(evt, "nvtx_markers", kernelActivity,
                  kernelActivity
                      ? (nvtxHasData ? "collected" : "enabled_no_data")
                      : "not_requested",
                  kernelActivity ? "cupti_marker_activity" : "disabled",
                  kernelActivity
                      ? (nvtxHasData ? "" : "enabled_but_no_records")
                      : "not_selected",
                  kernelActivity
                      ? (nvtxHasData
                            ? "NVTX marker activity records were collected."
                            : "NVTX marker activity was enabled but emitted no completed ranges this session.")
                      : "NVTX timeline activity was not requested by the selected engine domains.");
    const bool graphRequested =
        kernelActivity && opts_.enable_cuda_graphs_tracking && AllowSassGraphActivity();
    AddCapability(evt, "graph_activity", graphRequested,
                  graphRequested
                      ? (graphHasData ? "collected" : "enabled_no_data")
                      : (opts_.enable_cuda_graphs_tracking ? "skipped" : "not_requested"),
                  graphRequested ? "cupti_graph_trace" : "disabled",
                  graphRequested
                      ? (graphHasData ? "" : "enabled_but_no_records")
                      : (kernelActivity ? "disabled_by_option" : "not_selected"),
                  graphRequested
                      ? (graphHasData
                            ? "CUDA graph launch activity records were collected."
                            : "CUDA graph launch activity was enabled but emitted no rows this session.")
                      : "CUDA graph launch timeline activity was not collected.");
    AddCapability(evt, "cubin_disassembly", cubinRequested,
                  cubinCapture ? "collected" :
                      (cubinRequested ? "skipped" : "not_requested"),
                  cubinCapture ? "module_resource_callbacks" : "disabled",
                  cubinRequested && !cubinCapture ? "cubin_capture_disabled" : "",
                  cubinCapture
                      ? "CUBINs were captured for offline SASS disassembly."
                      : (cubinRequested
                            ? "This profiling path requested CUBIN capture, but it was disabled by policy or environment."
                            : "This profiling engine does not request CUBIN capture."));
    AddCapability(evt, "sass_metrics",
                  requests.sass,
                  engineState.sass.active
                      ? (engineState.sass.has_data ? "collected" : "enabled_no_data")
                      : (requests.sass ? "skipped" : "not_requested"),
                  engineState.sass.active ? "cupti_sass_metrics" : "disabled",
                  engineState.sass.active
                      ? (engineState.sass.has_data ? "" : "enabled_but_no_samples")
                      : "not_selected_or_not_operational",
                  engineState.sass.active
                      ? (engineState.sass.has_data
                            ? "SASS metrics were collected for this session."
                            : "SASS metrics were enabled but produced no instruction-level samples this session (e.g. kernels too short, or CUPTI replay returned no data).")
                      : "SASS metrics were not collected for this session.");
    // Distinguish WHY PC sampling ended up inactive - "skipped" alone sent
    // earlier sessions on a wild goose chase (the real cause was the engine
    // never getting a CUDA context under Windows injection).
    const bool pcPrivBlocked = engine_ && engine_->hasInsufficientPrivileges();
    const bool pcNoStallReasons = engine_ && engine_->stallReasonsUnavailable();
    const bool pcNoContext =
        engine_start_pending_.load(std::memory_order_acquire);
    const char* pcInactiveReason =
        pcPrivBlocked      ? "insufficient_privilege"
        : pcNoStallReasons ? "no_stall_reasons"
        : pcNoContext      ? "no_cuda_context"
                           : "not_selected_or_not_operational";
    const char* pcInactiveMessage =
        pcPrivBlocked
            ? "PC sampling was blocked by GPU profiling permissions - enable "
              "\"GPU performance counters for all users\" in the NVIDIA "
              "Control Panel or run elevated."
        : pcNoStallReasons
            ? "The driver exposed no PC sampling stall reasons "
              "(cuptiPCSamplingGetNumStallReasons returned 0). This usually "
              "means the CUPTI runtime bundled with gpufl is older than the "
              "installed display driver supports - update gpufl (or the "
              "CUDA toolkit it was built with) to match the driver "
              "generation."
        : pcNoContext
            ? "PC sampling never started because the target process did not "
              "create a CUDA context."
            : "PC sampling was not collected for this session.";
    AddCapability(evt, "pc_sampling",
                  requests.pc,
                  engineState.pc.active
                      ? (engineState.pc.has_data ? "collected" : "enabled_no_data")
                      : (opts_.profiling_engine == ProfilingEngine::Deep &&
                                 engineState.sass.active
                             ? "skipped"
                             : (requests.pc ? "skipped" : "not_requested")),
                  engineState.pc.active ? "cupti_pc_sampling" : "disabled",
                  opts_.profiling_engine == ProfilingEngine::Deep &&
                          engineState.sass.active
                      ? "mutually_exclusive_with_sass_metrics" :
                    (engineState.pc.active
                         ? (engineState.pc.has_data ? "" : "enabled_but_no_samples")
                         : pcInactiveReason),
                  opts_.profiling_engine == ProfilingEngine::Deep &&
                          engineState.sass.active
                      ? "Deep selected SASS metrics; PC sampling was skipped because SASS metrics and PC sampling are mutually exclusive in one run."
                      : (engineState.pc.active ? (engineState.pc.has_data
                            ? "PC sampling was collected for this session."
                            : "PC sampling was enabled but produced no stall samples this session (e.g. kernels too short for the sampling period).")
                                  : pcInactiveMessage));
    AddCapability(evt, "pm_sampling",
                  requests.pm,
                  engineState.pm.active
                      ? (engineState.pm.has_data ? "collected" : "enabled_no_data")
                      : (requests.pm ? "skipped" : "not_requested"),
                  engineState.pm.active ? "cupti_pm_sampling" : "disabled",
                  engineState.pm.active
                      ? (engineState.pm.has_data ? "" : "enabled_but_no_samples")
                      : "not_selected_or_not_operational",
                  engineState.pm.active
                      ? (engineState.pm.has_data
                            ? "PM sampling hardware metric samples were collected for this session."
                            : "PM sampling was enabled but produced no hardware samples this session.")
                      : "PM sampling was not collected for this session.");
    AddCapability(evt, "range_counters",
                  requests.range,
                  engineState.range.active
                      ? (engineState.range.has_data ? "collected" : "enabled_no_data")
                      : (requests.range ? "skipped" : "not_requested"),
                  engineState.range.active ? "cupti_range_profiler" : "disabled",
                  engineState.range.active
                      ? (engineState.range.has_data ? "" : "enabled_but_no_ranges")
                      : "not_selected_or_not_operational",
                  engineState.range.active
                      ? (engineState.range.has_data
                            ? "Range Profiler scope-level hardware counters were collected for this session."
                            : "Range Profiler was enabled but produced no decoded range counters this session.")
                      : "Range Profiler counters were not collected for this session.");
    AddCapability(evt, "kernel_replay_counters",
                  requests.range_kernel,
                  engineState.range_kernel.active
                      ? (engineState.range_kernel.has_data ? "collected" : "enabled_no_data")
                      : (requests.range_kernel ? "skipped" : "not_requested"),
                  engineState.range_kernel.active
                      ? "cupti_range_profiler_kernel_replay" : "disabled",
                  engineState.range_kernel.active
                      ? (engineState.range_kernel.has_data ? "" : "enabled_but_no_ranges")
                      : "not_selected_or_not_operational",
                  engineState.range_kernel.active
                      ? (engineState.range_kernel.has_data
                            ? "Range Profiler kernel replay counters were collected for this session."
                            : "Range Profiler kernel replay was enabled but produced no decoded kernel ranges this session.")
                      : "Range Profiler kernel replay counters were not collected for this session.");
    AddCapability(evt, "source_correlation", engineState.pc.active,
                  engineState.pc.active
                      ? (sourceHasData ? "collected" : "enabled_no_data")
                      : (engineState.sass.active ? "skipped" : "not_requested"),
                  engineState.pc.active ? "pc_sampling_source_locator" : "disabled",
                  engineState.pc.active
                      ? (sourceHasData ? "" : "enabled_but_no_records")
                      : (engineState.sass.active
                             ? "sass_metrics_have_no_source_lines"
                             : "not_requested"),
                  engineState.pc.active
                      ? (sourceHasData
                            ? "PC sampling source locator/function records were collected for CUDA source correlation."
                            : "PC sampling source correlation was enabled but emitted no source locator/function records.")
                      : "CUDA source-line correlation was not collected in this session.");
    const bool memoryRequestedAndAllowed =
        opts_.enable_memory_tracking && AllowSassMemory2Activity();
    AddCapability(evt, "memory_activity",
                  opts_.enable_memory_tracking,
                  memoryRequestedAndAllowed
                      ? (memoryHasData ? "collected" : "enabled_no_data")
                      : (opts_.enable_memory_tracking ? "skipped" : "not_requested"),
                  memoryRequestedAndAllowed ? "cupti_memory" : "disabled",
                  memoryRequestedAndAllowed
                      ? (memoryHasData ? "" : "enabled_but_no_records")
                      : (opts_.enable_memory_tracking && !AllowSassMemory2Activity()
                            ? "sass_safe_mode_memory_activity_disabled" : ""),
                  memoryRequestedAndAllowed
                      ? (memoryHasData
                            ? "CUPTI memory activity records were collected."
                            : "CUPTI memory activity was enabled but emitted no memory rows this session.")
                      : "CUPTI memory activity was not collected.");
    const bool externalRequestedAndAllowed =
        kernelActivity && opts_.enable_external_correlation &&
        AllowSassExternalCorrelation();
    AddCapability(evt, "external_correlation",
                  kernelActivity && opts_.enable_external_correlation,
                  externalRequestedAndAllowed
                      ? (externalHasData ? "collected" : "enabled_no_data")
                      : (opts_.enable_external_correlation
                            ? (kernelActivity ? "skipped" : "not_requested")
                            : "not_requested"),
                  externalRequestedAndAllowed ? "cupti_external_correlation" : "disabled",
                  externalRequestedAndAllowed
                      ? (externalHasData ? "" : "enabled_but_no_records")
                      : (kernelActivity && opts_.enable_external_correlation && !AllowSassExternalCorrelation()
                            ? "sass_safe_mode_external_correlation_disabled" : ""),
                  externalRequestedAndAllowed
                      ? (externalHasData
                            ? "Framework external correlation records were collected."
                            : "Framework external correlation was enabled but emitted no records this session.")
                      : "Framework external correlation was not collected.");

    rt->logger->write(model::CaptureCapabilitiesModel(evt));
}

}  // namespace gpufl

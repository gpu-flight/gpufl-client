#include "gpufl/backends/nvidia/cupti_backend.hpp"

#include <atomic>
#include <cstdio>
#include <string>

#include "gpufl/backends/nvidia/capture_capability_resolver.hpp"
#include "gpufl/backends/nvidia/cupti_engine_selection.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/logger/logger.hpp"
#include "gpufl/core/model/lifecycle_model.hpp"
#include "gpufl/core/runtime.hpp"

namespace gpufl {

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

    CaptureCapabilityInput input;
    input.session_id = rt->session_id;
    input.ts_ns = detail::GetTimestampNs();
    input.requested_engine = opts_.profiling_engine;
    input.combo_active = comboActive();
    input.requests = requests;
    input.engine_state = engineState;
    input.kernel_activity = kernelActivity;
    input.cubin_requested = cubinRequested;
    input.cubin_capture = cubinCapture;
    input.sass_metrics_only = SassMetricsOnlyMode();
    input.allow_sass_memory2_activity = AllowSassMemory2Activity();
    input.allow_sass_sync_activity = AllowSassSyncActivity();
    input.allow_sass_graph_activity = AllowSassGraphActivity();
    input.allow_sass_external_correlation = AllowSassExternalCorrelation();
    input.options.enable_memory_tracking = opts_.enable_memory_tracking;
    input.options.enable_external_correlation = opts_.enable_external_correlation;
    input.options.enable_synchronization = opts_.enable_synchronization;
    input.options.enable_cuda_graphs_tracking = opts_.enable_cuda_graphs_tracking;
    input.counters.kernel_rows =
        kernel_activity_emitted_.load(std::memory_order_relaxed);
    input.counters.memory_rows =
        memory_activity_emitted_.load(std::memory_order_relaxed);
    input.counters.mem_transfer_rows =
        mem_transfer_activity_emitted_.load(std::memory_order_relaxed);
    input.counters.sync_rows =
        sync_activity_emitted_.load(std::memory_order_relaxed);
    input.counters.nvtx_rows =
        nvtx_marker_emitted_.load(std::memory_order_relaxed);
    input.counters.graph_rows =
        graph_activity_emitted_.load(std::memory_order_relaxed);
    input.counters.external_rows =
        external_correlation_seen_.load(std::memory_order_relaxed);
    input.counters.source_rows =
        source_locator_seen_.load(std::memory_order_relaxed);
    input.counters.function_rows =
        function_record_seen_.load(std::memory_order_relaxed);
    input.counters.launch_count =
        kernel_launch_callback_count_.load(std::memory_order_acquire);
    input.pc_insufficient_privileges =
        engine_ && engine_->hasInsufficientPrivileges();
    input.pc_stall_reasons_unavailable =
        engine_ && engine_->stallReasonsUnavailable();
    input.pc_no_cuda_context =
        engine_start_pending_.load(std::memory_order_acquire);

    // Surface "armed but produced nothing" to the console (stderr). The
    // capability matrix below only reaches the dashboard, so a `gpufl trace`
    // run otherwise gives no local hint that a too-short workload starved the
    // sampler. Point at the remedies.
    for (const std::string& warning : BuildCaptureCapabilityWarnings(input)) {
        std::fprintf(stderr, "%s\n", warning.c_str());
    }

    const CaptureCapabilitiesEvent evt = BuildCaptureCapabilitiesEvent(input);
    rt->logger->write(model::CaptureCapabilitiesModel(evt));
}

}  // namespace gpufl

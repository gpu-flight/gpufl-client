#include "gpufl/backends/nvidia/cupti_backend.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <string>

#include "gpufl/backends/nvidia/capture_capability_resolver.hpp"
#include "gpufl/backends/nvidia/cupti_engine_selection.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/env_vars.hpp"
#include "gpufl/core/logger/logger.hpp"
#include "gpufl/core/model/lifecycle_model.hpp"
#include "gpufl/core/runtime.hpp"

namespace gpufl {

void CuptiBackend::EmitCaptureCapabilities_() const {
    const Runtime* rt = runtime();
    const auto segment = rt ? rt->acquireSegmentContext() : nullptr;
    if (!segment || !segment->logger) return;
    std::lock_guard capability_lock(capture_capabilities_mu_);
    if  (capture_capabilities_session_id_ == segment->session_id) return;

    const auto delta = [](const uint64_t value, const uint64_t baseline) {
        return value >= baseline ? value - baseline : value;
    };
    const uint64_t kernel_rows =
        kernel_activity_emitted_.load(std::memory_order_relaxed);
    const uint64_t memory_rows =
        memory_activity_emitted_.load(std::memory_order_relaxed);
    const uint64_t transfer_rows =
        mem_transfer_activity_emitted_.load(std::memory_order_relaxed);
    const uint64_t sync_rows =
        sync_activity_emitted_.load(std::memory_order_relaxed);
    const uint64_t nvtx_rows =
        nvtx_marker_emitted_.load(std::memory_order_relaxed);
    const uint64_t graph_rows =
        graph_activity_emitted_.load(std::memory_order_relaxed);
    const uint64_t graph_rows_dropped =
        graph_activity_dropped_.load(std::memory_order_relaxed);
    const uint64_t external_rows =
        external_correlation_seen_.load(std::memory_order_relaxed);
    const uint64_t source_rows =
        source_locator_seen_.load(std::memory_order_relaxed);
    const uint64_t function_rows =
        function_record_seen_.load(std::memory_order_relaxed);
    const uint64_t launch_count =
        kernel_launch_callback_count_.load(std::memory_order_acquire);

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
    input.session_id = segment->session_id;
    input.ts_ns = detail::GetTimestampNs();
    input.requested_engine = opts_.profiling_engine;
    input.combo_active = comboActive();
    input.requests = requests;
    input.engine_state = engineState;
    input.kernel_activity = kernelActivity;
    if (const char* value = std::getenv(env::kExpectNoKernelEvents)) {
        input.expect_no_kernel_events = std::strcmp(value, "1") == 0;
    }
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
        delta(kernel_rows, capability_kernel_rows_baseline_);
    input.counters.memory_rows =
        delta(memory_rows, capability_memory_rows_baseline_);
    input.counters.mem_transfer_rows =
        delta(transfer_rows, capability_mem_transfer_rows_baseline_);
    input.counters.sync_rows =
        delta(sync_rows, capability_sync_rows_baseline_);
    input.counters.nvtx_rows =
        delta(nvtx_rows, capability_nvtx_rows_baseline_);
    input.counters.graph_rows =
        delta(graph_rows, capability_graph_rows_baseline_);
    input.counters.graph_rows_dropped =
        delta(graph_rows_dropped, capability_graph_rows_dropped_baseline_);
    input.counters.external_rows =
        delta(external_rows, capability_external_rows_baseline_);
    input.counters.source_rows =
        delta(source_rows, capability_source_rows_baseline_);
    input.counters.function_rows =
        delta(function_rows, capability_function_rows_baseline_);
    input.counters.launch_count =
        delta(launch_count, capability_launch_count_baseline_);
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
        GFL_LOG_WARN(warning);
    }

    CaptureCapabilitiesEvent evt = BuildCaptureCapabilitiesEvent(input);

    // Scope attribution gave up on some samples. Reported through the
    // capability matrix rather than the local log alone: a log line is invisible
    // to whoever reads the session later, and the samples still upload - they
    // just carry no scope. Without this the dashboard presents partial
    // attribution as though it were complete.
    const uint64_t truncated_total = Monitor::ScopeAttributionTruncated();
    const uint64_t pm_rows_total = Monitor::PmSampleRowsSeen();
    const uint64_t truncated =
        delta(truncated_total, capability_scope_truncated_baseline_);
    const uint64_t pm_rows =
        delta(pm_rows_total, capability_pm_rows_baseline_);
    if (truncated > 0 && pm_rows > 0) {
        CaptureCapability cap;
        cap.feature = "scope_attribution";
        cap.requested = true;
        cap.status = "partial";
        cap.reason_code = "scope_attribution_truncated";
        cap.message = "Evicted " + std::to_string(truncated) +
                      " completed scope records after the retention cap was reached; "
                      "PM scope attribution may be incomplete. Raise pm_sampling_max_samples "
                      "or lengthen the sampling interval so decodes keep up.";
        evt.capabilities.push_back(std::move(cap));
    }

    segment->logger->write(model::CaptureCapabilitiesModel(evt));
    capability_kernel_rows_baseline_ = kernel_rows;
    capability_memory_rows_baseline_ = memory_rows;
    capability_mem_transfer_rows_baseline_ = transfer_rows;
    capability_sync_rows_baseline_ = sync_rows;
    capability_nvtx_rows_baseline_ = nvtx_rows;
    capability_graph_rows_baseline_ = graph_rows;
    capability_graph_rows_dropped_baseline_ = graph_rows_dropped;
    capability_external_rows_baseline_ = external_rows;
    capability_source_rows_baseline_ = source_rows;
    capability_function_rows_baseline_ = function_rows;
    capability_launch_count_baseline_ = launch_count;
    capability_scope_truncated_baseline_ = truncated_total;
    capability_pm_rows_baseline_ = pm_rows_total;
    capture_capabilities_session_id_ = segment->session_id;
}

}  // namespace gpufl

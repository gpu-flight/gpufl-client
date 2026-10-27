#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "gpufl/backends/nvidia/capture_capability_resolver.hpp"

namespace {

const gpufl::CaptureCapability* FindCapability(
    const gpufl::CaptureCapabilitiesEvent& evt,
    const std::string& feature) {
    for (const auto& cap : evt.capabilities) {
        if (cap.feature == feature) return &cap;
    }
    return nullptr;
}

gpufl::CaptureCapabilityInput BaseInput(gpufl::ProfilingEngine engine) {
    gpufl::CaptureCapabilityInput input;
    input.session_id = "session-1";
    input.ts_ns = 1234;
    input.requested_engine = engine;
    input.requests = gpufl::BuildEngineRequestSet(engine, {});
    return input;
}

}  // namespace

TEST(CaptureCapabilityResolver, TraceWithRowsReportsCollectedTimeline) {
    auto input = BaseInput(gpufl::ProfilingEngine::Trace);
    input.kernel_activity = true;
    input.counters.kernel_rows = 1;
    input.counters.mem_transfer_rows = 1;
    input.counters.sync_rows = 1;
    input.counters.nvtx_rows = 1;
    input.counters.graph_rows = 1;
    input.counters.memory_rows = 1;
    input.counters.external_rows = 1;
    input.options.enable_cuda_graphs_tracking = true;

    const auto evt = gpufl::BuildCaptureCapabilitiesEvent(input);

    EXPECT_EQ(evt.session_id, "session-1");
    EXPECT_EQ(evt.ts_ns, 1234);
    EXPECT_EQ(evt.requested_engine, "nvidia.trace");
    EXPECT_EQ(evt.selected_engine, "nvidia.trace");
    EXPECT_EQ(evt.capabilities.size(), 17u);

    const auto* kernel = FindCapability(evt, "kernel_events");
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->requested);
    EXPECT_EQ(kernel->status, "collected");
    EXPECT_EQ(kernel->mode, "cupti_activity");

    const auto* memcpy = FindCapability(evt, "memcpy_activity");
    ASSERT_NE(memcpy, nullptr);
    EXPECT_EQ(memcpy->status, "collected");

    const auto* graph = FindCapability(evt, "graph_activity");
    ASSERT_NE(graph, nullptr);
    EXPECT_TRUE(graph->requested);
    EXPECT_EQ(graph->status, "collected");

    const auto* memory = FindCapability(evt, "memory_activity");
    ASSERT_NE(memory, nullptr);
    EXPECT_TRUE(memory->requested);
    EXPECT_EQ(memory->status, "collected");
}

TEST(CaptureCapabilityResolver, HostOnlyTraceReportsExpectedEmptyKernelTimeline) {
    auto input = BaseInput(gpufl::ProfilingEngine::Trace);
    input.kernel_activity = true;
    input.expect_no_kernel_events = true;
    input.counters.nvtx_rows = 1;

    const auto evt = gpufl::BuildCaptureCapabilitiesEvent(input);

    const auto* kernel = FindCapability(evt, "kernel_events");
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->requested);
    EXPECT_EQ(kernel->status, "expected_no_data");
    EXPECT_EQ(kernel->reason_code, "host_only_trace");

    const auto* markers = FindCapability(evt, "nvtx_markers");
    ASSERT_NE(markers, nullptr);
    EXPECT_TRUE(markers->requested);
    EXPECT_EQ(markers->status, "collected");
}

// Demo session 85d889a6: 273 NVTX rows under `gpufl trace`, reported as
// enabled_no_data because only CUPTI-delivered markers were counted.
TEST(CaptureCapabilityResolver, InjectedNvtxRowsReportCollected) {
    auto input = BaseInput(gpufl::ProfilingEngine::Trace);
    input.kernel_activity = true;
    input.counters.injected_nvtx_rows = 273;

    const auto evt = gpufl::BuildCaptureCapabilitiesEvent(input);

    const auto* markers = FindCapability(evt, "nvtx_markers");
    ASSERT_NE(markers, nullptr);
    EXPECT_TRUE(markers->requested);
    EXPECT_EQ(markers->status, "collected");
    EXPECT_EQ(markers->mode, "nvtx_injection");
    EXPECT_EQ(markers->reason_code, "");
}

TEST(CaptureCapabilityResolver, StackTracesReportTheEffectiveSettingAndItsSource) {
    auto input = BaseInput(gpufl::ProfilingEngine::Trace);
    input.kernel_activity = true;
    input.counters.launch_count = 4;
    input.options.enable_stack_trace = false;
    input.options.stack_trace_source = gpufl::SettingSource::TraceFlag;

    auto evt = gpufl::BuildCaptureCapabilitiesEvent(input);
    const auto* stacks = FindCapability(evt, "stack_traces");
    ASSERT_NE(stacks, nullptr);
    EXPECT_FALSE(stacks->requested);
    EXPECT_EQ(stacks->status, "not_requested");
    EXPECT_EQ(stacks->mode, "trace_flag");

    input.options.enable_stack_trace = true;
    input.options.stack_trace_source = gpufl::SettingSource::Preset;
    evt = gpufl::BuildCaptureCapabilitiesEvent(input);
    stacks = FindCapability(evt, "stack_traces");
    ASSERT_NE(stacks, nullptr);
    EXPECT_TRUE(stacks->requested);
    EXPECT_EQ(stacks->status, "collected");
    EXPECT_EQ(stacks->mode, "preset_default");
}

// The H100 3 s windows: decodes failed with CUPTI_ERROR_UNKNOWN while other
// decodes kept rows, and the session still read `collected`.
TEST(CaptureCapabilityResolver, PmDecodeErrorsAreNotReportedAsCollected) {
    auto input = BaseInput(gpufl::ProfilingEngine::PmSampling);
    input.engine_state.pm.observe(/*armed=*/true, /*produced=*/true);
    input.counters.pm_rows = 980;
    input.counters.pm_decode_errors = 2;
    input.counters.pm_overflows = 1;

    auto evt = gpufl::BuildCaptureCapabilitiesEvent(input);
    const auto* pm = FindCapability(evt, "pm_sampling");
    ASSERT_NE(pm, nullptr);
    EXPECT_EQ(pm->status, "partial");
    EXPECT_EQ(pm->reason_code, "decode_errors");
    EXPECT_EQ(pm->message, "2 PM decode errors, 1 overflow; 980 samples kept.");

    // Every decode failed: nothing was kept.
    input.counters.pm_rows = 0;
    evt = gpufl::BuildCaptureCapabilitiesEvent(input);
    pm = FindCapability(evt, "pm_sampling");
    ASSERT_NE(pm, nullptr);
    EXPECT_EQ(pm->status, "enabled_no_data");
    EXPECT_EQ(pm->reason_code, "decode_errors");
}

TEST(CaptureCapabilityResolver,
     GraphTimingReportsPartialWhenInvalidRecordsWereDropped) {
    auto input = BaseInput(gpufl::ProfilingEngine::Trace);
    input.kernel_activity = true;
    input.options.enable_cuda_graphs_tracking = true;
    input.counters.graph_rows = 3;
    input.counters.graph_rows_dropped = 2;

    const auto evt = gpufl::BuildCaptureCapabilitiesEvent(input);

    const auto* graph = FindCapability(evt, "graph_activity");
    ASSERT_NE(graph, nullptr);
    EXPECT_TRUE(graph->requested);
    EXPECT_EQ(graph->status, "partial");
    EXPECT_EQ(graph->reason_code, "invalid_timestamps_dropped");
    EXPECT_NE(graph->message.find("2 invalid-timestamp record(s)"),
              std::string::npos);
}

TEST(CaptureCapabilityResolver,
     GraphTimingWithOnlyDroppedRecordsIsNotReportedAsNoActivity) {
    auto input = BaseInput(gpufl::ProfilingEngine::Trace);
    input.kernel_activity = true;
    input.options.enable_cuda_graphs_tracking = true;
    input.counters.graph_rows_dropped = 4;

    const auto evt = gpufl::BuildCaptureCapabilitiesEvent(input);

    const auto* graph = FindCapability(evt, "graph_activity");
    ASSERT_NE(graph, nullptr);
    EXPECT_EQ(graph->status, "partial");
    EXPECT_EQ(graph->reason_code, "invalid_timestamps_dropped");
}

// The L4 failure (2026-10-16): every kernel and memcpy record arrived with
// start == end == 0. That is not "no record arrived".
TEST(CaptureCapabilityResolver, AllZeroTimestampsReportTimestampsUnavailable) {
    auto input = BaseInput(gpufl::ProfilingEngine::Trace);
    input.kernel_activity = true;
    input.counters.launch_count = 4096;
    input.counters.kernel_records_eligible = 4096;
    input.counters.kernel_records_zero_ts = 4096;
    input.counters.mem_transfer_records_eligible = 2;
    input.counters.mem_transfer_records_zero_ts = 2;

    const auto evt = gpufl::BuildCaptureCapabilitiesEvent(input);

    const auto* kernel = FindCapability(evt, "kernel_events");
    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(kernel->status, "enabled_no_data");
    EXPECT_EQ(kernel->reason_code, "kernel_timestamps_unavailable");
    EXPECT_NE(kernel->message.find("eligible=4096 zero_timestamps=4096"),
              std::string::npos);
    const auto* memcpy = FindCapability(evt, "memcpy_activity");
    ASSERT_NE(memcpy, nullptr);
    EXPECT_EQ(memcpy->status, "enabled_no_data");
    EXPECT_EQ(memcpy->reason_code, "memcpy_timestamps_unavailable");
}

TEST(CaptureCapabilityResolver, SomeRejectedTimestampsReportPartialTimeline) {
    auto input = BaseInput(gpufl::ProfilingEngine::Trace);
    input.kernel_activity = true;
    input.counters.kernel_rows = 10;
    input.counters.kernel_records_eligible = 12;
    input.counters.kernel_records_zero_ts = 1;
    input.counters.kernel_records_invalid_ts = 1;

    const auto evt = gpufl::BuildCaptureCapabilitiesEvent(input);

    const auto* kernel = FindCapability(evt, "kernel_events");
    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(kernel->status, "collected");
    EXPECT_EQ(kernel->reason_code, "kernel_timestamps_partial");
    EXPECT_EQ(FindCapability(evt, "memcpy_activity")->reason_code,
              "enabled_but_no_records");
}

TEST(CaptureCapabilityResolver, RecordsThatNeverArrivedKeepNoRecordsReason) {
    auto input = BaseInput(gpufl::ProfilingEngine::Trace);
    input.kernel_activity = true;
    input.counters.launch_count = 3;

    const auto evt = gpufl::BuildCaptureCapabilitiesEvent(input);

    const auto* kernel = FindCapability(evt, "kernel_events");
    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(kernel->status, "enabled_no_data");
    EXPECT_EQ(kernel->reason_code, "enabled_but_no_records");
}

TEST(CaptureCapabilityResolver, PcSamplingSyntheticMajorityReportsFallback) {
    auto input = BaseInput(gpufl::ProfilingEngine::PcSampling);
    input.counters.kernel_rows = 2;
    input.counters.launch_count = 10;
    input.engine_state.pc.observe(true, false);

    const auto evt = gpufl::BuildCaptureCapabilitiesEvent(input);

    const auto* kernel = FindCapability(evt, "kernel_events");
    ASSERT_NE(kernel, nullptr);
    EXPECT_FALSE(kernel->requested);
    EXPECT_EQ(kernel->status, "fallback");
    EXPECT_EQ(kernel->mode, "launch_callbacks_synthetic");
    EXPECT_EQ(kernel->reason_code,
              "cupti_kernel_activity_conflicts_with_pc_sampling");

    const auto* names = FindCapability(evt, "kernel_names");
    ASSERT_NE(names, nullptr);
    EXPECT_EQ(names->status, "partial");

    const auto warnings = gpufl::BuildCaptureCapabilityWarnings(input);
    ASSERT_EQ(warnings.size(), 1u);
    EXPECT_NE(warnings[0].find("PC sampling collected 0 stall samples"),
              std::string::npos);
}

TEST(CaptureCapabilityResolver, DeepSassSelectionSkipsPcSampling) {
    auto input = BaseInput(gpufl::ProfilingEngine::Deep);
    input.engine_state.sass.observe(true, true);

    const auto evt = gpufl::BuildCaptureCapabilitiesEvent(input);

    EXPECT_EQ(evt.requested_engine, "nvidia.pc_sampling_with_sass");
    EXPECT_EQ(evt.selected_engine, "nvidia.sass_metrics");

    const auto* pc = FindCapability(evt, "pc_sampling");
    ASSERT_NE(pc, nullptr);
    EXPECT_TRUE(pc->requested);
    EXPECT_EQ(pc->status, "skipped");
    EXPECT_EQ(pc->reason_code, "mutually_exclusive_with_sass_metrics");

    const auto* source = FindCapability(evt, "source_correlation");
    ASSERT_NE(source, nullptr);
    EXPECT_EQ(source->status, "skipped");
    EXPECT_EQ(source->reason_code, "sass_metrics_have_no_source_lines");
}

TEST(CaptureCapabilityResolver, PcSamplingInactiveReasonKeepsNoContext) {
    auto input = BaseInput(gpufl::ProfilingEngine::PcSampling);
    input.pc_no_cuda_context = true;

    const auto evt = gpufl::BuildCaptureCapabilitiesEvent(input);

    const auto* pc = FindCapability(evt, "pc_sampling");
    ASSERT_NE(pc, nullptr);
    EXPECT_TRUE(pc->requested);
    EXPECT_EQ(pc->status, "skipped");
    EXPECT_EQ(pc->reason_code, "no_cuda_context");
    EXPECT_NE(pc->message.find("did not create a CUDA context"),
              std::string::npos);
}

TEST(CaptureCapabilityResolver, SassSafeModeCanSkipMemoryActivity) {
    auto input = BaseInput(gpufl::ProfilingEngine::SassMetrics);
    input.allow_sass_memory2_activity = false;

    const auto evt = gpufl::BuildCaptureCapabilitiesEvent(input);

    const auto* memory = FindCapability(evt, "memory_activity");
    ASSERT_NE(memory, nullptr);
    EXPECT_TRUE(memory->requested);
    EXPECT_EQ(memory->status, "skipped");
    EXPECT_EQ(memory->mode, "disabled");
    EXPECT_EQ(memory->reason_code,
              "sass_safe_mode_memory_activity_disabled");
}

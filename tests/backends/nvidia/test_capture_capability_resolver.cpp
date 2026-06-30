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
    EXPECT_EQ(evt.capabilities.size(), 16u);

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

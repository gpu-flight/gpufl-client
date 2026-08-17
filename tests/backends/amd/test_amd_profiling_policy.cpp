#include <gtest/gtest.h>

#include <string>

#include "gpufl/backends/amd/amd_capture_capabilities.hpp"
#include "gpufl/backends/amd/amd_profiling_policy.hpp"

namespace {

const gpufl::CaptureCapability* FindCapability(
    const gpufl::CaptureCapabilitiesEvent& event,
    const std::string& feature) {
    for (const auto& capability : event.capabilities) {
        if (capability.feature == feature) return &capability;
    }
    return nullptr;
}

}  // namespace

TEST(AmdProfilingPolicy, RequestIntentNeverInventsAmdNativeNames) {
    EXPECT_STREQ(gpufl::amd::AmdRequestIntentWireName(
                     gpufl::ProfilingEngine::PcSampling),
                 "pc_sampling");
    EXPECT_STREQ(gpufl::amd::AmdRequestIntentWireName(
                     gpufl::ProfilingEngine::SassMetrics),
                 "sass_metrics");
    EXPECT_STREQ(gpufl::amd::AmdRequestIntentWireName(
                     gpufl::ProfilingEngine::PmSampling),
                 "pm_sampling");
    EXPECT_STREQ(gpufl::amd::AmdRequestIntentWireName(
                     gpufl::ProfilingEngine::RangeProfiler),
                 "range_profiler");

    gpufl::amd::AmdProfilingSupport support;
    support.device_counting = true;
    const auto plan = gpufl::amd::ResolveAmdProfilingPlan(
        gpufl::ProfilingEngine::PmSampling, support);

    EXPECT_EQ(plan.selected_path,
              gpufl::amd::AmdProfilingPath::DeviceCounting);
    EXPECT_STREQ(gpufl::amd::AmdSelectedPathWireName(plan.selected_path),
                 "amd.device_counting");
}

TEST(AmdProfilingPolicy, TraceSelectsBufferTracingService) {
    const auto plan = gpufl::amd::ResolveAmdProfilingPlan(
        gpufl::ProfilingEngine::Trace, {});

    EXPECT_EQ(plan.selected_path, gpufl::amd::AmdProfilingPath::BufferTracing);
    EXPECT_FALSE(plan.degraded);
    EXPECT_TRUE(plan.reason_code.empty());
    EXPECT_STREQ(gpufl::amd::AmdRequestIntentWireName(plan.requested_engine),
                 "trace");
    EXPECT_STREQ(gpufl::amd::AmdSelectedPathWireName(plan.selected_path),
                 "amd.buffer_tracing");
}

TEST(AmdProfilingPolicy, PcSamplingFallsBackTruthfullyWhenUnavailable) {
    const auto plan = gpufl::amd::ResolveAmdProfilingPlan(
        gpufl::ProfilingEngine::PcSampling, {});

    EXPECT_EQ(plan.selected_path, gpufl::amd::AmdProfilingPath::BufferTracing);
    EXPECT_TRUE(plan.degraded);
    EXPECT_EQ(plan.reason_code, "amd_pc_sampling_unavailable_baseline_tracing_retained");
}

TEST(AmdProfilingPolicy, SassIntentFallsBackToDispatchCounting) {
    gpufl::amd::AmdProfilingSupport support;
    support.dispatch_counting = true;

    const auto plan = gpufl::amd::ResolveAmdProfilingPlan(
        gpufl::ProfilingEngine::SassMetrics, support);

    EXPECT_EQ(plan.selected_path,
              gpufl::amd::AmdProfilingPath::DispatchCounting);
    EXPECT_TRUE(plan.degraded);
    EXPECT_EQ(plan.reason_code, "requested_metric_model_unavailable_dispatch_counting_selected");
    EXPECT_STREQ(gpufl::amd::AmdSelectedPathWireName(plan.selected_path),
                 "amd.dispatch_counting");
}

TEST(AmdProfilingPolicy, DeepReportsDispatchOnlyPartialImplementation) {
    gpufl::amd::AmdProfilingSupport support;
    support.dispatch_counting = true;

    const auto plan = gpufl::amd::ResolveAmdProfilingPlan(
        gpufl::ProfilingEngine::Deep, support);

    EXPECT_EQ(plan.selected_path,
              gpufl::amd::AmdProfilingPath::DispatchCounting);
    EXPECT_TRUE(plan.degraded);
    EXPECT_EQ(plan.reason_code, "deep_services_unavailable_dispatch_counting_selected");
}

TEST(AmdCaptureCapabilities, PcFallbackNamesRequestAndSelectionSeparately) {
    gpufl::amd::AmdCaptureCapabilityInput input;
    input.session_id = "amd-session";
    input.ts_ns = 123;
    input.plan = gpufl::amd::ResolveAmdProfilingPlan(
        gpufl::ProfilingEngine::PcSampling, {});
    input.trace_configured = true;
    input.kernel_rows = 2;

    const auto event = gpufl::amd::BuildAmdCaptureCapabilitiesEvent(input);

    EXPECT_EQ(event.requested_engine, "pc_sampling");
    EXPECT_EQ(event.selected_engine, "amd.buffer_tracing");

    const auto* selection = FindCapability(event, "engine_selection");
    ASSERT_NE(selection, nullptr);
    EXPECT_EQ(selection->status, "fallback");
    EXPECT_EQ(selection->reason_code, "amd_pc_sampling_unavailable_baseline_tracing_retained");

    const auto* pc = FindCapability(event, "pc_sampling");
    ASSERT_NE(pc, nullptr);
    EXPECT_TRUE(pc->requested);
    EXPECT_EQ(pc->status, "skipped");
}

TEST(AmdCaptureCapabilities, DispatchSamplesAndDroppedTraceAreVisible) {
    gpufl::amd::AmdProfilingSupport support;
    support.dispatch_counting = true;

    gpufl::amd::AmdCaptureCapabilityInput input;
    input.session_id = "amd-session";
    input.plan = gpufl::amd::ResolveAmdProfilingPlan(
        gpufl::ProfilingEngine::RangeProfiler, support);
    input.trace_configured = true;
    input.profiling_sample_rows = 4;
    input.dropped_trace_records = 3;

    const auto event = gpufl::amd::BuildAmdCaptureCapabilitiesEvent(input);

    EXPECT_EQ(event.selected_engine, "amd.dispatch_counting");

    const auto* counters = FindCapability(event, "dispatch_counting");
    ASSERT_NE(counters, nullptr);
    EXPECT_TRUE(counters->requested);
    EXPECT_EQ(counters->status, "collected");

    const auto* delivery = FindCapability(event, "trace_buffer_delivery");
    ASSERT_NE(delivery, nullptr);
    EXPECT_EQ(delivery->status, "partial");
    EXPECT_EQ(delivery->reason_code, "rocprofiler_records_dropped");
    EXPECT_NE(delivery->message.find("3 dropped trace record(s)"),
              std::string::npos);
}

#include <gtest/gtest.h>

#include <string>

#include "gpufl/backends/amd/amd_capture_capabilities.hpp"
#include "gpufl/backends/amd/amd_dispatch_collection_gate.hpp"
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

TEST(AmdDispatchCollectionGate, AlwaysModeFollowsSessionLifetime) {
    gpufl::amd::AmdDispatchCollectionGate gate;
    gate.configure(gpufl::DeepArmMode::Always);

    EXPECT_FALSE(gate.armed());
    gate.start();
    EXPECT_TRUE(gate.armed());
    EXPECT_TRUE(gate.collectDispatch(false));
    gate.closeWindow();
    EXPECT_TRUE(gate.armed());
    gate.stop();
    EXPECT_FALSE(gate.armed());
}

TEST(AmdDispatchCollectionGate, WindowOnlyArmsExactlyInsideWindow) {
    gpufl::amd::AmdDispatchCollectionGate gate;
    gate.configure(gpufl::DeepArmMode::WindowOnly);

    gate.start();
    EXPECT_FALSE(gate.armed());
    gate.openWindow();
    EXPECT_TRUE(gate.armed());
    EXPECT_TRUE(gate.collectDispatch(true));
    EXPECT_FALSE(gate.collectDispatch(false));
    gate.closeWindow();
    EXPECT_FALSE(gate.armed());
    // A callback that claimed the final slot before collector-thread disarm
    // still owns that dispatch.
    EXPECT_TRUE(gate.collectDispatch(true));
    EXPECT_FALSE(gate.collectDispatch(false));
}

TEST(AmdDispatchCollectionGate, StopClearsWindowBeforeRestart) {
    gpufl::amd::AmdDispatchCollectionGate gate;
    gate.configure(gpufl::DeepArmMode::WindowOnly);

    gate.openWindow();
    gate.start();
    ASSERT_TRUE(gate.armed());
    gate.stop();
    gate.start();
    EXPECT_FALSE(gate.armed());
}

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
    input.unattributed_trace_records = 2;

    const auto event = gpufl::amd::BuildAmdCaptureCapabilitiesEvent(input);

    EXPECT_EQ(event.selected_engine, "amd.dispatch_counting");

    const auto* counters = FindCapability(event, "dispatch_counting");
    ASSERT_NE(counters, nullptr);
    EXPECT_TRUE(counters->requested);
    EXPECT_EQ(counters->status, "collected");

    const auto* attribution = FindCapability(event, "device_attribution");
    ASSERT_NE(attribution, nullptr);
    EXPECT_EQ(attribution->status, "partial");
    EXPECT_EQ(attribution->reason_code, "rocprofiler_agent_unmapped");

    const auto* delivery = FindCapability(event, "trace_buffer_delivery");
    ASSERT_NE(delivery, nullptr);
    EXPECT_EQ(delivery->status, "partial");
    EXPECT_EQ(delivery->reason_code, "rocprofiler_records_dropped");
    EXPECT_NE(delivery->message.find("3 dropped trace record(s)"),
              std::string::npos);
}

TEST(AmdCaptureCapabilities, LifecycleDeliveryAndCorrelationFailuresAreVisible) {
    gpufl::amd::AmdCaptureCapabilityInput input;
    input.session_id = "amd-session";
    input.plan = gpufl::amd::ResolveAmdProfilingPlan(
        gpufl::ProfilingEngine::Trace, {});
    input.trace_configured = true;
    input.dropped_client_records = 5;
    input.trace_buffer_flush_failures = 2;
    input.scope_correlation_failures = 3;

    const auto event = gpufl::amd::BuildAmdCaptureCapabilitiesEvent(input);

    const auto* delivery = FindCapability(event, "trace_buffer_delivery");
    ASSERT_NE(delivery, nullptr);
    EXPECT_EQ(delivery->status, "partial");
    EXPECT_EQ(delivery->reason_code, "rocprofiler_buffer_flush_failed");
    EXPECT_NE(delivery->message.find("failed 2 time(s)"), std::string::npos);
    EXPECT_NE(delivery->message.find("wrong segment"), std::string::npos);

    const auto* correlation = FindCapability(event, "scope_correlation");
    ASSERT_NE(correlation, nullptr);
    EXPECT_TRUE(correlation->requested);
    EXPECT_EQ(correlation->status, "partial");
    EXPECT_EQ(correlation->mode, "rocprofiler_external_correlation");
    EXPECT_EQ(correlation->reason_code,
              "rocprofiler_scope_correlation_failed");
    EXPECT_NE(correlation->message.find("failed 3 time(s)"),
              std::string::npos);
}

TEST(AmdCaptureCapabilities, ClientQueueDropsDegradeEndToEndDelivery) {
    gpufl::amd::AmdCaptureCapabilityInput input;
    input.session_id = "amd-session";
    input.plan = gpufl::amd::ResolveAmdProfilingPlan(
        gpufl::ProfilingEngine::Trace, {});
    input.trace_configured = true;
    input.dropped_client_records = 5;

    const auto event = gpufl::amd::BuildAmdCaptureCapabilitiesEvent(input);

    const auto* delivery = FindCapability(event, "trace_buffer_delivery");
    ASSERT_NE(delivery, nullptr);
    EXPECT_EQ(delivery->status, "partial");
    EXPECT_EQ(delivery->reason_code, "gpufl_activity_queue_full");
    EXPECT_NE(delivery->message.find("dropped 5 trace record(s)"),
              std::string::npos);

    const auto* correlation = FindCapability(event, "scope_correlation");
    ASSERT_NE(correlation, nullptr);
    EXPECT_EQ(correlation->status, "enabled");
}

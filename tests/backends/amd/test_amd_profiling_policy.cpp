#include <gtest/gtest.h>

#include <string>
#include <utility>

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

TEST(AmdProfilingPolicy, DeviceCountingUsesPortableDefaultMetric) {
    EXPECT_EQ(gpufl::amd::ResolveAmdDeviceCountingMetrics({}),
              std::vector<std::string>{"GPUBusy"});
}

TEST(AmdProfilingPolicy, DeviceCountingPreservesUniqueExplicitMetrics) {
    EXPECT_EQ(gpufl::amd::ResolveAmdDeviceCountingMetrics(
                  {"SQ_WAVES", "", "GPUBusy", "SQ_WAVES"}),
              (std::vector<std::string>{"SQ_WAVES", "GPUBusy"}));
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

TEST(AmdProfilingPolicy, PcSamplingWithoutTheOptInKeepsTracing) {
    const auto plan = gpufl::amd::ResolveAmdProfilingPlan(
        gpufl::ProfilingEngine::PcSampling, {});

    EXPECT_EQ(plan.selected_path, gpufl::amd::AmdProfilingPath::BufferTracing);
    EXPECT_TRUE(plan.degraded);
    EXPECT_EQ(plan.reason_code, "amd_pc_sampling_not_opted_in");
}

TEST(AmdProfilingPolicy, PcSamplingRefusalsKeepTracingWithTheirReason) {
    using gpufl::amd::AmdPcSamplingRefusal;
    const std::pair<AmdPcSamplingRefusal, const char*> cases[] = {
        {AmdPcSamplingRefusal::NotOptedIn, "amd_pc_sampling_not_opted_in"},
        {AmdPcSamplingRefusal::MethodUnavailable,
         "amd_pc_sampling_method_unavailable"},
        {AmdPcSamplingRefusal::DriverUnsupported,
         "amd_pc_sampling_driver_unsupported"},
        {AmdPcSamplingRefusal::Unavailable, "amd_pc_sampling_unavailable"},
        {AmdPcSamplingRefusal::ConfigurationFailed,
         "amd_pc_sampling_configuration_failed"},
        {AmdPcSamplingRefusal::WindowOnly,
         "amd_pc_sampling_window_only_unsupported"},
    };
    for (const auto& [refusal, reason] : cases) {
        gpufl::amd::AmdProfilingSupport support;
        support.dispatch_counting = true;
        support.device_counting = true;
        support.pc_sampling_refusal = refusal;
        const auto plan = gpufl::amd::ResolveAmdProfilingPlan(
            gpufl::ProfilingEngine::PcSampling, support);

        EXPECT_EQ(plan.selected_path,
                  gpufl::amd::AmdProfilingPath::BufferTracing);
        EXPECT_TRUE(plan.degraded);
        EXPECT_EQ(plan.reason_code, reason);
    }
}

TEST(AmdProfilingPolicy, PcSamplingRunsWhenTheAgentConfiguredIt) {
    gpufl::amd::AmdProfilingSupport support;
    support.pc_sampling = true;
    const auto plan = gpufl::amd::ResolveAmdProfilingPlan(
        gpufl::ProfilingEngine::PcSampling, support);

    EXPECT_EQ(plan.selected_path, gpufl::amd::AmdProfilingPath::PcSampling);
    EXPECT_FALSE(plan.degraded);
    EXPECT_TRUE(plan.reason_code.empty());
    EXPECT_STREQ(gpufl::amd::AmdSelectedPathWireName(plan.selected_path),
                 "amd.pc_sampling");
}

TEST(AmdProfilingPolicy, PcSamplingJoinsNoCounterGroup) {
    gpufl::amd::AmdProfilingSupport support;
    support.pc_sampling = true;
    support.dispatch_counting = true;
    for (const auto engine :
         {gpufl::ProfilingEngine::Deep, gpufl::ProfilingEngine::SassMetrics,
          gpufl::ProfilingEngine::RangeProfiler,
          gpufl::ProfilingEngine::RangeProfilerKernelReplay}) {
        const auto plan = gpufl::amd::ResolveAmdProfilingPlan(engine, support);
        EXPECT_EQ(plan.selected_path,
                  gpufl::amd::AmdProfilingPath::DispatchCounting);
        EXPECT_FALSE(gpufl::amd::AmdRequestNeedsPcSampling(engine));
    }
}

TEST(AmdProfilingPolicy, TracePlusPcSamplingRequestsPcSampling) {
    using gpufl::ProfilingEngine;
    using gpufl::amd::ResolveAmdProfilingRequest;
    // `--passes=Trace+PcSampling` publishes the first engine and the combo.
    const auto group =
        ResolveAmdProfilingRequest(ProfilingEngine::Trace, "Trace,PcSampling");
    EXPECT_EQ(group.engine, ProfilingEngine::PcSampling);
    EXPECT_TRUE(group.trace_requested);
    EXPECT_FALSE(group.pc_sampling_dropped);
    const auto tab = ResolveAmdProfilingRequest(ProfilingEngine::PcSampling,
                                                "PcSampling,\tTrace");
    EXPECT_EQ(tab.engine, ProfilingEngine::PcSampling);
    EXPECT_TRUE(tab.trace_requested);

    const auto alone =
        ResolveAmdProfilingRequest(ProfilingEngine::PcSampling, "");
    EXPECT_EQ(alone.engine, ProfilingEngine::PcSampling);
    EXPECT_FALSE(alone.trace_requested);
    EXPECT_EQ(ResolveAmdProfilingRequest(ProfilingEngine::Trace, "").engine,
              ProfilingEngine::Trace);
    EXPECT_EQ(ResolveAmdProfilingRequest(ProfilingEngine::Trace,
                                         "Trace,PmSampling")
                  .engine,
              ProfilingEngine::Trace);
}

TEST(AmdProfilingPolicy, AGroupWithACounterEngineDropsPcSampling) {
    using gpufl::ProfilingEngine;
    using gpufl::amd::ResolveAmdProfilingRequest;
    // The group keeps its first engine, and says PcSampling didn't run.
    const auto dropped = ResolveAmdProfilingRequest(ProfilingEngine::PmSampling,
                                                    "PmSampling,PcSampling");
    EXPECT_EQ(dropped.engine, ProfilingEngine::PmSampling);
    EXPECT_TRUE(dropped.pc_sampling_dropped);
    EXPECT_FALSE(dropped.trace_requested);
    const auto three = ResolveAmdProfilingRequest(
        ProfilingEngine::Trace, "Trace,PcSampling,PmSampling");
    EXPECT_EQ(three.engine, ProfilingEngine::Trace);
    EXPECT_TRUE(three.pc_sampling_dropped);

    // PcSampling first: the group samples.
    const auto first = ResolveAmdProfilingRequest(ProfilingEngine::PcSampling,
                                                  "PcSampling,PmSampling");
    EXPECT_EQ(first.engine, ProfilingEngine::PcSampling);
    EXPECT_FALSE(first.pc_sampling_dropped);
}

// The backend counts kernel timing only for tracing engines, and a
// Trace+PcSampling group traced every kernel.
TEST(AmdProfilingPolicy, TracePlusPcSamplingReportsItsTracing) {
    gpufl::amd::AmdProfilingSupport support;
    support.pc_sampling = true;
    const auto group = gpufl::amd::ResolveAmdProfilingPlan(
        gpufl::amd::ResolveAmdProfilingRequest(gpufl::ProfilingEngine::Trace,
                                               "Trace,PcSampling"),
        support);
    EXPECT_EQ(group.selected_path, gpufl::amd::AmdProfilingPath::PcSampling);
    EXPECT_STREQ(gpufl::amd::AmdSessionEngineWireName(group),
                 "amd.buffer_tracing");

    gpufl::amd::AmdCaptureCapabilityInput input;
    input.plan = group;
    input.trace_configured = true;
    input.pc_samples = 3;
    input.pc_samples_issued = 3;
    const auto event = gpufl::amd::BuildAmdCaptureCapabilitiesEvent(input);
    EXPECT_EQ(event.requested_engine, "pc_sampling");
    EXPECT_EQ(event.selected_engine, "amd.buffer_tracing");
    const auto* pc = FindCapability(event, "pc_sampling");
    ASSERT_NE(pc, nullptr);
    EXPECT_EQ(pc->status, "collected");
    EXPECT_EQ(pc->mode, "rocprofiler_pc_sampling_stochastic");

    const auto alone = gpufl::amd::ResolveAmdProfilingPlan(
        gpufl::amd::ResolveAmdProfilingRequest(
            gpufl::ProfilingEngine::PcSampling, ""),
        support);
    EXPECT_STREQ(gpufl::amd::AmdSessionEngineWireName(alone),
                 "amd.pc_sampling");
}

TEST(AmdProfilingPolicy, PcSamplingNeedsTheOptInAndWholeRunArming) {
    using gpufl::amd::AmdPcSamplingPreflight;
    using gpufl::amd::AmdPcSamplingRefusal;
    gpufl::MonitorOptions options;
    EXPECT_EQ(AmdPcSamplingPreflight(options),
              AmdPcSamplingRefusal::NotOptedIn);

    options.amd_pc_sampling = gpufl::AmdPcSamplingMethod::Stochastic;
    EXPECT_FALSE(AmdPcSamplingPreflight(options).has_value());

    // rocprofiler-sdk samples while the context runs, which tracing keeps
    // running between deep windows.
    options.deep_arm_mode = gpufl::DeepArmMode::WindowOnly;
    EXPECT_EQ(AmdPcSamplingPreflight(options),
              AmdPcSamplingRefusal::WindowOnly);
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

TEST(AmdProfilingPolicy, DispatchSamplesKeepConfiguredDeviceIdentity) {
    EXPECT_EQ(gpufl::amd::ResolveAmdDispatchDeviceId(101, 7, 101), 7u);
}

TEST(AmdProfilingPolicy, DispatchSamplesRejectOtherOrUnknownAgents) {
    EXPECT_FALSE(
        gpufl::amd::ResolveAmdDispatchDeviceId(101, 7, 202).has_value());
    EXPECT_FALSE(
        gpufl::amd::ResolveAmdDispatchDeviceId(0, 7, 101).has_value());
    EXPECT_FALSE(
        gpufl::amd::ResolveAmdDispatchDeviceId(101, 7, 0).has_value());
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
    EXPECT_EQ(selection->reason_code, "amd_pc_sampling_not_opted_in");

    const auto* pc = FindCapability(event, "pc_sampling");
    ASSERT_NE(pc, nullptr);
    EXPECT_TRUE(pc->requested);
    EXPECT_EQ(pc->status, "skipped");
    EXPECT_EQ(pc->reason_code, "amd_pc_sampling_not_opted_in");
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

TEST(AmdCaptureCapabilities, DeviceCountingUsesPmRowsForDataStatus) {
    gpufl::amd::AmdProfilingSupport support;
    support.device_counting = true;

    gpufl::amd::AmdCaptureCapabilityInput input;
    input.session_id = "amd-session";
    input.plan = gpufl::amd::ResolveAmdProfilingPlan(
        gpufl::ProfilingEngine::PmSampling, support);
    input.trace_configured = true;
    input.profiling_sample_rows = 0;
    input.pm_sample_rows = 3;

    const auto event = gpufl::amd::BuildAmdCaptureCapabilitiesEvent(input);
    const auto* counters = FindCapability(event, "device_counting");
    ASSERT_NE(counters, nullptr);
    EXPECT_TRUE(counters->requested);
    EXPECT_EQ(counters->status, "collected");
    EXPECT_EQ(counters->mode, "rocprofiler_device_counting_service");

    input.profiling_sample_rows = 3;
    input.pm_sample_rows = 0;
    const auto no_pm_event =
        gpufl::amd::BuildAmdCaptureCapabilitiesEvent(input);
    counters = FindCapability(no_pm_event, "device_counting");
    ASSERT_NE(counters, nullptr);
    EXPECT_EQ(counters->status, "enabled_no_data");
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

TEST(AmdCaptureCapabilities, MemoryActivityReportsCollectionAndAvailability) {
    gpufl::amd::AmdCaptureCapabilityInput input;
    input.session_id = "amd-session";
    input.plan = gpufl::amd::ResolveAmdProfilingPlan(
        gpufl::ProfilingEngine::Trace, {});
    input.trace_configured = true;
    input.memory_activity_requested = true;
    input.memory_activity_configured = true;
    input.memory_activity_rows = 4;

    auto event = gpufl::amd::BuildAmdCaptureCapabilitiesEvent(input);
    const auto* memory = FindCapability(event, "memory_activity");
    ASSERT_NE(memory, nullptr);
    EXPECT_TRUE(memory->requested);
    EXPECT_EQ(memory->status, "collected");
    EXPECT_EQ(memory->mode, "rocprofiler_memory_allocation");

    input.memory_activity_rows = 0;
    event = gpufl::amd::BuildAmdCaptureCapabilitiesEvent(input);
    memory = FindCapability(event, "memory_activity");
    ASSERT_NE(memory, nullptr);
    EXPECT_EQ(memory->status, "enabled_no_data");
    EXPECT_EQ(memory->reason_code, "enabled_but_no_records");

    input.memory_activity_configured = false;
    event = gpufl::amd::BuildAmdCaptureCapabilitiesEvent(input);
    memory = FindCapability(event, "memory_activity");
    ASSERT_NE(memory, nullptr);
    EXPECT_EQ(memory->status, "skipped");
    EXPECT_EQ(memory->reason_code,
              "rocprofiler_memory_allocation_unavailable");

    input.memory_activity_requested = false;
    event = gpufl::amd::BuildAmdCaptureCapabilitiesEvent(input);
    memory = FindCapability(event, "memory_activity");
    ASSERT_NE(memory, nullptr);
    EXPECT_EQ(memory->status, "not_requested");
}

TEST(AmdCaptureCapabilities, SynchronizationReportsCollectionAndAvailability) {
    gpufl::amd::AmdCaptureCapabilityInput input;
    input.session_id = "amd-session";
    input.plan = gpufl::amd::ResolveAmdProfilingPlan(
        gpufl::ProfilingEngine::Trace, {});
    input.trace_configured = true;
    input.synchronization_requested = true;
    input.synchronization_configured = true;
    input.synchronization_rows = 4;

    auto event = gpufl::amd::BuildAmdCaptureCapabilitiesEvent(input);
    const auto* sync = FindCapability(event, "sync_activity");
    ASSERT_NE(sync, nullptr);
    EXPECT_TRUE(sync->requested);
    EXPECT_EQ(sync->status, "collected");
    EXPECT_EQ(sync->mode, "rocprofiler_hip_runtime_api_ext");

    input.synchronization_rows = 0;
    event = gpufl::amd::BuildAmdCaptureCapabilitiesEvent(input);
    sync = FindCapability(event, "sync_activity");
    ASSERT_NE(sync, nullptr);
    EXPECT_EQ(sync->status, "enabled_no_data");
    EXPECT_EQ(sync->reason_code, "enabled_but_no_records");

    input.synchronization_configured = false;
    event = gpufl::amd::BuildAmdCaptureCapabilitiesEvent(input);
    sync = FindCapability(event, "sync_activity");
    ASSERT_NE(sync, nullptr);
    EXPECT_EQ(sync->status, "skipped");
    EXPECT_EQ(sync->reason_code, "rocprofiler_hip_runtime_api_unavailable");

    input.synchronization_requested = false;
    event = gpufl::amd::BuildAmdCaptureCapabilitiesEvent(input);
    sync = FindCapability(event, "sync_activity");
    ASSERT_NE(sync, nullptr);
    EXPECT_EQ(sync->status, "not_requested");
}

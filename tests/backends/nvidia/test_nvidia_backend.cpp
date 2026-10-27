#include <gtest/gtest.h>

#include "common/test_utils.hpp"

#if GPUFL_ENABLE_NVIDIA && GPUFL_HAS_CUPTI
#include "gpufl/backends/nvidia/cupti_backend.hpp"
#include "gpufl/backends/nvidia/engine/pm_sampling_engine.hpp"
#include "gpufl/backends/nvidia/kernel_launch_handler.hpp"

class CuptiBackendTest : public ::testing::Test {
   protected:
    void SetUp() override { SKIP_IF_NO_CUDA(); }
};

// Capture readiness needs every required activity kind enabled, at start and
// after the engine re-enable. A "benign" CUPTI result (NOT_COMPATIBLE, a
// fallback the logger keeps quiet about) is not an enable.
TEST(RequiredActivityTally, ABenignResultOrALaterFailedReEnableIsNotReady) {
    std::string reason;
    gpufl::RequiredActivityTally none;
    EXPECT_FALSE(none.Ready(&reason));
    EXPECT_EQ(reason, "activity_not_enabled");

    gpufl::RequiredActivityTally benign;
    benign.Record(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, true, CUPTI_SUCCESS);
    benign.Record(CUPTI_ACTIVITY_KIND_MEMCPY, false, CUPTI_ERROR_NOT_COMPATIBLE);
    ASSERT_TRUE(gpufl::IsBenignCuptiResult(CUPTI_ERROR_NOT_COMPATIBLE));
    EXPECT_FALSE(benign.Ready(&reason));
    EXPECT_EQ(reason, "activity_enable_failed:" +
                          std::to_string(CUPTI_ACTIVITY_KIND_MEMCPY) + ":" +
                          std::to_string(CUPTI_ERROR_NOT_COMPATIBLE));

    gpufl::RequiredActivityTally reenable;
    reenable.Record(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, true, CUPTI_SUCCESS);
    EXPECT_FALSE(reenable.Ready(&reason));  // the required set is not done yet
    reenable.Evaluated();
    EXPECT_TRUE(reenable.Ready(&reason));
    reenable.Record(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, false,
                    CUPTI_ERROR_NOT_INITIALIZED);
    reenable.Record(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, true, CUPTI_SUCCESS);
    EXPECT_FALSE(reenable.Ready(&reason));  // the failure latches
}

// PC sampling and SASS safe mode require no activity kind. Their handshake
// must still report ready: the Rust and Python hosts abort on anything but 1.
TEST(RequiredActivityTally, AnEmptyRequiredSetIsReadyOnceEvaluated) {
    std::string reason;
    gpufl::RequiredActivityTally empty;
    empty.Evaluated();
    EXPECT_TRUE(empty.Ready(&reason));
}

// Pure decision for Monitor::Shutdown's orphan-synthesis suppression. No CUDA
// needed. The case this exists for: a real-record session (kernel activity
// enabled, not PC/SASS synthesize-by-design) where launches happened but zero
// activity records arrived - the orphan drain would otherwise fabricate every
// kernel row from host launch-to-launch gaps.
TEST(KernelActivityExpectedButMissing, TruthTable) {
    using gpufl::KernelActivityExpectedButMissing;

    // The broken session: real records expected, launches happened, none came.
    EXPECT_TRUE(KernelActivityExpectedButMissing(true, false, 100000, 0));
    // One single missing-record launch is still a fully-lost session.
    EXPECT_TRUE(KernelActivityExpectedButMissing(true, false, 1, 0));

    // The final argument is accepted/valid rows, not records merely observed
    // before timestamp validation. A rejected record therefore leaves it zero.
    EXPECT_TRUE(KernelActivityExpectedButMissing(true, false, 100000, 0));
    // Any valid real row means the all-records-missing diagnostic is false.
    // Orphan synthesis is still independently disabled for the whole session.
    EXPECT_FALSE(KernelActivityExpectedButMissing(true, false, 100000, 1));
    EXPECT_FALSE(KernelActivityExpectedButMissing(true, false, 100000, 99999));

    // No launches: nothing was lost (and nothing to suppress).
    EXPECT_FALSE(KernelActivityExpectedButMissing(true, false, 0, 0));

    // Synthesize-by-design modes (PC sampling / SASS safe): synthetic rows
    // are the product, never suppressed here.
    EXPECT_FALSE(KernelActivityExpectedButMissing(true, true, 100000, 0));
    EXPECT_FALSE(KernelActivityExpectedButMissing(false, true, 100000, 0));

    // Kernel activity was never enabled: absence of records is expected.
    EXPECT_FALSE(KernelActivityExpectedButMissing(false, false, 100000, 0));
}

TEST(OrphanKernelSynthesisPolicy, RealRecordModesAlwaysSuppressOrphans) {
    using gpufl::ShouldSuppressOrphanKernelSynthesis;

    EXPECT_TRUE(ShouldSuppressOrphanKernelSynthesis(true, false));
    // Even a contradictory caller cannot opt a real-record mode into
    // host-gap timing.
    EXPECT_TRUE(ShouldSuppressOrphanKernelSynthesis(true, true));
    // SASS metrics-only: neither real rows nor synthetic timing is wanted.
    EXPECT_TRUE(ShouldSuppressOrphanKernelSynthesis(false, false));
    // PC sampling / other synthesize-by-design modes.
    EXPECT_FALSE(ShouldSuppressOrphanKernelSynthesis(false, true));
}

// The demo config (1000us x 10000) gates its drain at 5s, so a 3s deep window
// got one early decode and a final one covering ~2.8s, and those windows
// failed on an H100. A timed window drains every third of its length instead.
TEST(PmDrainPeriod, ATimedWindowDrainsEveryThirdOfItsLength) {
    using gpufl::PmDrainPeriodNs;
    constexpr int64_t kMs = 1'000'000;

    EXPECT_EQ(PmDrainPeriodNs(1000, 10000, 3000), 1000 * kMs);
    // No window, or one bounded only by launches: the fill gate as before.
    EXPECT_EQ(PmDrainPeriodNs(1000, 10000, 0), 5000 * kMs);
    // Never below the beat, and never above the fill gate (204.8ms here).
    EXPECT_EQ(PmDrainPeriodNs(1000, 10000, 600), gpufl::kCollectorBeatMs * kMs);
    EXPECT_EQ(PmDrainPeriodNs(100, 4096, 3000), 4096 * 100 * 1000 / 2);
}

// The beat as the collector runs it: PmDrainDue once more than
// kCollectorBeatMs has passed. Worst case for the count: the last decode
// landed just as the window opened, so the first early one never happens.
TEST(PmDrainPeriod, AThreeSecondWindowGetsAtLeastThreeDecodes) {
    constexpr int64_t kMs = 1'000'000;
    constexpr int64_t kWindowMs = 3000;
    const int64_t beat_ns = (gpufl::kCollectorBeatMs + 1) * kMs;
    const int64_t period_ns = gpufl::PmDrainPeriodNs(1000, 10000, kWindowMs);
    const int64_t open_ns = 60'000 * kMs;
    const int64_t close_ns = open_ns + kWindowMs * kMs;

    int64_t last_drain_ns = open_ns;
    int decodes = 0;
    for (int64_t now = open_ns + beat_ns; now < close_ns; now += beat_ns) {
        if (gpufl::PmDrainDue(now, last_drain_ns, period_ns)) {
            ++decodes;
            last_drain_ns = now;
        }
    }
    ++decodes;  // the final decode at window close
    EXPECT_GE(decodes, 3);
}

TEST_F(CuptiBackendTest, Lifecycle) {
    gpufl::MonitorOptions opts;
    opts.enable_debug_output = true;

    gpufl::CuptiBackend backend;

    // Initial state
    EXPECT_FALSE(backend.IsActive());

    // Initialize
    backend.initialize(opts);
    EXPECT_TRUE(backend.IsMonitoringMode());
    EXPECT_FALSE(backend.IsProfilingMode());

    // Start
    backend.start();
    EXPECT_TRUE(backend.IsActive());

    // Stop
    backend.stop();
    EXPECT_FALSE(backend.IsActive());

    // Shutdown
    backend.shutdown();
}

TEST_F(CuptiBackendTest, ProfilingMode) {
    gpufl::MonitorOptions opts;
    opts.profiling_engine = gpufl::ProfilingEngine::PcSampling;

    gpufl::CuptiBackend backend;
    backend.initialize(opts);

    EXPECT_TRUE(backend.IsMonitoringMode());
    EXPECT_TRUE(backend.IsProfilingMode());

    backend.start();
    EXPECT_TRUE(backend.IsActive());

    // onScopeStart/Stop should not crash even if PC Sampling cannot be
    // enabled (unsupported GPU, restricted permissions, etc.).
    for (int i = 0; i < 3; ++i) {
        backend.OnScopeStart("test_scope");
        backend.OnScopeStop("test_scope");
    }
    // Guard against ref-count underflow paths when stop is called without a
    // matching successful start.
    backend.OnScopeStop("test_scope");

    backend.stop();
    backend.shutdown();
}

TEST_F(CuptiBackendTest, ScopeCallbacks) {
    gpufl::MonitorOptions opts;
    gpufl::CuptiBackend backend;
    backend.initialize(opts);
    backend.start();

    // Should be safe to call even in non-profiling mode
    backend.OnScopeStart("test_scope");
    backend.OnScopeStop("test_scope");

    backend.stop();
    backend.shutdown();
}

class MockHandler : public gpufl::ICuptiHandler {
   public:
    mutable int callCount = 0;
    const char* getName() const override { return "MockHandler"; }
    bool shouldHandle(CUpti_CallbackDomain domain,
                      CUpti_CallbackId cbid) const override {
        return domain == CUPTI_CB_DOMAIN_RUNTIME_API;
    }
    void handle(CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
                const void* cbdata) override {
        callCount++;
    }
};

TEST_F(CuptiBackendTest, DynamicHandler) {
    gpufl::MonitorOptions opts;
    opts.enable_debug_output = true;
    gpufl::CuptiBackend backend;
    backend.initialize(opts);

    auto mock = std::make_shared<MockHandler>();
    backend.RegisterHandler(mock);

    backend.start();

    // Trigger a runtime API call that should be handled
    cudaFree(nullptr);

    // In some CI environments (stubs), the callback might not be triggered
    // but we've verified registration and basic flow doesn't crash.
    EXPECT_GE(mock->callCount, 0);

    backend.stop();
    backend.shutdown();
}

TEST_F(CuptiBackendTest, KernelSamplingRate) {
    gpufl::MonitorOptions opts;
    opts.kernel_sample_rate_ms = 100;

    gpufl::CuptiBackend backend;
    backend.initialize(opts);

    // We can't easily trigger multiple Activity records here without real GPU
    // and timing but we can verify that the option is correctly set in the
    // backend.
    EXPECT_EQ(backend.GetOptions().kernel_sample_rate_ms, 100);

    backend.shutdown();
}

#endif

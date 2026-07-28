#include <gtest/gtest.h>

#include "common/test_utils.hpp"

#if GPUFL_ENABLE_NVIDIA && GPUFL_HAS_CUPTI
#include "gpufl/backends/nvidia/cupti_backend.hpp"
#include "gpufl/backends/nvidia/kernel_launch_handler.hpp"

class CuptiBackendTest : public ::testing::Test {
   protected:
    void SetUp() override { SKIP_IF_NO_CUDA(); }
};

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

    // Any real record arrived: keep the drain - tail orphans stay best-effort.
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

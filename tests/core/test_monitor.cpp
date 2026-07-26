#include <gtest/gtest.h>

#include <chrono>
#include <thread>

#include "common/test_utils.hpp"
#include "gpufl/core/monitor.hpp"
#include "gpufl/core/monitor_batch_manager.hpp"

class MonitorTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Monitor tests might need CUDA if they initialize the CuptiBackend
        // But we can also test them in a way that is safe.
        // If we want to test the full lifecycle, we should probably check for
        // CUDA.
    }

    void TearDown() override { gpufl::Monitor::Shutdown(); }
};

TEST_F(MonitorTest, Lifecycle) {
    gpufl::MonitorOptions opts;
    opts.enable_debug_output = true;

    // Initialize
    gpufl::Monitor::Initialize(opts);

    // Start
    gpufl::Monitor::Start();

    // Stop
    gpufl::Monitor::Stop();

    // Shutdown
    gpufl::Monitor::Shutdown();
}

TEST_F(MonitorTest, RangePushPop) {
    gpufl::MonitorOptions opts;
    gpufl::Monitor::Initialize(opts);
    gpufl::Monitor::Start();

    // Test ranges - these should push events to the ring buffer
    gpufl::Monitor::PushRange("outer_range");
    gpufl::Monitor::PushRange("inner_range");
    gpufl::Monitor::PopRange();
    gpufl::Monitor::PopRange();

    // Give the collector thread a moment to process (though we don't strictly
    // check the output here)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    gpufl::Monitor::Stop();
    gpufl::Monitor::Shutdown();
}

TEST_F(MonitorTest, ProfilerScopes) {
    gpufl::MonitorOptions opts;
    gpufl::Monitor::Initialize(opts);
    gpufl::Monitor::Start();

    // Test profiler scopes
    gpufl::Monitor::BeginProfilerScope("prof_scope");
    gpufl::Monitor::EndProfilerScope("prof_scope");

    gpufl::Monitor::Stop();
    gpufl::Monitor::Shutdown();
}

TEST_F(MonitorTest, MultipleInitialize) {
    gpufl::MonitorOptions opts;
    gpufl::Monitor::Initialize(opts);
    gpufl::Monitor::Initialize(opts);  // Should be safe

    gpufl::Monitor::Shutdown();
    gpufl::Monitor::Shutdown();  // Should be safe
}

// ── scope name stack ────────────────────────────────────────────────────────
//
// The active scope name is what stamps every profile and PM sample. It has to
// behave as a stack: a deep window opens inside the process scope and must
// hand the name back when it closes, or every sample taken after the window
// keeps the window's name.

namespace {

gpufl::ScopeBatchRow ScopeRow(uint64_t instance_id, uint32_t name_id,
                              uint8_t event_type, int depth = 0) {
    gpufl::ScopeBatchRow row;
    row.ts_ns = 1000 + static_cast<int64_t>(instance_id);
    row.scope_instance_id = instance_id;
    row.name_id = name_id;
    row.event_type = event_type;
    row.depth = depth;
    return row;
}

}  // namespace

TEST(ScopeNameStackTest, NestedCloseRestoresTheEnclosingName) {
    gpufl::detail::MonitorBatchManager m;
    const uint32_t outer = m.internScopeName("process:app");
    const uint32_t inner = m.internScopeName("deep_window");

    m.pushTrackedScopeRow(ScopeRow(1, outer, 0, 0));
    EXPECT_EQ(m.activeScopeNameId(), outer);

    m.pushTrackedScopeRow(ScopeRow(2, inner, 0, 1));
    EXPECT_EQ(m.activeScopeNameId(), inner);

    // The regression this guards: without a stack the name stayed on the
    // window and every later sample was attributed to it.
    m.pushTrackedScopeRow(ScopeRow(2, inner, 1));
    EXPECT_EQ(m.activeScopeNameId(), outer);

    m.pushTrackedScopeRow(ScopeRow(1, outer, 1));
    EXPECT_EQ(m.activeScopeNameId(), 0u);
}

TEST(ScopeNameStackTest, OutOfOrderCloseRemovesOnlyItsOwnScope) {
    // A deep window closes from the collector thread, so it can close while an
    // application scope opened after it is still open. Closing must not pop
    // whatever happens to be on top.
    gpufl::detail::MonitorBatchManager m;
    const uint32_t process = m.internScopeName("process:app");
    const uint32_t window = m.internScopeName("deep_window");
    const uint32_t user = m.internScopeName("user_scope");

    m.pushTrackedScopeRow(ScopeRow(1, process, 0, 0));
    m.pushTrackedScopeRow(ScopeRow(2, window, 0, 1));
    m.pushTrackedScopeRow(ScopeRow(3, user, 0, 2));

    m.pushTrackedScopeRow(ScopeRow(2, window, 1));
    EXPECT_EQ(m.activeScopeNameId(), user) << "closing the window must leave the user scope active";

    m.pushTrackedScopeRow(ScopeRow(3, user, 1));
    EXPECT_EQ(m.activeScopeNameId(), process);
}

TEST(ScopeNameStackTest, UnmatchedCloseLeavesTheStackAlone) {
    gpufl::detail::MonitorBatchManager m;
    const uint32_t outer = m.internScopeName("process:app");

    m.pushTrackedScopeRow(ScopeRow(1, outer, 0, 0));
    m.pushTrackedScopeRow(ScopeRow(99, outer, 1));

    EXPECT_EQ(m.activeScopeNameId(), outer);
    EXPECT_EQ(m.openScopeDepth(), 1);
}

TEST(ScopeNameStackTest, DepthReportsWhereANewScopeWouldNest) {
    gpufl::detail::MonitorBatchManager m;
    const uint32_t outer = m.internScopeName("process:app");
    const uint32_t inner = m.internScopeName("deep_window");

    EXPECT_EQ(m.openScopeDepth(), 0);
    m.pushTrackedScopeRow(ScopeRow(1, outer, 0, 0));
    EXPECT_EQ(m.openScopeDepth(), 1);
    m.pushTrackedScopeRow(ScopeRow(2, inner, 0, 1));
    EXPECT_EQ(m.openScopeDepth(), 2);
    m.pushTrackedScopeRow(ScopeRow(2, inner, 1));
    EXPECT_EQ(m.openScopeDepth(), 1);
}

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

#include "common/test_utils.hpp"
#include "gpufl/core/monitor.hpp"

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

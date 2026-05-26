// Unit tests for Sampler's ref-counted activation model.
//
// These tests cover the mechanics of activate() / deactivate() / shutdown()
// directly — they do NOT exercise the full gpufl::init / GFL_SCOPE flow
// (those are integration concerns and require CUDA/CUPTI). The goal here
// is to verify:
//
//   * configure() alone does not start the worker thread
//   * activate() on 0→1 starts the worker; deactivate() on 1→0 stops it
//   * nested activations keep the worker running until all are released
//   * activate() before configure() is safe (counter increments, no thread)
//   * unbalanced deactivate() clamps at zero without crashing
//   * shutdown() force-zeroes the counter and stops the worker
//   * the worker actually invokes the collector while running
//
// Together these guarantee the contract that the new
// `continuous_system_sampling=false` + scope-bracketing path relies on:
// scope-driven activate/deactivate compose correctly with manual
// systemStart/stop and continuous-mode baseline activation.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

#include "gpufl/core/logger/logger.hpp"
#include "gpufl/core/sampler.hpp"

namespace {

/// Collector that returns no samples but counts how many times it was
/// invoked. Lets us verify the worker thread is actually polling without
/// having to set up an NVML context.
class CountingNullCollector
    : public gpufl::ISystemCollector<gpufl::DeviceSample> {
   public:
    std::vector<gpufl::DeviceSample> sampleAll() override {
        calls_.fetch_add(1, std::memory_order_relaxed);
        return {};
    }
    int calls() const { return calls_.load(std::memory_order_relaxed); }

   private:
    std::atomic<int> calls_{0};
};

// Logger doesn't need to be opened — with zero sinks attached, write()
// is a no-op. The Sampler still calls write() at flush boundaries, but
// the model serialization happens on an empty batch and goes nowhere.
std::shared_ptr<gpufl::Logger> makeUnopenedLogger() {
    return std::make_shared<gpufl::Logger>();
}

}  // namespace

TEST(SamplerRefCount, ConfiguredButNotActivated) {
    gpufl::Sampler s;
    s.configure("app", "session", makeUnopenedLogger(),
                std::make_shared<CountingNullCollector>(),
                /*sampleIntervalMs=*/10);
    EXPECT_FALSE(s.running());
    EXPECT_EQ(s.activations(), 0);
}

TEST(SamplerRefCount, ActivateStartsWorker) {
    gpufl::Sampler s;
    s.configure("app", "session", makeUnopenedLogger(),
                std::make_shared<CountingNullCollector>(),
                /*sampleIntervalMs=*/10);
    s.activate();
    EXPECT_EQ(s.activations(), 1);
    EXPECT_TRUE(s.running());

    s.deactivate();
    EXPECT_EQ(s.activations(), 0);
    EXPECT_FALSE(s.running());
}

TEST(SamplerRefCount, NestedActivationsKeepRunningUntilLastRelease) {
    gpufl::Sampler s;
    s.configure("app", "session", makeUnopenedLogger(),
                std::make_shared<CountingNullCollector>(),
                /*sampleIntervalMs=*/10);

    s.activate();  // baseline (e.g. continuous mode)
    s.activate();  // scope entry
    s.activate();  // explicit systemStart inside the scope
    EXPECT_EQ(s.activations(), 3);
    EXPECT_TRUE(s.running());

    s.deactivate();  // systemStop
    EXPECT_EQ(s.activations(), 2);
    EXPECT_TRUE(s.running());

    s.deactivate();  // scope exit
    EXPECT_EQ(s.activations(), 1);
    EXPECT_TRUE(s.running());

    s.deactivate();  // shutdown's matching deactivate of the baseline
    EXPECT_EQ(s.activations(), 0);
    EXPECT_FALSE(s.running());
}

TEST(SamplerRefCount, ActivateBeforeConfigureIsSafe) {
    gpufl::Sampler s;
    // No configure() — represents calling activate() from a code path
    // that races init(). Counter still tracks, but no worker spawns
    // because there's nothing useful to do.
    s.activate();
    EXPECT_EQ(s.activations(), 1);
    EXPECT_FALSE(s.running());

    s.deactivate();
    EXPECT_EQ(s.activations(), 0);
    EXPECT_FALSE(s.running());
}

TEST(SamplerRefCount, UnbalancedDeactivateClampsAtZero) {
    gpufl::Sampler s;
    // Caller's bug — deactivate() with no matching activate().
    // Must clamp at zero rather than going negative; that lets later
    // activate() calls still work correctly.
    s.deactivate();
    EXPECT_EQ(s.activations(), 0);
    EXPECT_FALSE(s.running());

    s.configure("app", "session", makeUnopenedLogger(),
                std::make_shared<CountingNullCollector>(),
                /*sampleIntervalMs=*/10);
    s.activate();
    EXPECT_EQ(s.activations(), 1);
    EXPECT_TRUE(s.running());

    s.deactivate();
    EXPECT_FALSE(s.running());
}

TEST(SamplerRefCount, ShutdownForceZeroesEvenWithPendingActivations) {
    gpufl::Sampler s;
    s.configure("app", "session", makeUnopenedLogger(),
                std::make_shared<CountingNullCollector>(),
                /*sampleIntervalMs=*/10);
    s.activate();
    s.activate();
    s.activate();
    EXPECT_EQ(s.activations(), 3);
    EXPECT_TRUE(s.running());

    // Simulates gpufl::shutdown() running while scopes are still alive
    // (defends against leaked ScopedMonitor destructors).
    s.shutdown();
    EXPECT_EQ(s.activations(), 0);
    EXPECT_FALSE(s.running());
}

TEST(SamplerRefCount, WorkerActuallyInvokesCollectorBetweenActivations) {
    gpufl::Sampler s;
    auto coll = std::make_shared<CountingNullCollector>();
    s.configure("app", "session", makeUnopenedLogger(), coll,
                /*sampleIntervalMs=*/5);
    s.activate();
    // Give the worker enough wall time to call sampleAll() at least a
    // few times. 5ms interval × ~6 iterations = 30ms of sleep.
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    s.deactivate();
    EXPECT_GE(coll->calls(), 3)
        << "Worker should have polled the collector multiple times "
        << "during the 40ms window with a 5ms interval.";
}

TEST(SamplerRefCount, ReactivationAfterFullReleaseSpawnsFreshWorker) {
    gpufl::Sampler s;
    auto coll = std::make_shared<CountingNullCollector>();
    s.configure("app", "session", makeUnopenedLogger(), coll,
                /*sampleIntervalMs=*/5);

    s.activate();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    s.deactivate();
    const int calls_after_first_run = coll->calls();
    EXPECT_GT(calls_after_first_run, 0);
    EXPECT_FALSE(s.running());

    // Second activation cycle — verifies that std::thread can be
    // re-spawned after a full join. (The previous implementation reused
    // a single thread handle; the new ref-counted impl moves the handle
    // out before joining and creates a fresh one on the next 0→1.)
    s.activate();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    s.deactivate();
    EXPECT_GT(coll->calls(), calls_after_first_run)
        << "Second activation should produce additional collector calls.";
}

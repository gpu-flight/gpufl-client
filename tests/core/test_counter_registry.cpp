#include <gtest/gtest.h>

#include <thread>
#include <vector>

#include "gpufl.hpp"
#include "gpufl/core/counter_registry.hpp"
#include "gpufl/core/monitor.hpp"

#include <limits>

using gpufl::detail::CounterRegistry;

namespace {

class CounterRegistryTest : public ::testing::Test {
   protected:
    void SetUp() override { CounterRegistry::instance().resetForTesting(); }
    void TearDown() override { CounterRegistry::instance().resetForTesting(); }

    CounterRegistry& reg() { return CounterRegistry::instance(); }
};

}  // namespace

TEST_F(CounterRegistryTest, SameNameResolvesToOneSlot) {
    const auto a = reg().registerCounter("token");
    const auto b = reg().registerCounter("token");
    EXPECT_EQ(a, b);
    EXPECT_EQ(reg().counterCount(), 1u);
}

TEST_F(CounterRegistryTest, InvalidNamesAreRejected) {
    EXPECT_EQ(reg().registerCounter(""), CounterRegistry::kInvalidSlot);
    EXPECT_EQ(reg().registerCounter("has space"), CounterRegistry::kInvalidSlot);
    EXPECT_EQ(reg().registerCounter("has/slash"), CounterRegistry::kInvalidSlot);
    EXPECT_EQ(reg().registerCounter(std::string(200, 'x')), CounterRegistry::kInvalidSlot);
    EXPECT_EQ(reg().counterCount(), 0u);
}

TEST_F(CounterRegistryTest, CardinalityIsCapped) {
    for (size_t i = 0; i < CounterRegistry::kMaxCounters; ++i) {
        ASSERT_NE(reg().registerCounter("c" + std::to_string(i)),
                  CounterRegistry::kInvalidSlot);
    }
    // A permanent slot table is exactly the thing that must not grow without
    // limit, so the one past the cap is refused rather than accepted.
    EXPECT_EQ(reg().registerCounter("one_too_many"), CounterRegistry::kInvalidSlot);
    EXPECT_EQ(reg().counterCount(), CounterRegistry::kMaxCounters);
}

TEST_F(CounterRegistryTest, NonPositiveAddsAreIgnored) {
    auto tokens = gpufl::counter("token");
    ASSERT_TRUE(tokens.valid());
    tokens.add(5);
    tokens.add(0);
    tokens.add(-100);   // must not wrap the unsigned counter
    EXPECT_EQ(reg().rawValue(reg().registerCounter("token")), 5u);
}

TEST_F(CounterRegistryTest, InvalidHandleAddIsANoOp) {
    const auto bad = gpufl::counter("not a valid name");
    EXPECT_FALSE(bad.valid());
    bad.add(1);   // must not crash
}

TEST_F(CounterRegistryTest, AddsBeforeASessionAreExcludedFromIt) {
    // Slots outlive the runtime, so anything ticked before init() is already in
    // the slot. Without a baseline the new session would count it as its own.
    auto tokens = gpufl::counter("token");
    tokens.add(1000);

    const auto slot = reg().registerCounter("token");
    EXPECT_EQ(reg().rawValue(slot), 1000u);

    reg().beginSession();
    EXPECT_EQ(reg().valueSinceBaseline(slot), 0u) << "pre-init ticks are not this session's";

    tokens.add(7);
    EXPECT_EQ(reg().valueSinceBaseline(slot), 7u);
}

TEST_F(CounterRegistryTest, AddsBetweenSessionsAreExcludedFromTheNext) {
    auto tokens = gpufl::counter("token");
    const auto slot = reg().registerCounter("token");

    reg().beginSession();
    tokens.add(10);
    EXPECT_EQ(reg().valueSinceBaseline(slot), 10u);
    reg().endSession();

    // gpufl is down; the handle still works and the slot still accumulates.
    tokens.add(500);

    reg().beginSession();
    EXPECT_EQ(reg().valueSinceBaseline(slot), 0u)
        << "ticks while no runtime was active belong to no session";
    tokens.add(3);
    EXPECT_EQ(reg().valueSinceBaseline(slot), 3u);
}

TEST_F(CounterRegistryTest, HandleSurvivesASessionChange) {
    // The point of a process-lifetime slot: a handle held in a static, or by an
    // embedded host across shutdown()/init(), keeps working. A generation
    // number alone could not make that safe, since it cannot stop the state
    // being freed underneath a concurrent add().
    auto tokens = gpufl::counter("token");
    const auto slot = reg().registerCounter("token");

    reg().beginSession();
    reg().endSession();
    reg().beginSession();

    tokens.add(4);
    EXPECT_EQ(reg().valueSinceBaseline(slot), 4u);
}

TEST_F(CounterRegistryTest, CounterRegisteredMidSessionStartsAtZero) {
    reg().beginSession();
    auto late = gpufl::counter("late");
    late.add(9);
    EXPECT_EQ(reg().valueSinceBaseline(reg().registerCounter("late")), 9u);
}

TEST_F(CounterRegistryTest, ConcurrentRegistrationOfOneNameYieldsOneSlot) {
    constexpr int kThreads = 8;
    std::vector<std::thread> threads;
    std::vector<CounterRegistry::SlotId> results(kThreads);
    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&, i] { results[i] = reg().registerCounter("shared"); });
    }
    for (auto& t : threads) t.join();

    for (const auto slot : results) EXPECT_EQ(slot, results[0]);
    EXPECT_EQ(reg().counterCount(), 1u);
}

TEST_F(CounterRegistryTest, ConcurrentAddsAreNotLost) {
    auto tokens = gpufl::counter("token");
    const auto slot = reg().registerCounter("token");
    reg().beginSession();

    constexpr int kThreads = 8;
    constexpr int kPerThread = 10'000;
    std::vector<std::thread> threads;
    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&] {
            for (int n = 0; n < kPerThread; ++n) tokens.add(1);
        });
    }
    for (auto& t : threads) t.join();

    EXPECT_EQ(reg().valueSinceBaseline(slot),
              static_cast<uint64_t>(kThreads) * kPerThread);
}

TEST_F(CounterRegistryTest, TickIsEquivalentButLooksThePriceUpEveryCall) {
    gpufl::tick("steps");
    gpufl::tick("steps", 4);
    EXPECT_EQ(reg().rawValue(reg().registerCounter("steps")), 5u);
}

// ── lifecycle wiring ────────────────────────────────────────────────────────
//
// The tests above drive the registry directly, so they pass whether or not
// Monitor actually calls it. These go through Monitor::Initialize/Shutdown so
// that deleting the calls in monitor.cpp fails something.

TEST_F(CounterRegistryTest, MonitorInitializeBaselinesThroughTheRealWiring) {
    auto tokens = gpufl::counter("token");
    const auto slot = reg().registerCounter("token");
    tokens.add(1000);   // before any session exists

    gpufl::MonitorOptions opts;
    gpufl::Monitor::Initialize(opts);
    EXPECT_EQ(reg().valueSinceBaseline(slot), 0u)
        << "Monitor::Initialize must baseline the registry";
    EXPECT_TRUE(reg().sessionActive());

    tokens.add(6);
    EXPECT_EQ(reg().valueSinceBaseline(slot), 6u);

    gpufl::Monitor::Shutdown();
    EXPECT_FALSE(reg().sessionActive())
        << "Monitor::Shutdown must close the session";
}

TEST_F(CounterRegistryTest, TicksBetweenTwoMonitorSessionsBelongToNeither) {
    auto tokens = gpufl::counter("token");
    const auto slot = reg().registerCounter("token");

    gpufl::MonitorOptions opts;
    gpufl::Monitor::Initialize(opts);
    tokens.add(10);
    EXPECT_EQ(reg().valueSinceBaseline(slot), 10u);
    gpufl::Monitor::Shutdown();

    // gpufl is down. The handle still works - the slot is process-lifetime -
    // but these ticks are nobody's.
    tokens.add(500);

    gpufl::Monitor::Initialize(opts);
    EXPECT_EQ(reg().valueSinceBaseline(slot), 0u);
    tokens.add(3);
    EXPECT_EQ(reg().valueSinceBaseline(slot), 3u);
    gpufl::Monitor::Shutdown();
}

TEST_F(CounterRegistryTest, ProcessExitPathAlsoClosesTheSession) {
    // DrainAndFinalizeForExit is a separate teardown from Shutdown(); an
    // embedded host that re-initialised after it would otherwise inherit the
    // previous session's ticks.
    gpufl::MonitorOptions opts;
    gpufl::Monitor::Initialize(opts);
    ASSERT_TRUE(reg().sessionActive());

    gpufl::Monitor::DrainAndFinalizeForExit();
    EXPECT_FALSE(reg().sessionActive());
}

TEST_F(CounterRegistryTest, AddAboveThePerCallBoundIsRefused) {
    // Not overflow protection: a value this large is a caller bug - a pointer,
    // an uninitialised field - and letting it through makes every later rate
    // meaningless.
    auto tokens = gpufl::counter("token");
    const auto slot = reg().registerCounter("token");
    tokens.add(gpufl::Counter::kMaxAddPerCall);
    tokens.add(gpufl::Counter::kMaxAddPerCall + 1);   // refused
    EXPECT_EQ(reg().rawValue(slot),
              static_cast<uint64_t>(gpufl::Counter::kMaxAddPerCall));
}

TEST_F(CounterRegistryTest, DeltaIsCorrectAcrossAWrap) {
    // Rates are unsigned deltas precisely so a wrap needs no saturation or CAS
    // loop. Drive the counter to just below the wrap and cross it.
    auto tokens = gpufl::counter("token");
    const auto slot = reg().registerCounter("token");
    auto* raw = reg().valueSlot(slot);
    ASSERT_NE(raw, nullptr);

    raw->store(std::numeric_limits<uint64_t>::max() - 5, std::memory_order_relaxed);
    reg().beginSession();
    tokens.add(10);   // wraps past zero

    EXPECT_EQ(reg().valueSinceBaseline(slot), 10u)
        << "unsigned subtraction must stay correct across a wrap";
}

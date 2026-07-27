#include <gtest/gtest.h>

#include <algorithm>
#include <thread>
#include <vector>

#include "gpufl.hpp"
#include "gpufl/core/counter_registry.hpp"
#include "gpufl/core/counter_provider.hpp"
#include "gpufl/core/monitor.hpp"

#include <cstring>
#include <limits>

using gpufl::detail::CounterRegistry;

namespace {

class CounterRegistryTest : public ::testing::Test {
   protected:
    void SetUp() override { CounterRegistry::instance().resetForTesting(); }
    void TearDown() override { CounterRegistry::instance().resetForTesting(); }

    CounterRegistry& reg() { return CounterRegistry::instance(); }
};

// ── runtime discovery ───────────────────────────────────────────────────────

TEST_F(CounterRegistryTest, ResolutionLooksBesideTheInjectionLibrary) {
    // The motivating case: an ordinary C++ target links gpufl STATICALLY, so
    // the provider's own module directory is the TARGET's directory, which has
    // no runtime beside it. If that target calls counter() before its first
    // CUDA call - entirely normal - it binds to a local registry and stays
    // there, invisible to the evaluator that loads later.
    //
    // The launcher puts CUDA_INJECTION64_PATH in the child environment before
    // exec, so the runtime's location is knowable from the first instruction.
    // This checks the candidate list uses it; whether the file is actually
    // there is a deployment question the build rules cover.
    const auto candidates = gpufl::detail::CounterRuntimeCandidatesForTesting(
        "/opt/gpufl/lib/libgpufl_inject.so", nullptr);

    const bool beside_inject = std::any_of(
        candidates.begin(), candidates.end(), [](const std::string& c) {
            return c.rfind("/opt/gpufl/lib/", 0) == 0;
        });
    EXPECT_TRUE(beside_inject)
        << "a target that ticks before CUDA init can never find the runtime";
}

TEST_F(CounterRegistryTest, AnExplicitRuntimePathIsTriedFirst) {
    // Deployment layouts we do not control need a way to say where it is
    // rather than have us guess.
    const auto candidates = gpufl::detail::CounterRuntimeCandidatesForTesting(
        nullptr, "/custom/place/libgpufl_counter_runtime.so");
    ASSERT_FALSE(candidates.empty());
    EXPECT_EQ(candidates.front(), "/custom/place/libgpufl_counter_runtime.so");
}

}  // namespace

namespace {

const gpufl_counter_provider_v1* prov() {
    return gpufl::detail::ActiveCounterProvider();
}

// Read back through the ACTIVE provider, not the local registry. With a shared
// runtime present, gpufl::counter() writes into the runtime's registry and this
// module's own is untouched - reading the latter would compare two different
// registries and fail for the right reason in the wrong test.
uint64_t Raw(const char* name) {
    auto h = prov()->register_counter(name, std::strlen(name));
    return h ? prov()->load(h) : 0;
}

uint64_t Since(const char* name) {
    auto h = prov()->register_counter(name, std::strlen(name));
    return h ? prov()->load_since_baseline(h) : 0;
}

void BeginSession() { prov()->begin_session(); }
void EndSession() { prov()->end_session(); }

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
    auto tokens = gpufl::counter("nonpositive_probe");
    ASSERT_TRUE(tokens.valid());
    tokens.add(5);
    tokens.add(0);
    tokens.add(-100);   // must not wrap the unsigned counter
    EXPECT_EQ(Raw("nonpositive_probe"), 5u);
}

TEST_F(CounterRegistryTest, InvalidHandleAddIsANoOp) {
    const auto bad = gpufl::counter("not a valid name");
    EXPECT_FALSE(bad.valid());
    bad.add(1);   // must not crash
}

TEST_F(CounterRegistryTest, AddsBeforeASessionAreExcludedFromIt) {
    // Slots outlive the runtime, so anything ticked before init() is already in
    // the slot. Without a baseline the new session would count it as its own.
    auto tokens = gpufl::counter("pre_session_probe");
    tokens.add(1000);

    EXPECT_EQ(Raw("pre_session_probe"), 1000u);

    BeginSession();
    EXPECT_EQ(Since("pre_session_probe"), 0u) << "pre-init ticks are not this session's";

    tokens.add(7);
    EXPECT_EQ(Since("pre_session_probe"), 7u);
}

TEST_F(CounterRegistryTest, AddsBetweenSessionsAreExcludedFromTheNext) {
    auto tokens = gpufl::counter("token");

    BeginSession();
    tokens.add(10);
    EXPECT_EQ(Since("token"), 10u);
    EndSession();

    // gpufl is down; the handle still works and the slot still accumulates.
    tokens.add(500);

    BeginSession();
    EXPECT_EQ(Since("token"), 0u)
        << "ticks while no runtime was active belong to no session";
    tokens.add(3);
    EXPECT_EQ(Since("token"), 3u);
}

TEST_F(CounterRegistryTest, HandleSurvivesASessionChange) {
    // The point of a process-lifetime slot: a handle held in a static, or by an
    // embedded host across shutdown()/init(), keeps working. A generation
    // number alone could not make that safe, since it cannot stop the state
    // being freed underneath a concurrent add().
    auto tokens = gpufl::counter("token");

    BeginSession();
    EndSession();
    BeginSession();

    tokens.add(4);
    EXPECT_EQ(Since("token"), 4u);
}

TEST_F(CounterRegistryTest, CounterRegisteredMidSessionStartsAtZero) {
    BeginSession();
    auto late = gpufl::counter("late");
    late.add(9);
    EXPECT_EQ(Since("late"), 9u);
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
    BeginSession();

    constexpr int kThreads = 8;
    constexpr int kPerThread = 10'000;
    std::vector<std::thread> threads;
    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&] {
            for (int n = 0; n < kPerThread; ++n) tokens.add(1);
        });
    }
    for (auto& t : threads) t.join();

    EXPECT_EQ(Since("token"),
              static_cast<uint64_t>(kThreads) * kPerThread);
}

TEST_F(CounterRegistryTest, TickIsEquivalentButLooksThePriceUpEveryCall) {
    gpufl::tick("steps");
    gpufl::tick("steps", 4);
    EXPECT_EQ(Raw("steps"), 5u);
}

// ── lifecycle wiring ────────────────────────────────────────────────────────
//
// The tests above drive the registry directly, so they pass whether or not
// Monitor actually calls it. These go through Monitor::Initialize/Shutdown so
// that deleting the calls in monitor.cpp fails something.

TEST_F(CounterRegistryTest, MonitorInitializeBaselinesThroughTheRealWiring) {
    auto tokens = gpufl::counter("token");
    tokens.add(1000);   // before any session exists

    gpufl::MonitorOptions opts;
    gpufl::Monitor::Initialize(opts);
    EXPECT_EQ(Since("token"), 0u)
        << "Monitor::Initialize must baseline the registry";
    EXPECT_TRUE(prov()->session_active() != 0);

    tokens.add(6);
    EXPECT_EQ(Since("token"), 6u);

    gpufl::Monitor::Shutdown();
    EXPECT_FALSE(prov()->session_active() != 0)
        << "Monitor::Shutdown must close the session";
}

TEST_F(CounterRegistryTest, TicksBetweenTwoMonitorSessionsBelongToNeither) {
    auto tokens = gpufl::counter("token");

    gpufl::MonitorOptions opts;
    gpufl::Monitor::Initialize(opts);
    tokens.add(10);
    EXPECT_EQ(Since("token"), 10u);
    gpufl::Monitor::Shutdown();

    // gpufl is down. The handle still works - the slot is process-lifetime -
    // but these ticks are nobody's.
    tokens.add(500);

    gpufl::Monitor::Initialize(opts);
    EXPECT_EQ(Since("token"), 0u);
    tokens.add(3);
    EXPECT_EQ(Since("token"), 3u);
    gpufl::Monitor::Shutdown();
}

TEST_F(CounterRegistryTest, ProcessExitPathAlsoClosesTheSession) {
    // DrainAndFinalizeForExit is a separate teardown from Shutdown(); an
    // embedded host that re-initialised after it would otherwise inherit the
    // previous session's ticks.
    gpufl::MonitorOptions opts;
    gpufl::Monitor::Initialize(opts);
    ASSERT_TRUE(prov()->session_active() != 0);

    gpufl::Monitor::DrainAndFinalizeForExit();
    EXPECT_FALSE(prov()->session_active() != 0);
}

TEST_F(CounterRegistryTest, AddAboveThePerCallBoundIsRefused) {
    // Not overflow protection: a value this large is a caller bug - a pointer,
    // an uninitialised field - and letting it through makes every later rate
    // meaningless.
    auto tokens = gpufl::counter("bound_probe");
    tokens.add(gpufl::Counter::kMaxAddPerCall);
    tokens.add(gpufl::Counter::kMaxAddPerCall + 1);   // refused
    EXPECT_EQ(Raw("bound_probe"),
              static_cast<uint64_t>(gpufl::Counter::kMaxAddPerCall));
}

TEST_F(CounterRegistryTest, DeltaIsCorrectAcrossAWrap) {
    // Rates are unsigned deltas precisely so a wrap needs no saturation or CAS
    // loop. Driven against the in-module registry, since reaching the wrap
    // means writing the raw value rather than adding to it.
    auto* slot = reg().slotFor(reg().registerCounter("wrap_probe"));
    ASSERT_NE(slot, nullptr);

    slot->value.store(std::numeric_limits<uint64_t>::max() - 5,
                      std::memory_order_relaxed);
    reg().beginSession();
    CounterRegistry::addRaw(slot, 10);   // wraps past zero

    EXPECT_EQ(CounterRegistry::valueSinceBaseline(slot), 10u)
        << "unsigned subtraction must stay correct across a wrap";
}

// ── shared runtime binding ──────────────────────────────────────────────────

TEST_F(CounterRegistryTest, BindsToTheSharedRuntimeWhenItIsPresent) {
    // The cross-module property this whole ABI exists for cannot be proven from
    // inside one executable - that needs the launcher + Python E2E. What can be
    // checked here is that binding works at all when the library is reachable,
    // and that the fallback is taken (rather than crashing) when it is not.
    gpufl::detail::CounterProvider::resetForTesting();
    const auto* provider = gpufl::detail::CounterProvider::get();

    if (provider == nullptr) {
        GTEST_SKIP() << "gpufl_counter_runtime not colocated with the test binary; "
                        "fallback path exercised by every other test here";
    }
    EXPECT_EQ(provider->abi_version, GPUFL_COUNTER_ABI_VERSION);
    EXPECT_GE(provider->struct_size, sizeof(gpufl_counter_provider_v1));
    EXPECT_TRUE(gpufl::detail::CounterProvider::isShared());

    // Round-trip through the C ABI rather than the registry directly.
    auto handle = provider->register_counter("abi_probe", 9);
    ASSERT_NE(handle, nullptr);
    provider->begin_session();
    provider->add(handle, 7);
    EXPECT_EQ(provider->load_since_baseline(handle), 7u);
    provider->end_session();
}

TEST_F(CounterRegistryTest, ActiveProviderIsNeverNull) {
    // Callers must not each carry a fallback branch, so this holds whether or
    // not the shared runtime is present.
    EXPECT_NE(gpufl::detail::ActiveCounterProvider(), nullptr);
}

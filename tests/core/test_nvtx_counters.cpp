#include <gtest/gtest.h>

#include <string>

#include "gpufl/core/counter_provider.hpp"
#include "gpufl/core/counter_registry.hpp"
#include "gpufl/core/metric_id.hpp"
#include "gpufl/core/metric_registry.hpp"
#include "gpufl/core/model/deep_window_model.hpp"
#include "gpufl/core/nvtx_counters.hpp"

using gpufl::detail::ActiveCounterProvider;
using gpufl::detail::CounterRegistry;
using gpufl::detail::NvtxCounterBridge;
using ValueType = NvtxCounterBridge::ValueType;
using RegisterStatus = NvtxCounterBridge::RegisterStatus;

namespace {

// NVTX_COUNTER_SAMPLE_* from nvToolsExtCounters.h, restated so a change to the
// released ABI shows up here rather than as a counter that quietly stops
// matching.
constexpr uint8_t kZero        = 0;
constexpr uint8_t kUnchanged   = 1;
constexpr uint8_t kUnavailable = 2;

class NvtxCounterBridgeTest : public ::testing::Test {
   protected:
    void SetUp() override {
        CounterRegistry::instance().resetForTesting();
        NvtxCounterBridge::instance().resetForTesting();
    }
    void TearDown() override {
        NvtxCounterBridge::instance().resetForTesting();
        CounterRegistry::instance().resetForTesting();
    }

    NvtxCounterBridge& bridge() { return NvtxCounterBridge::instance(); }

    /// What a rule would read: the value accrued since this session started.
    uint64_t observed(const std::string& name) {
        const auto* provider = ActiveCounterProvider();
        gpufl_counter_handle h = provider->lookup(name.c_str(), name.size());
        return h == nullptr ? 0 : provider->load_since_baseline(h);
    }

    NvtxCounterBridge::RegisterResult acceptDelta(const char* domain,
                                                  const char* name) {
        auto r = bridge().registerCounter(domain, name, 0, ValueType::Delta);
        EXPECT_EQ(r.status, RegisterStatus::Accepted);
        return r;
    }
};

// ── what a sample MEANS ─────────────────────────────────────────────────────

TEST_F(NvtxCounterBridgeTest, DeltaSamplesAccumulateIntoTheRegistry) {
    ActiveCounterProvider()->begin_session();
    const auto r = acceptDelta("inference", "tokens");

    bridge().sampleDelta(r.id, 8);
    bridge().sampleDelta(r.id, 16);

    // A rate is derived from this by the metric registry; what the bridge owes
    // is the running total the application actually reported.
    EXPECT_EQ(observed("inference.tokens"), 24u);
}

TEST_F(NvtxCounterBridgeTest, ANoValueSampleIsNotAnEvent) {
    ActiveCounterProvider()->begin_session();
    const auto r = acceptDelta("inference", "tokens");
    bridge().sampleDelta(r.id, 10);

    bridge().sampleNoValue(r.id, kZero);
    bridge().sampleNoValue(r.id, kUnchanged);

    // ZERO and UNCHANGED both say the delta was zero. Counting either as +1
    // would invent traffic out of a sample whose entire purpose is to report
    // that there was none - and would make an idle workload look busy to the
    // rule watching it.
    EXPECT_EQ(observed("inference.tokens"), 10u);
    EXPECT_EQ(bridge().unavailableSamples(), 0u);
}

TEST_F(NvtxCounterBridgeTest, AnUnavailableSampleIsRecordedNotCounted) {
    ActiveCounterProvider()->begin_session();
    const auto r = acceptDelta("inference", "tokens");
    bridge().sampleDelta(r.id, 10);

    bridge().sampleNoValue(r.id, kUnavailable);

    // The application could not read its own counter. That must not move the
    // total, but it must not vanish either: the resulting rate rests on less
    // data than its sample count suggests.
    EXPECT_EQ(observed("inference.tokens"), 10u);
    EXPECT_EQ(bridge().unavailableSamples(), 1u);
}

TEST_F(NvtxCounterBridgeTest, ANegativeDeltaIsDroppedRatherThanWrapped) {
    ActiveCounterProvider()->begin_session();
    const auto r = acceptDelta("inference", "tokens");
    bridge().sampleDelta(r.id, 10);

    bridge().sampleDelta(r.id, -4);

    // The registry accumulates unsigned and rates are unsigned deltas, so
    // subtracting here would not read as -4; the very next bucket would report
    // a rate near 2^64.
    EXPECT_EQ(observed("inference.tokens"), 10u);
    EXPECT_EQ(bridge().negativeSamples(), 1u);
}

// ── what is refused, and why ────────────────────────────────────────────────

TEST_F(NvtxCounterBridgeTest, ACounterWithNoSemanticsIsRefused) {
    // nvtxCounterSampleInt64 says the value is an int64; it does not say
    // whether it is a delta or an absolute reading. Accumulating an absolute
    // series would produce a rate that looks reasonable and is wrong.
    const auto r = bridge().registerCounter("d", "c", 0, ValueType::Unspecified);
    EXPECT_EQ(r.status, RegisterStatus::UnsupportedValueType);
    EXPECT_EQ(r.id, 0u);
}

TEST_F(NvtxCounterBridgeTest, AnAbsoluteCounterIsRefused) {
    EXPECT_EQ(bridge().registerCounter("d", "c", 0, ValueType::Absolute).status,
              RegisterStatus::UnsupportedValueType);
    EXPECT_EQ(bridge()
                  .registerCounter("d", "c", 0, ValueType::DeltaSinceStart)
                  .status,
              RegisterStatus::UnsupportedValueType);
}

TEST_F(NvtxCounterBridgeTest, AnApplicationAssignedIdIsRefused) {
    // An NVTX counter id is unique only WITHIN its domain. Binding one to a
    // slot without a (domain, id) table would let two domains that both chose
    // id 123 share a slot and add their rates together.
    const auto r = bridge().registerCounter("d", "c", 1u << 24, ValueType::Delta);
    EXPECT_EQ(r.status, RegisterStatus::StaticIdUnsupported);
    EXPECT_EQ(r.id, 0u);
}

TEST_F(NvtxCounterBridgeTest, SamplesForAnIdWeNeverIssuedGoNowhere) {
    ActiveCounterProvider()->begin_session();
    const auto r = acceptDelta("inference", "tokens");

    // A refused registration returns 0, and the application keeps sampling.
    bridge().sampleDelta(0, 99);
    bridge().sampleDelta(1u << 24, 99);            // a static id
    bridge().sampleDelta(r.id + 1000, 99);         // past the table

    EXPECT_EQ(observed("inference.tokens"), 0u)
        << "an unknown id landed on someone else's slot";
    EXPECT_EQ(bridge().unknownIdSamples(), 3u);
}

// ── naming ──────────────────────────────────────────────────────────────────

TEST_F(NvtxCounterBridgeTest, TheDomainIsPartOfTheName) {
    // Two teams each calling their counter "tokens" in their own domain are
    // two counters, and a rule has to be able to name one of them.
    const auto a = acceptDelta("inference", "tokens");
    const auto b = acceptDelta("training", "tokens");
    EXPECT_NE(a.id, b.id);
    EXPECT_EQ(a.metric, "custom.inference.tokens_rate");
    EXPECT_EQ(b.metric, "custom.training.tokens_rate");
}

TEST_F(NvtxCounterBridgeTest, AnUndomainedCounterKeepsItsBareName) {
    EXPECT_EQ(NvtxCounterBridge::canonicalName("", "tokens"), "tokens");
}

TEST_F(NvtxCounterBridgeTest, OutOfCharsetBytesAreMappedNotDropped) {
    // Rules address counters as custom.<name>_rate over [A-Za-z0-9._-], and
    // NVTX names are free-form.
    EXPECT_EQ(NvtxCounterBridge::canonicalName("My Server", "tokens/sec"),
              "My_Server.tokens_sec");
    // Nothing usable left is a refusal, not a metric called custom.___rate.
    EXPECT_TRUE(NvtxCounterBridge::canonicalName("", "///").empty());
    EXPECT_TRUE(NvtxCounterBridge::canonicalName("", "").empty());
}

TEST_F(NvtxCounterBridgeTest, TwoNamesThatCanonicaliseAlikeAreRefused) {
    ActiveCounterProvider()->begin_session();
    // "a b" and "a/b" both canonicalise to "a_b", but they are DIFFERENT NVTX
    // counters. Sharing a binding would silently add two unrelated workloads
    // into one rate - a wrong number that looks like a real one - so the
    // second registration is refused and told to rename.
    const auto a = acceptDelta("", "a b");
    const auto b = bridge().registerCounter("", "a/b", 0, ValueType::Delta);
    EXPECT_EQ(b.status, RegisterStatus::BadName);
    EXPECT_EQ(b.id, 0u);
    EXPECT_EQ(bridge().trackedCount(), 1u);

    bridge().sampleDelta(a.id, 3);
    EXPECT_EQ(observed("a_b"), 3u) << "the refused counter leaked into the slot";
}

TEST_F(NvtxCounterBridgeTest, TheDomainJoinIsACollisionAxisToo) {
    // The domain joins with '.', so ("a", "b.c") and ("a.b", "c") meet at
    // "a.b.c" without any out-of-charset byte involved. Only the original
    // (domain, counter) pair can tell them apart.
    const auto first = acceptDelta("a", "b.c");
    const auto second = bridge().registerCounter("a.b", "c", 0, ValueType::Delta);
    EXPECT_EQ(first.metric, "custom.a.b.c_rate");
    EXPECT_EQ(second.status, RegisterStatus::BadName);
    EXPECT_EQ(bridge().trackedCount(), 1u);
}

TEST_F(NvtxCounterBridgeTest, RegisteringTheSameCounterTwiceIsIdempotent) {
    // Same ORIGINAL pair - not merely the same canonical name - is the one
    // case that returns the existing binding.
    const auto first = acceptDelta("inference", "tokens");
    const auto again = acceptDelta("inference", "tokens");
    EXPECT_EQ(first.id, again.id);
    EXPECT_EQ(bridge().trackedCount(), 1u);
}

// ── the failed-read contract, all the way to the metric layer ───────────────

TEST_F(NvtxCounterBridgeTest, AnUnavailableSampleDiscardsTheRateWindow) {
    ActiveCounterProvider()->begin_session();
    const auto r = acceptDelta("inference", "tokens");

    const auto parsed = gpufl::detail::parseMetric("custom.inference.tokens_rate");
    ASSERT_EQ(parsed.error, gpufl::detail::MetricParseError::None);
    gpufl::detail::MetricWindowConfig cfg;
    cfg.rate_window_ms = 100;      // bucket = 10ms, 10 buckets
    cfg.stale_after_ms = 20000;
    gpufl::detail::MetricFeeds feeds;
    gpufl::detail::MetricSource src(parsed.id, cfg, &feeds,
                                    ActiveCounterProvider());

    constexpr int64_t kStep = 10 * 1000000;   // one bucket interval
    int64_t t = 0;
    gpufl::detail::MetricSample s;

    // Steady traffic until the window is full and the metric is usable.
    int polls = 0;
    do {
        bridge().sampleDelta(r.id, 5);
        s = src.poll(t);
        t += kStep;
    } while (s.state != gpufl::detail::MetricState::Fresh && ++polls < 40);
    ASSERT_EQ(s.state, gpufl::detail::MetricState::Fresh)
        << "never reached a usable reading; the fixture is wrong";

    // The application fails to read its counter. Real throughput has NOT
    // dropped - only the observation did. Without the discard, the missing
    // deltas read as a rate collapse and a stall rule fires on a workload
    // that never slowed down.
    bridge().sampleNoValue(r.id, kUnavailable);
    s = src.poll(t);
    t += kStep;
    EXPECT_EQ(s.state, gpufl::detail::MetricState::WarmingUp)
        << "a window containing a failed read was presented as evidence";

    // Recovery: the window must refill from post-failure data before the
    // metric is usable again - a new baseline, not a resumed one.
    int refill_polls = 0;
    bool fresh_before_refill = false;
    do {
        bridge().sampleDelta(r.id, 5);
        s = src.poll(t);
        t += kStep;
        ++refill_polls;
        if (s.state == gpufl::detail::MetricState::Fresh && refill_polls < 10) {
            fresh_before_refill = true;
        }
    } while (s.state != gpufl::detail::MetricState::Fresh && refill_polls < 40);
    EXPECT_EQ(s.state, gpufl::detail::MetricState::Fresh)
        << "the metric never recovered after the failed read";
    EXPECT_FALSE(fresh_before_refill)
        << "usable again before a full window of post-failure data";
}

// ── the routing that keeps the evaluator and the target together ────────────

TEST_F(NvtxCounterBridgeTest, TheCounterIsVisibleThroughTheActiveProvider) {
    ActiveCounterProvider()->begin_session();
    const auto r = acceptDelta("inference", "tokens");
    bridge().sampleDelta(r.id, 5);

    // The evaluator reads whatever ActiveCounterProvider() resolves. Writing
    // straight to this module's CounterRegistry instead would split the two
    // apart wherever a shared runtime is present - being in the same injection
    // module as the evaluator is not enough to assume otherwise.
    const std::string name = "inference.tokens";
    gpufl_counter_handle via_provider =
        ActiveCounterProvider()->lookup(name.c_str(), name.size());
    ASSERT_NE(via_provider, nullptr);
    EXPECT_EQ(ActiveCounterProvider()->load_since_baseline(via_provider), 5u);
}

TEST_F(NvtxCounterBridgeTest, TicksBeforeTheSessionStartsAreNotCounted) {
    // Registration and the first samples can legitimately arrive before
    // gpufl::init(): NVTX fires on the application's first NVTX call, which
    // may precede any CUDA call.
    const auto r = acceptDelta("inference", "tokens");
    bridge().sampleDelta(r.id, 100);

    ActiveCounterProvider()->begin_session();
    bridge().sampleDelta(r.id, 7);

    EXPECT_EQ(observed("inference.tokens"), 7u)
        << "pre-session traffic leaked into this session's rate";
}

// ── the session data-quality contract ───────────────────────────────────────

TEST_F(NvtxCounterBridgeTest, EveryRefusedRegistrationIsCountedOnce) {
    bridge().registerCounter("d", "c1", 1u << 24, ValueType::Delta);   // static id
    bridge().registerCounter("d", "c2", 0, ValueType::Absolute);        // value type
    bridge().registerCounter("", "///", 0, ValueType::Delta);           // bad name
    acceptDelta("", "a b");
    bridge().registerCounter("", "a/b", 0, ValueType::Delta);           // collision
    EXPECT_EQ(bridge().registrationRejected(), 4u)
        << "a refusal path forgot to count itself";
    // The accepted one, and its idempotent repeat, are not refusals.
    acceptDelta("", "a b");
    EXPECT_EQ(bridge().registrationRejected(), 4u);
}

TEST_F(NvtxCounterBridgeTest, ASessionSnapshotReportsOnlyItsOwnSession) {
    // The tallies live for the process, like the counter slots; an embedded
    // host re-initialises in one process, and exporting raw totals would
    // re-report session one's problems as session two's.
    bridge().registerCounter("d", "c", 0, ValueType::Absolute);
    const auto r = acceptDelta("inference", "tokens");
    bridge().sampleDelta(r.id, -1);
    bridge().sampleNoValue(r.id, kUnavailable);

    auto first = bridge().takeSessionSnapshot();
    EXPECT_EQ(first.registration_rejected, 1u);
    EXPECT_EQ(first.negative_delta_samples, 1u);
    EXPECT_EQ(first.unavailable_samples, 1u);
    EXPECT_TRUE(first.any());

    // Session two: only what happened after the previous report.
    bridge().sampleDelta(0, 5);   // unknown id
    auto second = bridge().takeSessionSnapshot();
    EXPECT_EQ(second.registration_rejected, 0u)
        << "session one's refusal was reported twice";
    EXPECT_EQ(second.unknown_id_samples, 1u);
    EXPECT_EQ(second.unavailable_samples, 0u);

    // Session three: nothing happened, nothing to say.
    EXPECT_FALSE(bridge().takeSessionSnapshot().any());
}

TEST_F(NvtxCounterBridgeTest, PreInitEventsBelongToTheFirstSession) {
    // NVTX registration legitimately runs before gpufl::init() - proven on
    // hardware - so a refusal during startup happens before any session
    // exists. It belongs to the first session that reports: no other session
    // can, and dropping it hides exactly the config error the event is for.
    bridge().registerCounter("d", "c", 0, ValueType::Unspecified);

    // ... gpufl::init() happens here ...
    const auto snap = bridge().takeSessionSnapshot();
    EXPECT_EQ(snap.registration_rejected, 1u)
        << "a startup config error was attributed to no session at all";
}

// ── the wire shape ──────────────────────────────────────────────────────────
//
// The backend parses these by field name from hand-built JSON; nothing in the
// type system connects the two sides. Pinned here so a renamed field fails a
// test instead of silently landing as an empty column.

TEST(CounterDataQualityWireTest, TheSummaryCarriesEveryTallyByName) {
    gpufl::CounterDataQualitySummaryEvent ev;
    ev.pid = 7;
    ev.app = "serve";
    ev.session_id = "s1";
    ev.tracked_counters = 1;
    ev.samples_observed = 12000;
    ev.registration_rejected = 2;
    ev.unknown_id_samples = 3;
    ev.unavailable_samples = 4;
    ev.negative_delta_samples = 1;
    ev.rate_windows_discarded = 2;
    ev.emitted_ns = 42;

    const std::string json =
        gpufl::model::CounterDataQualitySummaryModel(ev).buildJson();
    EXPECT_NE(json.find("\"type\":\"counter_data_quality_summary\""),
              std::string::npos) << json;
    // Only "nvtx" is observed today; a generic-looking row would claim
    // coverage of gpufl::counter() failures it does not have.
    EXPECT_NE(json.find("\"source\":\"nvtx\""), std::string::npos);
    EXPECT_NE(json.find("\"schema_version\":1"), std::string::npos);
    // The denominators that make an all-zero row readable at all.
    EXPECT_NE(json.find("\"tracked_counters\":1"), std::string::npos);
    EXPECT_NE(json.find("\"samples_observed\":12000"), std::string::npos);
    EXPECT_NE(json.find("\"registration_rejected\":2"), std::string::npos);
    EXPECT_NE(json.find("\"unknown_id_samples\":3"), std::string::npos);
    EXPECT_NE(json.find("\"unavailable_samples\":4"), std::string::npos);
    EXPECT_NE(json.find("\"negative_delta_samples\":1"), std::string::npos);
    EXPECT_NE(json.find("\"rate_windows_discarded\":2"), std::string::npos);
}

TEST(CounterDataQualityWireTest, TheRuleSummaryCarriesItsOwnQualityFields) {
    gpufl::DeepWindowRuleSummaryEvent ev;
    ev.session_id = "s1";
    ev.outcome = "never_true";
    ev.metric_quality_resets = 2;
    ev.last_quality_reason = "counter_unavailable";

    const std::string json =
        gpufl::model::DeepWindowRuleSummaryModel(ev).buildJson();
    EXPECT_NE(json.find("\"metric_quality_resets\":2"), std::string::npos)
        << json;
    EXPECT_NE(json.find("\"last_quality_reason\":\"counter_unavailable\""),
              std::string::npos);
}

TEST_F(NvtxCounterBridgeTest, SamplesObservedCountsValidObservationsOnly) {
    ActiveCounterProvider()->begin_session();
    const auto r = acceptDelta("inference", "tokens");

    bridge().sampleDelta(r.id, 8);
    bridge().sampleDelta(r.id, 0);                 // real observation of "no traffic"
    bridge().sampleNoValue(r.id, kZero);           // ditto
    bridge().sampleNoValue(r.id, kUnchanged);      // ditto
    bridge().sampleNoValue(r.id, kUnavailable);    // a FAILURE, not an observation
    bridge().sampleDelta(r.id, -1);                // ditto
    bridge().sampleDelta(0, 5);                    // unknown id: ditto

    const auto snap = bridge().takeSessionSnapshot();
    // 8 + 0 + zero + unchanged = 4 observations; the three failures are
    // tallied on their own axes, or the failure RATE would be diluted by its
    // own failures.
    EXPECT_EQ(snap.samples_observed, 4u);
    EXPECT_EQ(snap.unavailable_samples, 1u);
    EXPECT_EQ(snap.negative_delta_samples, 1u);
    EXPECT_EQ(snap.unknown_id_samples, 1u);
}

TEST_F(NvtxCounterBridgeTest, SamplesObservedIsSessionScopedToo) {
    // The stale-clean-row problem: a counter registered in session one must
    // not make every later session claim it observed something.
    ActiveCounterProvider()->begin_session();
    const auto r = acceptDelta("inference", "tokens");
    bridge().sampleDelta(r.id, 8);
    EXPECT_EQ(bridge().takeSessionSnapshot().samples_observed, 1u);

    // Session two: the counter is still tracked, but nothing sampled it.
    const auto second = bridge().takeSessionSnapshot();
    EXPECT_EQ(second.samples_observed, 0u)
        << "an idle session inherited the previous session's denominator";
    EXPECT_FALSE(second.any());
}

}  // namespace

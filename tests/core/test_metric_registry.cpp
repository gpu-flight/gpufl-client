#include <gtest/gtest.h>

#include <cstring>
#include <string>

#include "gpufl.hpp"
#include "gpufl/core/counter_provider.hpp"
#include "gpufl/core/counter_registry.hpp"
#include "gpufl/core/events.hpp"
#include "gpufl/core/metric_id.hpp"
#include "gpufl/core/metric_registry.hpp"

using gpufl::detail::ActiveCounterProvider;
using gpufl::detail::ConfigError;
using gpufl::detail::CounterRegistry;
using gpufl::detail::MetricFeeds;
using gpufl::detail::MetricId;
using gpufl::detail::MetricKind;
using gpufl::detail::MetricParseError;
using gpufl::detail::MetricShape;
using gpufl::detail::MetricSource;
using gpufl::detail::MetricState;
using gpufl::detail::MetricWindowConfig;
using gpufl::detail::parseMetric;
using gpufl::detail::validate;

namespace {

constexpr int64_t kMs = 1000000;   // ns per ms

// ------------------------------------------------------------ metric contract

TEST(MetricContractTest, ParsesBuiltinRates) {
    const auto launch = parseMetric("kernel_launch_rate");
    ASSERT_TRUE(launch.ok());
    EXPECT_EQ(launch.id.kind, MetricKind::KernelLaunchRate);
    EXPECT_EQ(launch.id.shape(), MetricShape::Rate);

    const auto kernel = parseMetric("recent_kernel_ms");
    ASSERT_TRUE(kernel.ok());
    EXPECT_EQ(kernel.id.shape(), MetricShape::Percentile);
}

TEST(MetricContractTest, CanonicalisesDeviceIndex) {
    const auto a = parseMetric("gpu[0].util_pct");
    const auto b = parseMetric("gpu[00].util_pct");
    ASSERT_TRUE(a.ok());
    ASSERT_TRUE(b.ok());
    // Two spellings of one rule must not hash to two different rule ids.
    EXPECT_EQ(a.id.canonical, b.id.canonical);
    EXPECT_EQ(a.id.canonical, "gpu[0].util_pct");
}

TEST(MetricContractTest, UnknownBuiltinFieldRejectedAtParse) {
    const auto r = parseMetric("gpu[0].temperature_pct");
    EXPECT_FALSE(r.ok());
    EXPECT_EQ(r.error, MetricParseError::UnknownBuiltinMetric);
}

TEST(MetricContractTest, MisspelledMetricWithoutPrefixRejectedAtParse) {
    // The whole reason `custom.` exists: without a prefix this is
    // indistinguishable from a counter that has not registered yet, and the
    // mistake would only surface once the run was over.
    const auto r = parseMetric("tokne_rate");
    EXPECT_FALSE(r.ok());
    EXPECT_EQ(r.error, MetricParseError::MissingCustomPrefix);
}

TEST(MetricContractTest, CustomMetricParsesAndKeepsCounterName) {
    const auto r = parseMetric("custom.token_rate");
    ASSERT_TRUE(r.ok());
    EXPECT_EQ(r.id.kind, MetricKind::CustomRate);
    EXPECT_EQ(r.id.custom_name, "token");
    EXPECT_TRUE(r.id.resolvesLazily());
}

TEST(MetricContractTest, CustomMetricNeedsTheRateSuffix) {
    EXPECT_EQ(parseMetric("custom.token").error,
              MetricParseError::MalformedCustomMetric);
    EXPECT_EQ(parseMetric("custom._rate").error,
              MetricParseError::MalformedCustomMetric);
}

TEST(MetricContractTest, MalformedDeviceIndexRejected) {
    EXPECT_EQ(parseMetric("gpu[].util_pct").error,
              MetricParseError::MalformedDeviceIndex);
    EXPECT_EQ(parseMetric("gpu[x].util_pct").error,
              MetricParseError::MalformedDeviceIndex);
    EXPECT_EQ(parseMetric("gpu[-1].util_pct").error,
              MetricParseError::MalformedDeviceIndex);
}

TEST(MetricContractTest, EmptyNameRejected) {
    EXPECT_EQ(parseMetric("").error, MetricParseError::Empty);
}

// ------------------------------------------------------------ config validity

TEST(MetricConfigTest, BucketIntervalIsBoundedAndDerivedOnce) {
    EXPECT_EQ((MetricWindowConfig{100, 0, 5000}).bucketIntervalMs(), 10);
    EXPECT_EQ((MetricWindowConfig{1000, 0, 5000}).bucketIntervalMs(), 100);
    EXPECT_EQ((MetricWindowConfig{60000, 0, 90000}).bucketIntervalMs(), 100);
}

TEST(MetricConfigTest, AcceptsAWorkableCombination) {
    EXPECT_EQ(validate(MetricWindowConfig{1000, 2000, 5000}), ConfigError::None);
}

TEST(MetricConfigTest, RejectsStaleShorterThanTheEvidenceItNeeds) {
    // window=4s, sustained=4s, stale=5s produces its first zero at t=4 and goes
    // stale at t=5 - four seconds short of ever firing. Each field on its own
    // looks reasonable, which is exactly why the combined check has to exist.
    const MetricWindowConfig cfg{4000, 4000, 5000};
    EXPECT_EQ(validate(cfg), ConfigError::StaleBeforeEvidence);
    const std::string why = explain(cfg, ConfigError::StaleBeforeEvidence);
    EXPECT_NE(why.find("8100"), std::string::npos) << why;   // 4000+4000+100
}

TEST(MetricConfigTest, RejectsNonPositiveAndOversizedWindows) {
    EXPECT_EQ(validate(MetricWindowConfig{0, 0, 100}),
              ConfigError::RateWindowNotPositive);
    EXPECT_EQ(validate(MetricWindowConfig{MetricWindowConfig::kMaxRateWindowMs + 1,
                                          0, 100000000}),
              ConfigError::RateWindowTooLarge);
    EXPECT_EQ(validate(MetricWindowConfig{1000, -1, 5000}),
              ConfigError::SustainedNegative);
    EXPECT_EQ(validate(MetricWindowConfig{1000, 0, 0}),
              ConfigError::StaleAfterNotPositive);
}

TEST(MetricConfigTest, StaleArithmeticDoesNotOverflow) {
    constexpr int64_t kBig = 9223372036854775000LL;
    const MetricWindowConfig cfg{1000, kBig, kBig};
    // Must decide, not wrap into an accidental "ok".
    EXPECT_EQ(validate(cfg), ConfigError::StaleBeforeEvidence);
}

// ---------------------------------------------------------------- rate source

class MetricSourceTest : public ::testing::Test {
   protected:
    void SetUp() override { CounterRegistry::instance().resetForTesting(); }
    void TearDown() override { CounterRegistry::instance().resetForTesting(); }

    MetricFeeds feeds;

    static MetricId parse(const char* text) {
        const auto r = parseMetric(text);
        EXPECT_TRUE(r.ok()) << text;
        return r.id;
    }

    // Drive the source forward to `now`, polling on every bucket boundary the
    // way the collector loop would.
    static gpufl::detail::MetricSample advance(MetricSource& src, int64_t from_ns,
                                               int64_t to_ns, int64_t step_ns) {
        gpufl::detail::MetricSample last;
        for (int64_t t = from_ns; t <= to_ns; t += step_ns) last = src.poll(t);
        return last;
    }
};

TEST_F(MetricSourceTest, LaunchRateWarmsUpBeforeTheWindowIsFull) {
    const MetricWindowConfig cfg{1000, 2000, 5000};   // 100ms buckets, 10 of them
    MetricSource src(parse("kernel_launch_rate"), cfg, &feeds, ActiveCounterProvider());
    feeds.seedStartup(0);

    // Half a window in: no verdict yet. A rule that fired here would be acting
    // on a partial window, which reads as a lower rate than reality.
    const auto mid = advance(src, 0, 500 * kMs, 10 * kMs);
    EXPECT_EQ(mid.state, MetricState::WarmingUp);
}

TEST_F(MetricSourceTest, LaunchRateWithNoLaunchesIsFreshZeroNotStale) {
    const MetricWindowConfig cfg{1000, 2000, 5000};
    MetricSource src(parse("kernel_launch_rate"), cfg, &feeds, ActiveCounterProvider());
    feeds.seedStartup(0);

    const auto s = advance(src, 0, 1100 * kMs, 10 * kMs);
    // The launch source exists from startup, so "no launches" is a measurement,
    // not an absence. Reporting Stale here would be the opposite verdict.
    EXPECT_EQ(s.state, MetricState::Fresh);
    EXPECT_DOUBLE_EQ(s.value, 0.0);
}

TEST_F(MetricSourceTest, EmptyBucketsStillPublishNewSequences) {
    const MetricWindowConfig cfg{1000, 2000, 5000};
    MetricSource src(parse("kernel_launch_rate"), cfg, &feeds, ActiveCounterProvider());
    feeds.seedStartup(0);

    const auto a = advance(src, 0, 1100 * kMs, 10 * kMs);
    const auto b = advance(src, 1110 * kMs, 1500 * kMs, 10 * kMs);
    // Without this a genuine zero never accumulates as evidence and a total
    // stall reads as "no new data" - the one case the feature exists for.
    EXPECT_GT(b.sequence, a.sequence);
    EXPECT_EQ(b.state, MetricState::Fresh);
}

TEST_F(MetricSourceTest, LaunchRateReflectsObservedLaunches) {
    const MetricWindowConfig cfg{1000, 2000, 5000};
    MetricSource src(parse("kernel_launch_rate"), cfg, &feeds, ActiveCounterProvider());
    feeds.seedStartup(0);

    // 500 launches spread over one window == 500/s.
    for (int64_t t = 0; t < 1000 * kMs; t += 2 * kMs) {
        feeds.noteKernelLaunch(t);
        src.poll(t);
    }
    const auto s = advance(src, 1000 * kMs, 1010 * kMs, 10 * kMs);
    EXPECT_EQ(s.state, MetricState::Fresh);
    EXPECT_NEAR(s.value, 500.0, 60.0) << "rate=" << s.value;
}

TEST_F(MetricSourceTest, PollingBetweenBucketsDoesNotAdvanceTheSequence) {
    const MetricWindowConfig cfg{1000, 2000, 5000};
    MetricSource src(parse("kernel_launch_rate"), cfg, &feeds, ActiveCounterProvider());
    feeds.seedStartup(0);
    const auto warm = advance(src, 0, 1100 * kMs, 10 * kMs);

    const auto a = src.poll(1105 * kMs);
    const auto b = src.poll(1110 * kMs);
    const auto c = src.poll(1115 * kMs);
    // The evaluator runs ~100x faster than a bucket closes. If every poll
    // looked like new evidence, one reading would satisfy any sustained_ms.
    EXPECT_EQ(a.sequence, b.sequence);
    EXPECT_EQ(b.sequence, c.sequence);
    EXPECT_GE(a.sequence, warm.sequence);
}

TEST_F(MetricSourceTest, CustomCounterIsMissingUntilTheAppRegistersIt) {
    const MetricWindowConfig cfg{1000, 2000, 5000};
    MetricSource src(parse("custom.metric_absent_rate"), cfg, &feeds,
                     ActiveCounterProvider());

    const auto before = advance(src, 0, 1100 * kMs, 10 * kMs);
    EXPECT_EQ(before.state, MetricState::Missing);
    EXPECT_FALSE(src.customResolved());

    // Asking about a counter must not create it - that would make "never
    // registered" indistinguishable from "registered but idle".
    //
    // Checked through the ACTIVE provider, which is what the source consults.
    // With a shared runtime present the local registry is a different registry
    // entirely, and asserting against it would pass without proving anything.
    EXPECT_EQ(ActiveCounterProvider()->lookup("metric_absent",
                                              std::strlen("metric_absent")),
              nullptr);
}

TEST_F(MetricSourceTest, CustomCounterResolvesLazilyAfterRegistration) {
    const MetricWindowConfig cfg{1000, 2000, 5000};
    MetricSource src(parse("custom.metric_lazy_rate"), cfg, &feeds,
                     ActiveCounterProvider());
    EXPECT_EQ(src.poll(0).state, MetricState::Missing);

    // An env rule is parsed during init(), long before application code reaches
    // gpufl::counter(). Rejecting it at install time would kill every such rule.
    auto tokens = gpufl::counter("metric_lazy");
    const auto registered = advance(src, 10 * kMs, 1100 * kMs, 10 * kMs);
    EXPECT_EQ(registered.state, MetricState::WarmingUp)
        << "registered but never ticked is not Missing, and not Fresh 0 either";

    for (int64_t t = 1100 * kMs; t < 2200 * kMs; t += 10 * kMs) {
        tokens.add(10);
        src.poll(t);
    }
    const auto s = src.poll(2200 * kMs);
    EXPECT_EQ(s.state, MetricState::Fresh);
    EXPECT_GT(s.value, 0.0);
}

TEST_F(MetricSourceTest, CustomCounterGoesStaleWhenTicksStop) {
    const MetricWindowConfig cfg{1000, 2000, 5000};
    MetricSource src(parse("custom.metric_stall_rate"), cfg, &feeds,
                     ActiveCounterProvider());
    // Unique to this test: slots are permanent by design, so a name shared with
    // another test makes the result depend on execution order.
    auto c = gpufl::counter("metric_stall");

    for (int64_t t = 0; t < 2000 * kMs; t += 10 * kMs) {
        c.add(5);
        src.poll(t);
    }
    ASSERT_EQ(src.poll(2000 * kMs).state, MetricState::Fresh);

    // Zeros first accumulate as fresh evidence...
    const auto zeros = advance(src, 2010 * kMs, 4000 * kMs, 10 * kMs);
    EXPECT_EQ(zeros.state, MetricState::Fresh);
    EXPECT_DOUBLE_EQ(zeros.value, 0.0);

    // ...and only later does the source itself read as dead. Both halves matter:
    // a rule must be able to fire on the stall before staleness hides it.
    const auto dead = advance(src, 4010 * kMs, 9000 * kMs, 10 * kMs);
    EXPECT_EQ(dead.state, MetricState::Stale);
}

TEST_F(MetricSourceTest, PercentilePublishesNothingForAnEmptyWindow) {
    const MetricWindowConfig cfg{1000, 2000, 5000};
    MetricSource src(parse("recent_kernel_ms"), cfg, &feeds, ActiveCounterProvider());
    feeds.seedStartup(0);

    const auto s = advance(src, 0, 1500 * kMs, 10 * kMs);
    // 0 ms would read as instantaneous kernels rather than no kernels, and a
    // rule watching for slow kernels would silently never fire.
    EXPECT_NE(s.state, MetricState::Fresh);
    EXPECT_DOUBLE_EQ(s.value, 0.0);
}

TEST_F(MetricSourceTest, PercentileReportsTheMedianOverTheWindow) {
    const MetricWindowConfig cfg{1000, 2000, 5000};
    MetricSource src(parse("recent_kernel_ms"), cfg, &feeds, ActiveCounterProvider());
    feeds.seedStartup(0);

    for (int64_t t = 0; t < 1200 * kMs; t += 10 * kMs) {
        feeds.noteKernelDuration(t, 4.0);
        src.poll(t);
    }
    const auto s = src.poll(1200 * kMs);
    EXPECT_EQ(s.state, MetricState::Fresh);
    EXPECT_NEAR(s.value, 4.0, 0.001);
}

TEST_F(MetricSourceTest, GaugeAdvancesOnMeasurementsNotPolls) {
    const MetricWindowConfig cfg{1000, 2000, 5000};
    MetricSource src(parse("gpu[0].util_pct"), cfg, &feeds, ActiveCounterProvider());

    EXPECT_EQ(src.poll(0).state, MetricState::WarmingUp);

    gpufl::DeviceSample sample;
    sample.device_id = 0;
    sample.gpu_util = 91;
    feeds.noteDeviceSample(sample, 100 * kMs);

    const auto a = src.poll(110 * kMs);
    EXPECT_EQ(a.state, MetricState::Fresh);
    EXPECT_DOUBLE_EQ(a.value, 91.0);

    // NVML publishes every 100-500ms; the evaluator polls every ~1ms. A
    // sequence that moved on polling would let one measurement satisfy any
    // sustained_ms all by itself.
    const auto b = src.poll(300 * kMs);
    EXPECT_EQ(a.sequence, b.sequence);

    feeds.noteDeviceSample(sample, 400 * kMs);
    EXPECT_GT(src.poll(410 * kMs).sequence, b.sequence);
}

TEST_F(MetricSourceTest, GaugeGoesStaleWhenMeasurementsStop) {
    const MetricWindowConfig cfg{1000, 2000, 5000};
    MetricSource src(parse("gpu[0].util_pct"), cfg, &feeds, ActiveCounterProvider());

    gpufl::DeviceSample sample;
    sample.device_id = 0;
    sample.gpu_util = 50;
    feeds.noteDeviceSample(sample, 0);

    EXPECT_EQ(src.poll(1000 * kMs).state, MetricState::Fresh);
    EXPECT_EQ(src.poll(9000 * kMs).state, MetricState::Stale);
}

TEST_F(MetricSourceTest, MissingDeviceReadsAsWarmingUpNotZero) {
    const MetricWindowConfig cfg{1000, 2000, 5000};
    MetricSource src(parse("gpu[3].util_pct"), cfg, &feeds, ActiveCounterProvider());

    gpufl::DeviceSample sample;
    sample.device_id = 0;
    sample.gpu_util = 77;
    feeds.noteDeviceSample(sample, 0);

    // A rule on a GPU that does not exist must not read as 0% utilisation,
    // which is a perfectly firable value for a "GPU went idle" rule.
    EXPECT_EQ(src.poll(1000 * kMs).state, MetricState::WarmingUp);
}

TEST_F(MetricSourceTest, EpochResetDiscardsEvidenceButKeepsSequenceMonotonic) {
    const MetricWindowConfig cfg{1000, 2000, 5000};
    MetricSource src(parse("kernel_launch_rate"), cfg, &feeds, ActiveCounterProvider());
    feeds.seedStartup(0);

    for (int64_t t = 0; t < 1200 * kMs; t += 10 * kMs) {
        feeds.noteKernelLaunch(t);
        src.poll(t);
    }
    const auto before = src.poll(1200 * kMs);
    ASSERT_EQ(before.state, MetricState::Fresh);

    src.resetEpoch(1200 * kMs);
    const auto after = src.poll(1210 * kMs);
    // Buckets filled while profiling was active describe a contaminated
    // workload; letting them prove recovery is how a rule re-fires on its own
    // overhead.
    EXPECT_EQ(after.state, MetricState::WarmingUp);
    // The sequence must not go backwards, or the evaluator would ignore real
    // samples whose numbers it had already seen.
    EXPECT_GE(after.sequence, before.sequence);
}

TEST_F(MetricSourceTest, ALongCollectorStallDoesNotReplayFabricatedZeros) {
    const MetricWindowConfig cfg{1000, 2000, 5000};
    MetricSource src(parse("kernel_launch_rate"), cfg, &feeds, ActiveCounterProvider());
    feeds.seedStartup(0);

    for (int64_t t = 0; t < 1200 * kMs; t += 10 * kMs) {
        feeds.noteKernelLaunch(t);
        src.poll(t);
    }
    ASSERT_EQ(src.poll(1200 * kMs).state, MetricState::Fresh);

    // The collector was blocked for a minute. Replaying 600 empty buckets would
    // look like a measured run of zeros and could fire a stall rule that
    // nothing actually observed.
    const auto after = src.poll(61200 * kMs);
    EXPECT_EQ(after.state, MetricState::WarmingUp);
}

TEST_F(MetricSourceTest, CustomCounterIgnoresTicksFromAnEarlierSession) {
    // Slots are permanent by design, so a counter ticked by a previous session
    // still holds that value. Reading the raw total would make the new session
    // believe the counter had already moved, arm on evidence it never saw, and
    // fire a stall rule on a workload that had not started.
    auto c = gpufl::counter("metric_prev_session");
    c.add(5000);                       // "previous session" traffic

    ActiveCounterProvider()->begin_session();   // this session's baseline

    const MetricWindowConfig cfg{1000, 2000, 5000};
    MetricSource src(parse("custom.metric_prev_session_rate"), cfg, &feeds,
                     ActiveCounterProvider());

    const auto s = advance(src, 0, 3000 * kMs, 10 * kMs);
    EXPECT_EQ(s.state, MetricState::WarmingUp)
        << "counted a previous session's ticks as this session's first tick";
    ActiveCounterProvider()->end_session();
}

TEST_F(MetricSourceTest, LaunchFeedIsVisibleWithoutTakingALock) {
    // The launch path writes atomics only. This does not prove the absence of a
    // lock, but it does pin the visibility contract the atomics have to keep:
    // seeded is released last, so observing it means count and timestamp are
    // already visible.
    MetricFeeds f;
    f.noteKernelLaunch(1234);
    const auto feed = f.launchFeed();
    EXPECT_TRUE(feed.seeded);
    EXPECT_EQ(feed.count, 1u);
    EXPECT_EQ(feed.last_event_ns, 1234);
}

}  // namespace

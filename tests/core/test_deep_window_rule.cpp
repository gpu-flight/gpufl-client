#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "gpufl.hpp"
#include "gpufl/core/counter_provider.hpp"
#include "gpufl/core/counter_registry.hpp"
#include "gpufl/core/deep_window_rule.hpp"
#include "gpufl/core/metric_registry.hpp"

using namespace gpufl::detail;

namespace {

constexpr int64_t kMs = 1000000;

// A stand-in window coordinator. Lets the state machine be driven without a
// GPU, and - more usefully - lets a test refuse an open on purpose, which is
// the branch that decides whether budget is consumed.
struct FakeCoordinator {
    bool     active = false;
    bool     refuse = false;
    uint64_t next_token = 1;
    uint64_t last_opened_token = 0;
    uint64_t pending_token = 0;
    int      opens = 0;

    /// What the coordinator should answer next. Set per test.
    gpufl::OpenRequestStatus refusal = gpufl::OpenRequestStatus::Cooldown;

    /// A broken coordinator: says yes and hands back nothing to wait on.
    bool accept_without_token = false;

    static gpufl::OpenRequestResult RequestOpen(void* ctx,
                                                const gpufl::DeepWindowSpec&) {
        auto* self = static_cast<FakeCoordinator*>(ctx);
        if (self->accept_without_token) {
            return {0, gpufl::OpenRequestStatus::Accepted};
        }
        if (self->refuse) return {0, self->refusal};
        if (self->active) return {0, gpufl::OpenRequestStatus::Busy};
        self->pending_token = self->next_token++;
        return {self->pending_token, gpufl::OpenRequestStatus::Accepted};
    }
    static bool Active(void* ctx) {
        return static_cast<FakeCoordinator*>(ctx)->active;
    }
    static uint64_t LastOpenedToken(void* ctx) {
        return static_cast<FakeCoordinator*>(ctx)->last_opened_token;
    }
    static uint64_t OpensCompleted(void* ctx) {
        return static_cast<uint64_t>(static_cast<FakeCoordinator*>(ctx)->opens);
    }
    static uint64_t PendingOpenToken(void* ctx) {
        return static_cast<FakeCoordinator*>(ctx)->pending_token;
    }

    RuleEvaluator::Hooks hooks() {
        RuleEvaluator::Hooks h;
        h.request_open = &RequestOpen;
        h.window_active = &Active;
        h.last_opened_token = &LastOpenedToken;
        h.pending_open_token = &PendingOpenToken;
        h.opens_completed = &OpensCompleted;
        h.ctx = this;
        return h;
    }

    /** The coordinator refusing the queued request when it gets to it. */
    void dropRequest() { pending_token = 0; }

    /** The coordinator servicing a queued request. */
    void serviceOpen() {
        if (pending_token == 0) return;
        last_opened_token = pending_token;
        pending_token = 0;
        active = true;
        ++opens;
    }
    /** A window opened by someone else - manual, or the scheduled trigger. */
    void openManual() {
        last_opened_token = 0;
        active = true;
        ++opens;
    }
    void close() { active = false; }
};

DeepWindowRule makeRule(const char* expr) {
    const auto parsed = parseRuleExpression(expr);
    EXPECT_TRUE(parsed.ok()) << expr << " -> " << toString(parsed.error);
    DeepWindowRule r = parsed.rule;
    r.timing.rate_window_ms = 1000;
    r.timing.stale_after_ms = 20000;
    r.window.max_duration_ms = 500;
    r.max_windows = 3;
    return r;
}

// ----------------------------------------------------------------- parsing

TEST(RuleParseTest, ParsesMetricOperatorThresholdAndDuration) {
    const auto r = parseRuleExpression("custom.token_rate<1000 for 2s");
    ASSERT_TRUE(r.ok()) << toString(r.error);
    EXPECT_EQ(r.rule.metric.canonical, "custom.token_rate");
    EXPECT_EQ(r.rule.op, Comparison::LessThan);
    EXPECT_DOUBLE_EQ(r.rule.threshold, 1000.0);
    EXPECT_EQ(r.rule.timing.sustained_ms, 2000);
    // No hysteresis unless asked for: rearm degenerates to "condition false".
    EXPECT_DOUBLE_EQ(r.rule.rearm_threshold, 1000.0);
}

TEST(RuleParseTest, AcceptsMillisecondsAndBareNumbers) {
    EXPECT_EQ(parseRuleExpression("kernel_launch_rate>50 for 500ms").rule
                  .timing.sustained_ms, 500);
    EXPECT_EQ(parseRuleExpression("kernel_launch_rate>50 for 750").rule
                  .timing.sustained_ms, 750);
    EXPECT_EQ(parseRuleExpression("kernel_launch_rate>50").rule
                  .timing.sustained_ms, 0);
}

TEST(RuleParseTest, RejectsGarbage) {
    EXPECT_EQ(parseRuleExpression("kernel_launch_rate 50").error,
              RuleError::Unparsable);
    EXPECT_EQ(parseRuleExpression("kernel_launch_rate>abc").error,
              RuleError::Unparsable);
    EXPECT_EQ(parseRuleExpression("kernel_launch_rate>50 for 2 parsecs").error,
              RuleError::Unparsable);
    // The metric reason must survive, not be flattened into "unparsable".
    const auto bad = parseRuleExpression("tokne_rate<5");
    EXPECT_EQ(bad.error, RuleError::BadMetric);
    EXPECT_EQ(bad.metric_error, MetricParseError::MissingCustomPrefix);
}

// -------------------------------------------------------------- validation

TEST(RuleValidateTest, AcceptsAWorkableRule) {
    EXPECT_EQ(validateRule(makeRule("custom.token_rate<1000 for 2s")).error,
              RuleError::None);
}

TEST(RuleValidateTest, RejectsRearmOnTheWrongSide) {
    DeepWindowRule r = makeRule("custom.token_rate<1000 for 2s");
    r.rearm_threshold = 900;   // below the threshold: can never be reached
    // Otherwise the rule fires once and then waits forever for a condition
    // that cannot occur, which looks identical to a healthy armed rule.
    EXPECT_EQ(validateRule(r).error, RuleError::RearmWrongSide);

    DeepWindowRule g = makeRule("kernel_launch_rate>1000 for 2s");
    g.rearm_threshold = 1100;
    EXPECT_EQ(validateRule(g).error, RuleError::RearmWrongSide);
}

TEST(RuleValidateTest, RejectsNonFiniteThreshold) {
    DeepWindowRule r = makeRule("custom.token_rate<1000 for 2s");
    r.threshold = std::nan("");
    r.rearm_threshold = r.threshold;
    // Every comparison against NaN is false, so the rule would never fire and
    // never report a problem.
    EXPECT_EQ(validateRule(r).error, RuleError::ThresholdNotFinite);
}

TEST(RuleValidateTest, RejectsBudgetOutOfRange) {
    DeepWindowRule r = makeRule("custom.token_rate<1000 for 2s");
    r.max_windows = 0;
    EXPECT_EQ(validateRule(r).error, RuleError::MaxWindowsOutOfRange);
    r.max_windows = 100000;
    EXPECT_EQ(validateRule(r).error, RuleError::MaxWindowsOutOfRange);
}

TEST(RuleValidateTest, RejectsAWindowWithNoBound) {
    DeepWindowRule r = makeRule("custom.token_rate<1000 for 2s");
    r.window.max_duration_ms = 0;
    r.window.max_launches = 0;
    // A window that never closes turns a bounded-cost feature into an
    // always-on one.
    EXPECT_EQ(validateRule(r).error, RuleError::WindowBoundsMissing);
}

TEST(RuleValidateTest, RejectsTimingThatCanNeverFire) {
    DeepWindowRule r = makeRule("custom.token_rate<1000 for 2s");
    r.timing.rate_window_ms = 4000;
    r.timing.sustained_ms = 4000;
    r.timing.stale_after_ms = 5000;
    const auto v = validateRule(r);
    EXPECT_EQ(v.error, RuleError::BadTiming);
    EXPECT_EQ(v.config_error, ConfigError::StaleBeforeEvidence);
    EXPECT_FALSE(v.detail.empty()) << "the arithmetic has to be in the message";
}

// ----------------------------------------------------------- state machine

class RuleEvaluatorTest : public ::testing::Test {
   protected:
    void SetUp() override { CounterRegistry::instance().resetForTesting(); }
    void TearDown() override { CounterRegistry::instance().resetForTesting(); }

    MetricFeeds     feeds;
    FakeCoordinator coord;

    // Drive the collector beat, feeding `launches` launches per bucket.
    void run(RuleEvaluator& ev, MetricSource& src, int64_t from_ns, int64_t to_ns,
             int launches_per_beat) {
        for (int64_t t = from_ns; t <= to_ns; t += 10 * kMs) {
            for (int i = 0; i < launches_per_beat; ++i) feeds.noteKernelLaunch(t);
            ev.poll(t);
        }
    }
};

TEST_F(RuleEvaluatorTest, FiresOnlyAfterTheConditionIsSustained) {
    DeepWindowRule rule = makeRule("kernel_launch_rate<100 for 2s");
    MetricSource src(rule.metric, rule.timing, &feeds, ActiveCounterProvider());
    RuleEvaluator ev(rule, "r1", RuleCapabilities{}, &src, coord.hooks());
    feeds.seedStartup(0);

    // Busy: 10 launches per 10ms beat == 1000/s, above the threshold.
    run(ev, src, 0, 2000 * kMs, 10);
    EXPECT_EQ(ev.state(), RuleState::Armed);

    // Goes quiet. The rule must not fire on the first true reading.
    run(ev, src, 2010 * kMs, 4000 * kMs, 0);
    EXPECT_EQ(ev.state(), RuleState::Pending)
        << "fired before the condition was held for 2s";

    run(ev, src, 4010 * kMs, 6000 * kMs, 0);
    EXPECT_EQ(ev.state(), RuleState::Opening);
    EXPECT_EQ(ev.windowsOpened(), 0u) << "budget consumed before a window opened";
}

TEST_F(RuleEvaluatorTest, BudgetCountsOnlyWindowsThatActuallyOpened) {
    DeepWindowRule rule = makeRule("kernel_launch_rate<100 for 500ms");
    MetricSource src(rule.metric, rule.timing, &feeds, ActiveCounterProvider());
    RuleEvaluator ev(rule, "r1", RuleCapabilities{}, &src, coord.hooks());
    feeds.seedStartup(0);

    run(ev, src, 0, 1500 * kMs, 10);
    run(ev, src, 1510 * kMs, 4000 * kMs, 0);
    ASSERT_EQ(ev.state(), RuleState::Opening);
    EXPECT_EQ(ev.windowsOpened(), 0u);

    coord.serviceOpen();
    ev.poll(4010 * kMs);
    EXPECT_EQ(ev.state(), RuleState::Blackout);
    EXPECT_EQ(ev.windowsOpened(), 1u);
}

TEST_F(RuleEvaluatorTest, ADroppedRequestCostsNoBudget) {
    DeepWindowRule rule = makeRule("kernel_launch_rate<100 for 500ms");
    MetricSource src(rule.metric, rule.timing, &feeds, ActiveCounterProvider());
    RuleEvaluator ev(rule, "r1", RuleCapabilities{}, &src, coord.hooks());
    feeds.seedStartup(0);

    run(ev, src, 0, 1500 * kMs, 10);
    run(ev, src, 1510 * kMs, 4000 * kMs, 0);
    ASSERT_EQ(ev.state(), RuleState::Opening);

    // The coordinator discards the request instead of opening.
    coord.dropRequest();
    ev.poll(4010 * kMs);
    EXPECT_EQ(ev.state(), RuleState::Armed);
    EXPECT_EQ(ev.windowsOpened(), 0u)
        << "budget bounds what the rule cost, and this cost nothing";
}

TEST_F(RuleEvaluatorTest, ARefusedOpenReturnsToArmedRatherThanHoldingPending) {
    DeepWindowRule rule = makeRule("kernel_launch_rate<100 for 500ms");
    MetricSource src(rule.metric, rule.timing, &feeds, ActiveCounterProvider());
    RuleEvaluator ev(rule, "r1", RuleCapabilities{}, &src, coord.hooks());
    feeds.seedStartup(0);
    coord.refuse = true;   // cooldown still running

    run(ev, src, 0, 1500 * kMs, 10);
    run(ev, src, 1510 * kMs, 4000 * kMs, 0);
    EXPECT_EQ(ev.windowsOpened(), 0u);

    // The refusal must discard the evidence, not bank it. Accepting requests
    // again is not enough on its own - the rule has to gather a fresh
    // sustained run first, or a cooldown would just delay a fire that then
    // lands on readings from long before.
    coord.refuse = false;
    ev.poll(4010 * kMs);
    EXPECT_NE(ev.state(), RuleState::Opening)
        << "fired on evidence gathered while the coordinator was refusing";

    run(ev, src, 4020 * kMs, 5000 * kMs, 0);
    EXPECT_EQ(ev.state(), RuleState::Opening)
        << "never recovered after the refusal";
}

TEST_F(RuleEvaluatorTest, AManualWindowBlacksOutButCostsNoBudget) {
    DeepWindowRule rule = makeRule("kernel_launch_rate<100 for 2s");
    MetricSource src(rule.metric, rule.timing, &feeds, ActiveCounterProvider());
    RuleEvaluator ev(rule, "r1", RuleCapabilities{}, &src, coord.hooks());
    feeds.seedStartup(0);

    run(ev, src, 0, 1500 * kMs, 10);
    ASSERT_EQ(ev.state(), RuleState::Armed);

    coord.openManual();
    ev.poll(1510 * kMs);
    // Contamination does not care who opened the window.
    EXPECT_EQ(ev.state(), RuleState::Blackout);
    EXPECT_EQ(ev.windowsOpened(), 0u)
        << "a manual window must not spend the rule's budget";
}

TEST_F(RuleEvaluatorTest, ContaminatedSamplesDoNotProveRecovery) {
    DeepWindowRule rule = makeRule("kernel_launch_rate<100 for 500ms");
    MetricSource src(rule.metric, rule.timing, &feeds, ActiveCounterProvider());
    RuleEvaluator ev(rule, "r1", RuleCapabilities{}, &src, coord.hooks());
    feeds.seedStartup(0);

    run(ev, src, 0, 1500 * kMs, 10);
    run(ev, src, 1510 * kMs, 4000 * kMs, 0);
    coord.serviceOpen();
    ev.poll(4010 * kMs);
    ASSERT_EQ(ev.state(), RuleState::Blackout);

    // Busy traffic while the window is open must not count as recovery.
    run(ev, src, 4020 * kMs, 6000 * kMs, 10);
    EXPECT_EQ(ev.state(), RuleState::Blackout);

    coord.close();
    ev.poll(6010 * kMs);
    EXPECT_EQ(ev.state(), RuleState::Recovery)
        << "the clean epoch only starts once the window has closed";
}

TEST_F(RuleEvaluatorTest, NoRefireUntilTheWorkloadRecovers) {
    DeepWindowRule rule = makeRule("kernel_launch_rate<100 for 500ms");
    MetricSource src(rule.metric, rule.timing, &feeds, ActiveCounterProvider());
    RuleEvaluator ev(rule, "r1", RuleCapabilities{}, &src, coord.hooks());
    feeds.seedStartup(0);

    run(ev, src, 0, 1500 * kMs, 10);
    run(ev, src, 1510 * kMs, 4000 * kMs, 0);
    coord.serviceOpen();
    ev.poll(4010 * kMs);
    coord.close();
    ev.poll(4020 * kMs);

    // Still quiet - the condition is still true. It must NOT reopen.
    run(ev, src, 4030 * kMs, 9000 * kMs, 0);
    EXPECT_EQ(ev.windowsOpened(), 1u)
        << "refired while the condition never stopped holding";
    EXPECT_NE(ev.state(), RuleState::Blackout);

    // Recover, then degrade again: the second window is legitimate.
    run(ev, src, 9010 * kMs, 11000 * kMs, 10);
    run(ev, src, 11010 * kMs, 14000 * kMs, 0);
    coord.serviceOpen();
    ev.poll(14010 * kMs);
    EXPECT_EQ(ev.windowsOpened(), 2u);
}

TEST_F(RuleEvaluatorTest, ExhaustionIsMarkedAtTheOpenThatReachesTheLimit) {
    DeepWindowRule rule = makeRule("kernel_launch_rate<100 for 500ms");
    rule.max_windows = 1;
    MetricSource src(rule.metric, rule.timing, &feeds, ActiveCounterProvider());
    RuleEvaluator ev(rule, "r1", RuleCapabilities{}, &src, coord.hooks());
    feeds.seedStartup(0);

    run(ev, src, 0, 1500 * kMs, 10);
    run(ev, src, 1510 * kMs, 4000 * kMs, 0);
    coord.serviceOpen();
    ev.poll(4010 * kMs);
    ASSERT_EQ(ev.windowsOpened(), 1u);

    // Marked at the transition, not at shutdown, so a crashed run still
    // explains itself.
    EXPECT_EQ(ev.snapshot(4020 * kMs).outcome, RuleOutcome::Exhausted);

    coord.close();
    ev.poll(4030 * kMs);
    // The evaluator has nowhere left to stand, so it goes inactive rather than
    // back into recovery.
    EXPECT_EQ(ev.state(), RuleState::Inactive);

    run(ev, src, 4040 * kMs, 10000 * kMs, 0);
    EXPECT_EQ(ev.windowsOpened(), 1u);
}

TEST_F(RuleEvaluatorTest, BeginRunPartRefreshesASpentBudget) {
    DeepWindowRule rule = makeRule("kernel_launch_rate<100 for 500ms");
    rule.max_windows = 1;
    MetricSource src(rule.metric, rule.timing, &feeds, ActiveCounterProvider());
    RuleEvaluator ev(rule, "r1", RuleCapabilities{}, &src, coord.hooks());
    feeds.seedStartup(0);

    // Part 1: spend the single-window budget, ending Inactive/Exhausted.
    run(ev, src, 0, 1500 * kMs, 10);
    run(ev, src, 1510 * kMs, 4000 * kMs, 0);
    coord.serviceOpen();
    ev.poll(4010 * kMs);
    ASSERT_EQ(ev.windowsOpened(), 1u);
    coord.close();
    ev.poll(4030 * kMs);
    ASSERT_EQ(ev.state(), RuleState::Inactive);

    // The roll: a fresh part gets its own budget and re-arms.
    ev.beginRunPart();
    EXPECT_EQ(ev.windowsOpened(), 0u) << "budget carried across the roll";
    EXPECT_EQ(ev.state(), RuleState::Armed);
    EXPECT_EQ(ev.snapshot(4040 * kMs).outcome, RuleOutcome::None)
        << "the new part starts with no terminal outcome";

    // Part 2: recover to healthy, then degrade again. The window opens, which
    // it could not do if the spent budget had carried over.
    run(ev, src, 5000 * kMs, 7000 * kMs, 10);
    run(ev, src, 7010 * kMs, 10000 * kMs, 0);
    coord.serviceOpen();
    ev.poll(10010 * kMs);
    EXPECT_EQ(ev.windowsOpened(), 1u)
        << "the new part could not open a window with its reset budget";
}

TEST_F(RuleEvaluatorTest, BeginRunPartLeavesAnUnsupportedRuleDead) {
    DeepWindowRule rule = makeRule("kernel_launch_rate<100 for 500ms");
    MetricSource src(rule.metric, rule.timing, &feeds, ActiveCounterProvider());
    RuleCapabilities caps;
    caps.deep_engine_prepared = false;
    RuleEvaluator ev(rule, "r1", caps, &src, coord.hooks());
    feeds.seedStartup(0);

    run(ev, src, 0, 4000 * kMs, 0);
    ASSERT_EQ(ev.state(), RuleState::Inactive);

    // A roll cannot revive a rule that can never be supported.
    ev.beginRunPart();
    EXPECT_EQ(ev.state(), RuleState::Inactive)
        << "an unsupported rule was wrongly re-armed on a roll";
    EXPECT_EQ(ev.snapshot(4010 * kMs).outcome, RuleOutcome::Unsupported);
}

TEST_F(RuleEvaluatorTest, BeginRunPartResetsBudgetWithoutDisturbingALiveRule) {
    DeepWindowRule rule = makeRule("kernel_launch_rate<100 for 500ms");
    rule.max_windows = 3;  // plenty of budget; the rule never exhausts
    MetricSource src(rule.metric, rule.timing, &feeds, ActiveCounterProvider());
    RuleEvaluator ev(rule, "r1", RuleCapabilities{}, &src, coord.hooks());
    feeds.seedStartup(0);

    // Open one window, then let it close: the rule is live in Recovery, not
    // exhausted.
    run(ev, src, 0, 1500 * kMs, 10);
    run(ev, src, 1510 * kMs, 4000 * kMs, 0);
    coord.serviceOpen();
    ev.poll(4010 * kMs);
    ASSERT_EQ(ev.windowsOpened(), 1u);
    coord.close();
    ev.poll(4020 * kMs);
    const RuleState live_state = ev.state();
    ASSERT_NE(live_state, RuleState::Inactive);

    // The roll resets the budget but must not re-arm or otherwise disturb a
    // rule that had not spent it.
    ev.beginRunPart();
    EXPECT_EQ(ev.windowsOpened(), 0u);
    EXPECT_EQ(ev.state(), live_state) << "a live rule's state was disturbed";
    EXPECT_EQ(ev.snapshot(4030 * kMs).outcome, RuleOutcome::None);
}

TEST_F(RuleEvaluatorTest, HysteresisNeedsARealRecoveryNotABrushPastTheThreshold) {
    DeepWindowRule rule = makeRule("kernel_launch_rate<500 for 500ms");
    rule.rearm_threshold = 900;   // must climb back above 900 to rearm
    MetricSource src(rule.metric, rule.timing, &feeds, ActiveCounterProvider());
    RuleEvaluator ev(rule, "r1", RuleCapabilities{}, &src, coord.hooks());
    feeds.seedStartup(0);

    run(ev, src, 0, 1500 * kMs, 10);
    run(ev, src, 1510 * kMs, 4000 * kMs, 0);
    coord.serviceOpen();
    ev.poll(4010 * kMs);
    coord.close();
    ev.poll(4020 * kMs);

    // ~600/s: above the trigger threshold but below the rearm threshold.
    run(ev, src, 4030 * kMs, 7000 * kMs, 6);
    EXPECT_EQ(ev.state(), RuleState::WaitingForRearm)
        << "rearmed on a value that had not actually recovered";

    run(ev, src, 7010 * kMs, 10000 * kMs, 10);
    EXPECT_EQ(ev.state(), RuleState::Armed);
}

TEST_F(RuleEvaluatorTest, StaleDataBreaksTheSustainedRun) {
    // A gauge, not a rate. For a rate metric the validated timing GUARANTEES a
    // stall fires before it goes stale, so a rate could never demonstrate this.
    // A gauge stops for its own reasons - NVML no longer answering - while the
    // measured quantity is still low, and that is the case worth guarding.
    DeepWindowRule rule = makeRule("gpu[0].util_pct<10 for 2s");
    rule.timing.stale_after_ms = 3100;
    MetricSource src(rule.metric, rule.timing, &feeds, ActiveCounterProvider());
    RuleEvaluator ev(rule, "r1", RuleCapabilities{}, &src, coord.hooks());

    gpufl::DeviceSample idle;
    idle.device_id = 0;
    idle.gpu_util = 2;
    for (int64_t t = 0; t <= 1000 * kMs; t += 100 * kMs) {
        feeds.noteDeviceSample(idle, t);
        ev.poll(t);
    }
    ASSERT_EQ(ev.state(), RuleState::Pending);

    // Measurements stop. Stale is not evidence the condition continued, so the
    // run is broken rather than completed by readings nobody took.
    for (int64_t t = 1010 * kMs; t <= 9000 * kMs; t += 10 * kMs) ev.poll(t);
    EXPECT_EQ(ev.state(), RuleState::Armed);
    EXPECT_EQ(ev.windowsOpened(), 0u);
}

// ----------------------------------------------------------------- gates

TEST_F(RuleEvaluatorTest, RuleIsRefusedWhenNoDeepEngineIsPrepared) {
    DeepWindowRule rule = makeRule("kernel_launch_rate<100 for 500ms");
    MetricSource src(rule.metric, rule.timing, &feeds, ActiveCounterProvider());
    RuleCapabilities caps;
    caps.deep_engine_prepared = false;
    RuleEvaluator ev(rule, "r1", caps, &src, coord.hooks());
    feeds.seedStartup(0);

    run(ev, src, 0, 4000 * kMs, 0);
    EXPECT_EQ(ev.state(), RuleState::Inactive);
    EXPECT_EQ(ev.windowsOpened(), 0u)
        << "burned budget opening windows that would arm nothing";

    const RuleSummary s = ev.finish(4000 * kMs);
    EXPECT_EQ(s.outcome, RuleOutcome::Unsupported);
    EXPECT_EQ(s.reason, "no_deep_engine");
}

TEST_F(RuleEvaluatorTest, CustomCounterRuleIsRefusedWhenCountersAreNotShared) {
    DeepWindowRule rule = makeRule("custom.token_rate<100 for 500ms");
    MetricSource src(rule.metric, rule.timing, &feeds, ActiveCounterProvider());
    RuleCapabilities caps;
    caps.multi_module = true;      // injection: target and evaluator differ
    caps.counters_shared = false;
    RuleEvaluator ev(rule, "r1", caps, &src, coord.hooks());

    // The target would tick one registry and this evaluator read another, so
    // the counter can never be seen. Saying so beats reporting a counter that
    // is being ticked as Missing for the whole run.
    EXPECT_EQ(ev.state(), RuleState::Inactive);
    EXPECT_EQ(ev.finish(1000 * kMs).reason, "counters_not_shared");
}

TEST_F(RuleEvaluatorTest, UnsharedCountersAreFineInAnEmbeddedHost) {
    DeepWindowRule rule = makeRule("custom.token_rate<100 for 500ms");
    MetricSource src(rule.metric, rule.timing, &feeds, ActiveCounterProvider());
    RuleCapabilities caps;
    caps.multi_module = false;   // one copy of gpufl in the process
    caps.counters_shared = false;
    RuleEvaluator ev(rule, "r1", caps, &src, coord.hooks());
    EXPECT_NE(ev.state(), RuleState::Inactive);
}

TEST_F(RuleEvaluatorTest, GaugeRuleOnAMissingDeviceIsRefusedEagerly) {
    DeepWindowRule rule = makeRule("gpu[3].util_pct<10 for 500ms");
    MetricSource src(rule.metric, rule.timing, &feeds, ActiveCounterProvider());
    RuleCapabilities caps;
    caps.device_count = 1;
    RuleEvaluator ev(rule, "r1", caps, &src, coord.hooks());
    // A built-in metric's availability cannot change later, unlike a custom
    // counter's, so it is decided now.
    EXPECT_EQ(ev.state(), RuleState::Inactive);
    EXPECT_EQ(ev.finish(0).reason, "metric_unavailable");
}

// --------------------------------------------------------------- summaries

TEST_F(RuleEvaluatorTest, ARuleThatNeverMatchedStillReportsAnOutcome) {
    DeepWindowRule rule = makeRule("kernel_launch_rate<100 for 500ms");
    MetricSource src(rule.metric, rule.timing, &feeds, ActiveCounterProvider());
    RuleEvaluator ev(rule, "r1", RuleCapabilities{}, &src, coord.hooks());
    feeds.seedStartup(0);

    run(ev, src, 0, 4000 * kMs, 10);   // always busy
    const RuleSummary s = ev.finish(4000 * kMs);
    // "log once" is invisible in the UI, and absence of a record cannot be
    // read as evidence the rule never fired.
    EXPECT_EQ(s.outcome, RuleOutcome::NeverTrue);
    EXPECT_EQ(s.reason, "condition_never_held");
    EXPECT_GT(s.samples_seen, 0u);
    EXPECT_TRUE(s.last_value.has_value());
}

TEST_F(RuleEvaluatorTest, StateAndOutcomeAreSeparateFields) {
    DeepWindowRule rule = makeRule("kernel_launch_rate<100 for 500ms");
    MetricSource src(rule.metric, rule.timing, &feeds, ActiveCounterProvider());
    RuleEvaluator ev(rule, "r1", RuleCapabilities{}, &src, coord.hooks());
    feeds.seedStartup(0);

    run(ev, src, 0, 1500 * kMs, 10);
    const RuleSummary mid = ev.snapshot(1500 * kMs);
    // `armed` is where the evaluator is standing, not a verdict on the session.
    EXPECT_EQ(mid.state, RuleState::Armed);
    EXPECT_EQ(mid.outcome, RuleOutcome::None);
}

TEST_F(RuleEvaluatorTest, ANeverRegisteredCounterIsNamedInTheSummary) {
    DeepWindowRule rule = makeRule("custom.never_ticked_rate<100 for 500ms");
    MetricSource src(rule.metric, rule.timing, &feeds, ActiveCounterProvider());
    RuleEvaluator ev(rule, "r1", RuleCapabilities{}, &src, coord.hooks());

    for (int64_t t = 0; t <= 4000 * kMs; t += 10 * kMs) ev.poll(t);
    const RuleSummary s = ev.finish(4000 * kMs);
    EXPECT_EQ(s.outcome, RuleOutcome::NeverTrue);
    EXPECT_EQ(s.reason, "custom_metric_never_registered");
    EXPECT_EQ(s.last_metric_state, MetricState::Missing);
}

TEST_F(RuleEvaluatorTest, StateSequenceIsMonotonic) {
    DeepWindowRule rule = makeRule("kernel_launch_rate<100 for 500ms");
    MetricSource src(rule.metric, rule.timing, &feeds, ActiveCounterProvider());
    RuleEvaluator ev(rule, "r1", RuleCapabilities{}, &src, coord.hooks());
    feeds.seedStartup(0);

    uint64_t previous = 0;
    for (int64_t t = 0; t <= 6000 * kMs; t += 10 * kMs) {
        if (t < 1500 * kMs) feeds.noteKernelLaunch(t);
        ev.poll(t);
        const uint64_t seq = ev.snapshot(t).state_sequence;
        // A late record must never overwrite a newer one at the backend, and
        // that ordering is only as good as this number.
        EXPECT_GE(seq, previous);
        previous = seq;
    }
    EXPECT_GT(ev.finish(6000 * kMs).state_sequence, previous);
}

TEST(RuleRefusedTest, AnInvalidRuleStillProducesAReportableSummary) {
    // Configuration is parsed during init(); failing hard there would leave no
    // session and no telemetry writer - nowhere to record this very outcome.
    const RuleSummary s = RuleEvaluator::refused(
        "r1", RuleOutcome::InvalidConfig, "rearm_wrong_side", 42);
    EXPECT_EQ(s.state, RuleState::Inactive);
    EXPECT_EQ(s.outcome, RuleOutcome::InvalidConfig);
    EXPECT_EQ(s.reason, "rearm_wrong_side");
    EXPECT_GT(s.state_sequence, 0u);
}

TEST_F(RuleEvaluatorTest, AWindowThatOpensAndClosesBetweenBeatsIsNotMissed) {
    DeepWindowRule rule = makeRule("kernel_launch_rate<100 for 500ms");
    MetricSource src(rule.metric, rule.timing, &feeds, ActiveCounterProvider());
    RuleEvaluator ev(rule, "r1", RuleCapabilities{}, &src, coord.hooks());
    feeds.seedStartup(0);

    run(ev, src, 0, 1500 * kMs, 10);
    ASSERT_EQ(ev.state(), RuleState::Armed);

    // A launch-bounded window over a busy loop can open and close inside one
    // collector beat. Polling a boolean would see nothing and let the samples
    // taken during it count as clean.
    coord.openManual();
    coord.close();

    ev.poll(1510 * kMs);
    EXPECT_EQ(ev.state(), RuleState::Recovery)
        << "a whole window came and went unnoticed";
}


TEST_F(RuleEvaluatorTest, ATerminalOutcomeIsReportableOnceAtTheTransition) {
    DeepWindowRule rule = makeRule("kernel_launch_rate<100 for 500ms");
    rule.max_windows = 1;
    MetricSource src(rule.metric, rule.timing, &feeds, ActiveCounterProvider());
    RuleEvaluator ev(rule, "r1", RuleCapabilities{}, &src, coord.hooks());
    feeds.seedStartup(0);

    EXPECT_FALSE(ev.takeTerminalToEmit()) << "nothing terminal has happened yet";

    run(ev, src, 0, 1500 * kMs, 10);
    run(ev, src, 1510 * kMs, 4000 * kMs, 0);
    coord.serviceOpen();
    ev.poll(4010 * kMs);
    ASSERT_EQ(ev.snapshot(4010 * kMs).outcome, RuleOutcome::Exhausted);

    // Available at the transition, so a run that crashes afterwards still
    // explains itself instead of looking like one whose rule never fired.
    EXPECT_TRUE(ev.takeTerminalToEmit());
    // Once only: the shutdown summary follows with a higher sequence, and two
    // rows claiming the same thing help nobody.
    EXPECT_FALSE(ev.takeTerminalToEmit());
}

TEST_F(RuleEvaluatorTest, AnUnsupportedRuleIsReportableImmediately) {
    DeepWindowRule rule = makeRule("kernel_launch_rate<100 for 500ms");
    MetricSource src(rule.metric, rule.timing, &feeds, ActiveCounterProvider());
    RuleCapabilities caps;
    caps.deep_engine_prepared = false;
    RuleEvaluator ev(rule, "r1", caps, &src, coord.hooks());

    // Decided at construction, so there is nothing to wait for.
    EXPECT_TRUE(ev.takeTerminalToEmit());
    EXPECT_FALSE(ev.takeTerminalToEmit());
}

TEST_F(RuleEvaluatorTest, TheShutdownSummaryOutranksTheTransitionOne) {
    DeepWindowRule rule = makeRule("kernel_launch_rate<100 for 500ms");
    rule.max_windows = 1;
    MetricSource src(rule.metric, rule.timing, &feeds, ActiveCounterProvider());
    RuleEvaluator ev(rule, "r1", RuleCapabilities{}, &src, coord.hooks());
    feeds.seedStartup(0);

    run(ev, src, 0, 1500 * kMs, 10);
    run(ev, src, 1510 * kMs, 4000 * kMs, 0);
    coord.serviceOpen();
    ev.poll(4010 * kMs);
    const uint64_t at_transition = ev.snapshot(4010 * kMs).state_sequence;

    // The backend upsert only accepts a strictly greater sequence, so the final
    // row must outrank the early one or it would be silently discarded.
    EXPECT_GT(ev.finish(5000 * kMs).state_sequence, at_transition);
}


TEST_F(RuleEvaluatorTest, TheSummaryReportsHowMuchDataWasDiscarded) {
    // A conclusion drawn from a partial percentile must not read like one
    // drawn from all of it, and the summary is where that conclusion lands.
    DeepWindowRule rule = makeRule("recent_kernel_ms>1000 for 500ms");
    MetricSource src(rule.metric, rule.timing, &feeds, ActiveCounterProvider());
    RuleEvaluator ev(rule, "r1", RuleCapabilities{}, &src, coord.hooks());
    feeds.seedStartup(0);

    for (size_t i = 0; i < MetricFeeds::kMaxPendingDurations + 500; ++i) {
        feeds.noteKernelDuration(10 * kMs, 2.0);
    }
    for (int64_t t = 0; t <= 2000 * kMs; t += 10 * kMs) ev.poll(t);

    EXPECT_GT(ev.finish(2000 * kMs).truncated_samples, 0u)
        << "the session concluded from a subset and never said so";
}

// ── a refusal has to say WHY ────────────────────────────────────────────────
//
// One undifferentiated "refused" made every refusal look temporary, so a rule
// whose deep engine had permanently failed retried until shutdown and then
// reported `never_true` - which says the condition was never met, the exact
// opposite of what happened.

TEST_F(RuleEvaluatorTest, AFailedEnginePreparationEndsTheRuleAsUnsupported) {
    DeepWindowRule rule = makeRule("kernel_launch_rate<100 for 500ms");
    MetricSource src(rule.metric, rule.timing, &feeds, ActiveCounterProvider());
    RuleEvaluator ev(rule, "r1", RuleCapabilities{}, &src, coord.hooks());
    feeds.seedStartup(0);
    coord.refuse = true;
    coord.refusal = gpufl::OpenRequestStatus::EngineUnavailable;

    run(ev, src, 0, 1500 * kMs, 10);
    run(ev, src, 1510 * kMs, 4000 * kMs, 0);

    EXPECT_EQ(ev.state(), RuleState::Inactive);
    const RuleSummary s = ev.finish(4000 * kMs);
    EXPECT_EQ(s.outcome, RuleOutcome::Unsupported)
        << "a condition that held but could not be acted on is not never_true";
    EXPECT_EQ(s.reason, "deep_engine_not_prepared");
    EXPECT_EQ(s.windows_opened, 0u);
}

TEST_F(RuleEvaluatorTest, PreparationPendingIsRetriedRatherThanConcluded) {
    // Windows injection installs a rule before CONTEXT_CREATED, so "not ready
    // yet" is an ordinary startup state. Treating it as failure would kill
    // every rule on that platform.
    DeepWindowRule rule = makeRule("kernel_launch_rate<100 for 500ms");
    MetricSource src(rule.metric, rule.timing, &feeds, ActiveCounterProvider());
    RuleEvaluator ev(rule, "r1", RuleCapabilities{}, &src, coord.hooks());
    feeds.seedStartup(0);
    coord.refuse = true;
    coord.refusal = gpufl::OpenRequestStatus::PreparationPending;

    run(ev, src, 0, 1500 * kMs, 10);
    run(ev, src, 1510 * kMs, 4000 * kMs, 0);
    ASSERT_NE(ev.state(), RuleState::Inactive) << "gave up on a pending engine";

    // Preparation completes; the rule must still be able to fire.
    coord.refuse = false;
    run(ev, src, 4010 * kMs, 5200 * kMs, 0);
    EXPECT_EQ(ev.state(), RuleState::Opening);
}

TEST_F(RuleEvaluatorTest, ACooldownRefusalIsNotAPermanentFailure) {
    DeepWindowRule rule = makeRule("kernel_launch_rate<100 for 500ms");
    MetricSource src(rule.metric, rule.timing, &feeds, ActiveCounterProvider());
    RuleEvaluator ev(rule, "r1", RuleCapabilities{}, &src, coord.hooks());
    feeds.seedStartup(0);
    coord.refuse = true;
    coord.refusal = gpufl::OpenRequestStatus::Cooldown;

    run(ev, src, 0, 1500 * kMs, 10);
    run(ev, src, 1510 * kMs, 4000 * kMs, 0);

    EXPECT_NE(ev.state(), RuleState::Inactive);
    // Recording an ordinary quiet period as a permanent failure would tell the
    // user their engine is broken when the rule is simply waiting its turn.
    EXPECT_NE(ev.snapshot(4000 * kMs).outcome, RuleOutcome::Unsupported);
}

TEST_F(RuleEvaluatorTest, ABusyRefusalStillFiresOnceTheWindowIsFree) {
    // The scheduled --deep-after window holds the queue; the rule waits.
    DeepWindowRule rule = makeRule("kernel_launch_rate<100 for 500ms");
    MetricSource src(rule.metric, rule.timing, &feeds, ActiveCounterProvider());
    RuleEvaluator ev(rule, "r1", RuleCapabilities{}, &src, coord.hooks());
    feeds.seedStartup(0);
    coord.refuse = true;
    coord.refusal = gpufl::OpenRequestStatus::Busy;

    run(ev, src, 0, 1500 * kMs, 10);
    run(ev, src, 1510 * kMs, 4000 * kMs, 0);
    ASSERT_NE(ev.state(), RuleState::Inactive);

    coord.refuse = false;
    run(ev, src, 4010 * kMs, 5200 * kMs, 0);
    EXPECT_EQ(ev.state(), RuleState::Opening);

    coord.serviceOpen();
    ev.poll(5210 * kMs);
    EXPECT_EQ(ev.windowsOpened(), 1u);
}

// ── refused to the end is not "never true" ──────────────────────────────────

TEST_F(RuleEvaluatorTest, ARunRefusedToTheEndReportsBlockedNotNeverTrue) {
    // The refusal is temporary and correctly retried, but the run ends before
    // it lifts - a manual window held for the rest of the session, or a
    // cooldown longer than the time left. The condition held throughout.
    DeepWindowRule rule = makeRule("kernel_launch_rate<100 for 500ms");
    MetricSource src(rule.metric, rule.timing, &feeds, ActiveCounterProvider());
    RuleEvaluator ev(rule, "r1", RuleCapabilities{}, &src, coord.hooks());
    feeds.seedStartup(0);
    coord.refuse = true;
    coord.refusal = gpufl::OpenRequestStatus::Cooldown;

    run(ev, src, 0, 1500 * kMs, 10);
    run(ev, src, 1510 * kMs, 4000 * kMs, 0);

    const RuleSummary s = ev.finish(4000 * kMs);
    EXPECT_EQ(s.outcome, RuleOutcome::Blocked)
        << "never_true sends the user to their threshold; the threshold was fine";
    EXPECT_EQ(s.reason, "cooldown") << "and it has to name what was in the way";
    EXPECT_EQ(s.windows_opened, 0u);
}

TEST_F(RuleEvaluatorTest, AConditionThatNeverHeldIsStillNeverTrue) {
    // The other side of the same line: nothing was ever asked for, so
    // `blocked` would be inventing an obstacle that did not exist.
    DeepWindowRule rule = makeRule("kernel_launch_rate>10000 for 500ms");
    MetricSource src(rule.metric, rule.timing, &feeds, ActiveCounterProvider());
    RuleEvaluator ev(rule, "r1", RuleCapabilities{}, &src, coord.hooks());
    feeds.seedStartup(0);

    run(ev, src, 0, 4000 * kMs, 1);

    const RuleSummary s = ev.finish(4000 * kMs);
    EXPECT_EQ(s.outcome, RuleOutcome::NeverTrue);
    EXPECT_EQ(s.reason, "condition_never_held");
}

TEST_F(RuleEvaluatorTest, AnAcceptedRequestWithNoTokenIsAFaultNotAWait) {
    // A coordinator that answers "accepted, token 0" has a bug. Waiting on it
    // parks the evaluator in Opening for the rest of the run, watching a
    // request that was never queued - no window, no error, no explanation.
    DeepWindowRule rule = makeRule("kernel_launch_rate<100 for 500ms");
    MetricSource src(rule.metric, rule.timing, &feeds, ActiveCounterProvider());
    RuleEvaluator ev(rule, "r1", RuleCapabilities{}, &src, coord.hooks());
    feeds.seedStartup(0);
    coord.accept_without_token = true;

    run(ev, src, 0, 1500 * kMs, 10);
    run(ev, src, 1510 * kMs, 4000 * kMs, 0);

    EXPECT_NE(ev.state(), RuleState::Opening) << "stalled on a phantom request";
    const RuleSummary s = ev.finish(4000 * kMs);
    EXPECT_EQ(s.outcome, RuleOutcome::Unsupported);
    EXPECT_EQ(s.reason, "invalid_open_result");
}

TEST_F(RuleEvaluatorTest, ADefaultConstructedResultIsARefusal) {
    // Nobody writes {} on purpose; a hook that forgets a return path produces
    // it. Defaulting the status to Accepted made that silently mean "yes".
    const gpufl::OpenRequestResult unset;
    EXPECT_FALSE(unset.accepted());
    EXPECT_NE(unset.status, gpufl::OpenRequestStatus::Accepted);
}

}  // namespace

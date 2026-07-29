#include "gpufl/core/deep_window_rules.hpp"

#include <atomic>
#include <cerrno>
#include <limits>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <sstream>

#include "gpufl/core/common.hpp"
#include "gpufl/core/counter_provider.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/deep_window_rule.hpp"
#include "gpufl/core/env_vars.hpp"
#include "gpufl/core/events.hpp"
#include "gpufl/core/logger/logger.hpp"
#include "gpufl/core/metric_registry.hpp"
#include "gpufl/core/nvtx_counters.hpp"
#include "gpufl/core/model/deep_window_model.hpp"
#include "gpufl.hpp"
#include "gpufl/core/monitor.hpp"
#include "gpufl/core/monitor_backend.hpp"
#include "gpufl/core/runtime.hpp"

namespace gpufl::detail {
namespace {

std::mutex g_mu;
bool     g_installed = false;
bool     g_finished  = false;
std::string g_expression;
std::string g_rule_id;

/**
 * Process-lifetime feeds, never destroyed.
 *
 * The launch callback writes here without taking any lock, so the object must
 * not be able to disappear underneath it. Same reasoning as the counter slots:
 * a lifetime that ends is a race the hot path would have to pay to defend
 * against. Contents are cleared on install instead.
 */
MetricFeeds& Feeds() {
    static MetricFeeds feeds;
    return feeds;
}

std::unique_ptr<MetricSource>  g_source;
std::unique_ptr<RuleEvaluator> g_eval;
// Set when the rule was refused before it could run, so Finish() still has
// something to report. Holding the summary rather than an error string keeps
// the refused and the ran paths on one shape.
std::unique_ptr<RuleSummary>   g_refused;
bool g_refused_emitted = false;

// Read by the launch callback on every launch, so it must not take the lock.
std::atomic<bool> g_wants_launch_feed{false};
std::atomic<bool> g_wants_duration_feed{false};
// Mirrors g_installed without the mutex, for the feed entry points.
std::atomic<bool> g_installed_relaxed{false};

const char* EnvOrNull(const char* name) {
    const char* v = std::getenv(name);
    return (v && v[0] != '\0') ? v : nullptr;
}

/**
 * Integer from env, or the fallback when unset.
 *
 * A malformed value is a HARD failure, reported through @p bad rather than
 * quietly replaced by the default. Substituting silently means a typo in a
 * threshold or a budget still opens real windows, under settings the user never
 * asked for and cannot see.
 */
int64_t EnvIntOr(const char* name, const int64_t fallback, std::string* bad) {
    const char* v = EnvOrNull(name);
    if (!v) return fallback;
    char* end = nullptr;
    errno = 0;
    const long long n = std::strtoll(v, &end, 10);
    // ERANGE too: strtoll saturates at LLONG_MAX, and a saturated value then
    // overflows the derived-default arithmetic below rather than being caught.
    if (end == v || *end != 0 || errno == ERANGE) {
        if (bad->empty()) *bad = std::string(name) + "='" + v + "' is not an integer";
        return fallback;
    }
    return n;
}

bool EnvDoubleOr(const char* name, double* out, std::string* bad) {
    const char* v = EnvOrNull(name);
    if (!v) return false;
    char* end = nullptr;
    errno = 0;
    const double d = std::strtod(v, &end);
    if (end == v || *end != 0 || errno == ERANGE) {
        if (bad->empty()) *bad = std::string(name) + "='" + v + "' is not a number";
        return false;
    }
    *out = d;
    return true;
}

/**
 * Short, stable id for a rule.
 *
 * Hashed over the CANONICAL form, so two spellings of one rule do not produce
 * two ids. An invalid rule has no canonical form, so it hashes its normalised
 * raw input instead - a rejected rule still needs an id to be reported under.
 */
std::string RuleId(const std::string& canonical_input) {
    uint64_t h = 1469598103934665603ull;   // FNV-1a
    for (const unsigned char c : canonical_input) {
        h ^= c;
        h *= 1099511628211ull;
    }
    std::ostringstream oss;
    oss << std::hex << h;
    return oss.str().substr(0, 12);
}

/** Upper bound accepted from config; the rule validator enforces the same. */
constexpr int64_t kMaxWindowsConfigurable = 64;

/** rate_window + sustained + bucket + 1s, or 0 with @p bad set on overflow. */
int64_t CheckedStaleDefault(const MetricWindowConfig& t, std::string* bad) {
    constexpr int64_t kMax = INT64_MAX;   // not numeric_limits: windows.h defines max()
    constexpr int64_t kSlack = 1000;
    int64_t total = t.rate_window_ms;
    for (const int64_t term : {t.sustained_ms, t.bucketIntervalMs(), kSlack}) {
        if (term < 0 || total > kMax - term) {
            if (bad->empty()) {
                *bad = "rate window and sustained are too large to derive a "
                       "stale-after default; set " +
                       std::string(env::kDeepStaleAfterMs) + " explicitly";
            }
            return 0;
        }
        total += term;
    }
    return total;
}

std::string CanonicalConfig(const DeepWindowRule& r) {
    std::ostringstream oss;
    oss << "v1|" << r.metric.canonical << '|' << toString(r.op) << '|'
        << r.threshold << '|' << r.rearm_threshold << '|'
        << r.timing.rate_window_ms << '|' << r.timing.sustained_ms << '|'
        << r.timing.stale_after_ms << '|' << r.max_windows << '|'
        << r.window.max_duration_ms << '|' << r.window.max_launches;
    return oss.str();
}

RuleCapabilities QueryCapabilities() {
    RuleCapabilities caps;
    // A Monitor-only run has no engine to arm inside a window at all.
    // The RESOLVED engine, not the requested one: a request can be overridden
    // at init, and gating on what was asked for would answer about a run that
    // is not the one happening.
    caps.windows_supported =
        Monitor::ResolvedProfilingEngine() != ProfilingEngine::Monitor;

    IMonitorBackend* backend = Monitor::GetBackend();
    // Installation can run before Windows injection receives CONTEXT_CREATED.
    // Pending is acceptable here; the queued open checks actual preparation
    // after the first CUDA context exists. A completed preparation failure is
    // rejected now instead of leaving a rule that can only open empty windows.
    caps.deep_engine_prepared =
        backend != nullptr &&
        (backend->DeepEnginesPrepared() ||
         backend->DeepEnginePreparationPending());

    caps.counters_shared = CounterProvider::isShared();
    // Under injection the target and this evaluator are separate modules, which
    // is the only situation where an unshared registry actually breaks a rule.
    caps.multi_module = EnvOrNull(env::kCudaInjection64Path) != nullptr;

    // 0 = unknown. The device list is not enumerated until the sampler takes
    // its first measurement, which is after init(), so a rule naming a device
    // cannot be refused eagerly without guessing. Guessing 1 would refuse a
    // valid gpu[3] rule on a four-GPU host - worse than deciding late. A device
    // that never reports is named in the summary instead.
    caps.device_count = 0;
    return caps;
}

void RefuseLocked(const RuleOutcome outcome, const std::string& reason) {
    g_refused = std::make_unique<RuleSummary>(RuleEvaluator::refused(
        g_rule_id, outcome, reason, detail::GetTimestampNs()));
    g_installed = true;
    GFL_LOG_ERROR("[DeepWindowRule] ", env::kDeepWhen, "='", g_expression,
                  "' disabled: ", reason,
                  ". Profiling continues; only the trigger is off.");
}

/**
 * Write one summary row.
 *
 * Shared by the terminal-transition emit and the shutdown emit so the two
 * cannot describe the same rule differently. The backend upsert accepts only a
 * strictly greater state_sequence, so the shutdown row - which always carries a
 * higher one - wins, and a redelivery of either is a no-op.
 */
void EmitSummary(const RuleSummary& summary, const std::string& expression) {
    const Runtime* rt = runtime();
    if (!rt || !rt->logger) {
        GFL_LOG_ERROR("[DeepWindowRule] no logger; summary lost: ",
                      toString(summary.outcome), " ", summary.reason);
        return;
    }

    DeepWindowRuleSummaryEvent ev;
    ev.pid            = detail::GetPid();
    ev.app            = rt->app_name;
    ev.session_id     = rt->session_id;
    ev.rule_id        = summary.rule_id;
    ev.expression     = expression;
    ev.state          = toString(summary.state);
    ev.outcome        = toString(summary.outcome);
    ev.reason         = summary.reason;
    ev.metric_state   = toString(summary.last_metric_state);
    ev.samples_seen   = summary.samples_seen;
    ev.windows_opened = summary.windows_opened;
    ev.truncated_samples = summary.truncated_samples;
    ev.metric_quality_resets = summary.metric_quality_resets;
    ev.last_quality_reason = summary.last_quality_reason;
    ev.has_last_value = summary.last_value.has_value();
    if (ev.has_last_value) {
        ev.last_value = *summary.last_value;
        ev.last_observed_ns = summary.last_observed_ns.value_or(0);
    }
    ev.state_sequence = summary.state_sequence;
    ev.emitted_ns     = summary.emitted_ns;

    rt->logger->write(model::DeepWindowRuleSummaryModel(ev));
    GFL_LOG_DEBUG("[DeepWindowRule] summary id=", ev.rule_id,
                  " outcome=", ev.outcome, " windows=", ev.windows_opened,
                  " seq=", ev.state_sequence, " reason=", ev.reason);
}

/**
 * Drop the session so a later init() installs cleanly.
 *
 * An embedded host can shutdown() and init() again in one process, and a
 * session left claimed makes the second run's rule look like a duplicate -
 * that run then has no trigger at all and nothing says why.
 */
void ReleaseSession() {
    std::lock_guard lk(g_mu);
    g_installed = false;
    g_finished = false;
    g_eval.reset();
    g_source.reset();
    g_refused.reset();
    g_refused_emitted = false;
    g_expression.clear();
    g_rule_id.clear();
}

}  // namespace

void DeepWindowRules::InstallFromEnv() {
    const char* expr = EnvOrNull(env::kDeepWhen);
    if (expr == nullptr) return;

    std::lock_guard lk(g_mu);
    if (g_installed) {
        // One rule for the MVP. Rejected loudly rather than silently ignored:
        // a second rule that quietly does nothing is worse than an error.
        GFL_LOG_ERROR("[DeepWindowRule] a rule is already installed; ignoring '",
                      expr, "'");
        return;
    }

    g_expression = expr;
    g_rule_id = RuleId("v1|raw|" + g_expression);

    RuleParseResult parsed = parseRuleExpression(g_expression);
    if (!parsed.ok()) {
        RefuseLocked(RuleOutcome::InvalidConfig,
                     parsed.detail.empty() ? toString(parsed.error) : parsed.detail);
        return;
    }

    std::string bad_env;
    DeepWindowRule rule = parsed.rule;
    rule.timing.rate_window_ms = EnvIntOr(env::kDeepRateWindowMs, 1000, &bad_env);
    // Derived from the other two so the out-of-the-box combination is one that
    // CAN fire, rather than one the validator then rejects. Summed with checks:
    // the inputs are attacker- or typo-supplied, and an overflow here would
    // produce a negative default that then reads as a different error entirely.
    const int64_t derived_stale = CheckedStaleDefault(rule.timing, &bad_env);
    rule.timing.stale_after_ms =
        EnvIntOr(env::kDeepStaleAfterMs, derived_stale, &bad_env);
    double rearm = 0.0;
    if (EnvDoubleOr(env::kDeepRearmAt, &rearm, &bad_env)) rule.rearm_threshold = rearm;
    // Range-checked BEFORE narrowing. 4294967297 fits an int64 and survives
    // the ERANGE check, then narrows to 1 - a value the validator happily
    // accepts, so the run silently uses a budget nobody asked for.
    const int64_t max_windows = EnvIntOr(env::kDeepMaxWindows, 3, &bad_env);
    if (max_windows < 1 || max_windows > kMaxWindowsConfigurable) {
        if (bad_env.empty()) {
            bad_env = std::string(env::kDeepMaxWindows) + "=" +
                      std::to_string(max_windows) + " is out of range";
        }
    }
    rule.max_windows = static_cast<int>(
        max_windows < 1 || max_windows > kMaxWindowsConfigurable ? 1 : max_windows);

    rule.window.max_duration_ms = EnvIntOr(env::kDeepWindowMs, 0, &bad_env);
    rule.window.max_launches =
        static_cast<uint64_t>(EnvIntOr(env::kDeepWindowMaxLaunches, 0, &bad_env));
    rule.window.cooldown_ms = EnvIntOr(env::kDeepWindowCooldownMs, 0, &bad_env);
    rule.window.name = "deep_window";

    if (!bad_env.empty()) {
        // Fail closed. A malformed number silently replaced by a default opens
        // real windows under settings nobody chose.
        RefuseLocked(RuleOutcome::InvalidConfig, bad_env);
        return;
    }

    const RuleParseResult checked = validateRule(rule);
    if (!checked.ok()) {
        RefuseLocked(RuleOutcome::InvalidConfig,
                     checked.detail.empty() ? toString(checked.error)
                                            : checked.detail);
        return;
    }

    g_rule_id = RuleId(CanonicalConfig(rule));
    // Cleared rather than reallocated: the feeds outlive every session so the
    // lock-free launch path never has to check whether they still exist.
    Feeds().resetForTesting();
    Feeds().seedStartup(detail::GetTimestampNs());
    g_source = std::make_unique<MetricSource>(rule.metric, rule.timing,
                                              &Feeds(),
                                              ActiveCounterProvider());
    g_eval = std::make_unique<RuleEvaluator>(rule, g_rule_id, QueryCapabilities(),
                                             g_source.get(),
                                             RuleEvaluator::liveHooks());
    g_installed = true;
    // The launch feed costs an atomic per launch, so only a rule that reads it
    // turns it on.
    g_wants_launch_feed.store(rule.metric.kind == MetricKind::KernelLaunchRate,
                              std::memory_order_release);
    g_wants_duration_feed.store(rule.metric.kind == MetricKind::RecentKernelMs,
                                std::memory_order_release);
    g_installed_relaxed.store(true, std::memory_order_release);

    GFL_LOG_DEBUG("[DeepWindowRule] installed id=", g_rule_id, " '", g_expression,
                  "' state=", toString(g_eval->state()));
}

bool DeepWindowRules::Installed() {
    std::lock_guard lk(g_mu);
    return g_installed;
}

bool DeepWindowRules::WantsLaunchFeed() {
    return g_wants_launch_feed.load(std::memory_order_acquire);
}

void DeepWindowRules::NoteKernelLaunch(const int64_t ts_ns) {
    // No lock at all: one atomic gate, then three relaxed atomics. This runs on
    // the application's launch path, and taking g_mu here would queue it behind
    // the collector's evaluation - changing the launch rate being measured.
    if (!WantsLaunchFeed()) return;
    Feeds().noteKernelLaunch(ts_ns);
}

void DeepWindowRules::NoteKernelDuration(const int64_t ts_ns, const double ms) {
    // Fed from activity processing, not from the per-launch path, so the
    // feed's own lock is acceptable here.
    if (!g_wants_duration_feed.load(std::memory_order_acquire)) return;
    Feeds().noteKernelDuration(ts_ns, ms);
}

void DeepWindowRules::NoteDeviceSample(const DeviceSample& sample,
                                       const int64_t ts_ns) {
    if (!g_installed_relaxed.load(std::memory_order_acquire)) return;
    Feeds().noteDeviceSample(sample, ts_ns);
}

void DeepWindowRules::Service() {
    RuleSummary terminal;
    std::string expression;
    {
        std::lock_guard lk(g_mu);
        if (g_finished) return;

        // A rule refused before it could run reports as soon as there is a
        // logger to report to, rather than waiting for a shutdown that may
        // never come.
        if (g_refused && !g_refused_emitted) {
            g_refused_emitted = true;
            terminal = *g_refused;
            expression = g_expression;
        } else if (g_eval) {
            const int64_t now = detail::GetTimestampNs();
            g_eval->poll(now);
            // Emitted at the transition, not at shutdown: a run that crashes
            // after spending its budget would otherwise look like one whose
            // rule simply never fired.
            if (!g_eval->takeTerminalToEmit()) return;
            terminal = g_eval->snapshot(now);
            expression = g_expression;
        } else {
            return;
        }
    }
    // Outside the lock: the logger write can block, and the launch path must
    // never queue behind it.
    EmitSummary(terminal, expression);
}

void DeepWindowRules::EmitCounterQuality() {
    // Separate from Finish() because this event exists WITHOUT a rule: NVTX
    // counters flow through the bridge whether or not anything watches them,
    // and a refused registration with no rule configured still needs a place
    // to be reported.
    NvtxCounterBridge::QualitySnapshot snap =
        NvtxCounterBridge::instance().takeSessionSnapshot();

    uint64_t discarded = 0;
    {
        std::lock_guard lk(g_mu);
        if (g_source) discarded = g_source->qualityResets();
    }

    // Silent when this SESSION had nothing to say. Gating on trackedCount()
    // was wrong: the table is process-lifetime, so one registration in an
    // earlier embedded session would emit an all-zero row for every session
    // after it - and an all-zero row with no denominator cannot tell "clean"
    // from "nothing was watched". samples_observed is session-scoped, so this
    // gate is too.
    if (!snap.any() && discarded == 0 && snap.samples_observed == 0) {
        return;
    }

    const Runtime* rt = runtime();
    if (!rt || !rt->logger) return;

    CounterDataQualitySummaryEvent ev;
    ev.pid = detail::GetPid();
    ev.app = rt->app_name;
    ev.session_id = rt->session_id;
    ev.tracked_counters = NvtxCounterBridge::instance().trackedCount();
    ev.samples_observed = snap.samples_observed;
    ev.registration_rejected = snap.registration_rejected;
    ev.unknown_id_samples = snap.unknown_id_samples;
    ev.unavailable_samples = snap.unavailable_samples;
    ev.negative_delta_samples = snap.negative_delta_samples;
    ev.rate_windows_discarded = discarded;
    ev.emitted_ns = detail::GetTimestampNs();
    rt->logger->write(model::CounterDataQualitySummaryModel(ev));
    GFL_LOG_DEBUG("[NvtxCounters] quality summary rejected=",
                  ev.registration_rejected, " unknown=", ev.unknown_id_samples,
                  " unavailable=", ev.unavailable_samples,
                  " negative=", ev.negative_delta_samples,
                  " discarded=", ev.rate_windows_discarded);
}

void DeepWindowRules::Finish() {
    RuleSummary summary;
    std::string expression;
    {
        std::lock_guard lk(g_mu);
        if (!g_installed || g_finished) return;
        g_finished = true;
        // Set before producing the summary, so a collector beat that is already
        // inside Service() cannot advance the evaluator past what is reported.
        g_wants_launch_feed.store(false, std::memory_order_release);
        g_wants_duration_feed.store(false, std::memory_order_release);
        g_installed_relaxed.store(false, std::memory_order_release);
        if (g_refused) {
            summary = *g_refused;
        } else if (g_eval) {
            summary = g_eval->finish(detail::GetTimestampNs());
        } else {
            return;
        }
        expression = g_expression;
    }

    // Released BEFORE the write is attempted. Tying the release to a successful
    // write meant that a shutdown with no logger left the session claimed
    // forever, and a host that called init() again got no rule at all - a
    // reporting failure silently turning into a functional one.
    ReleaseSession();
    EmitSummary(summary, expression);
}

void DeepWindowRules::ResetForTesting() {
    std::lock_guard lk(g_mu);
    g_installed = false;
    g_finished = false;
    g_expression.clear();
    g_rule_id.clear();
    g_eval.reset();
    g_source.reset();
    Feeds().resetForTesting();
    g_refused.reset();
    g_refused_emitted = false;
    g_wants_launch_feed.store(false, std::memory_order_release);
    g_wants_duration_feed.store(false, std::memory_order_release);
    g_installed_relaxed.store(false, std::memory_order_release);
}

}  // namespace gpufl::detail

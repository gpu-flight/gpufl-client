#include "gpufl/core/deep_window_rules.hpp"

#include <atomic>
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

std::unique_ptr<MetricFeeds>   g_feeds;
std::unique_ptr<MetricSource>  g_source;
std::unique_ptr<RuleEvaluator> g_eval;
// Set when the rule was refused before it could run, so Finish() still has
// something to report. Holding the summary rather than an error string keeps
// the refused and the ran paths on one shape.
std::unique_ptr<RuleSummary>   g_refused;

// Read by the launch callback on every launch, so it must not take the lock.
std::atomic<bool> g_wants_launch_feed{false};

const char* EnvOrNull(const char* name) {
    const char* v = std::getenv(name);
    return (v && v[0] != '\0') ? v : nullptr;
}

int64_t EnvIntOr(const char* name, const int64_t fallback) {
    const char* v = EnvOrNull(name);
    if (!v) return fallback;
    char* end = nullptr;
    const long long n = std::strtoll(v, &end, 10);
    if (end == v || *end != '\0') {
        GFL_LOG_ERROR(name, "='", v, "' is not an integer. Using ", fallback, ".");
        return fallback;
    }
    return n;
}

bool EnvDoubleOr(const char* name, double* out) {
    const char* v = EnvOrNull(name);
    if (!v) return false;
    char* end = nullptr;
    const double d = std::strtod(v, &end);
    if (end == v || *end != '\0') {
        GFL_LOG_ERROR(name, "='", v, "' is not a number. Ignoring.");
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
    caps.windows_supported = g_opts.profiling_engine != ProfilingEngine::Monitor;

    IMonitorBackend* backend = Monitor::GetBackend();
    caps.deep_engine_prepared = backend != nullptr && backend->DeepEnginesPrepared();

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

    DeepWindowRule rule = parsed.rule;
    rule.timing.rate_window_ms = EnvIntOr(env::kDeepRateWindowMs, 1000);
    rule.timing.stale_after_ms = EnvIntOr(
        env::kDeepStaleAfterMs,
        // Default derived from the other two so the out-of-the-box combination
        // is one that CAN fire, rather than one the validator then rejects.
        rule.timing.rate_window_ms + rule.timing.sustained_ms +
            rule.timing.bucketIntervalMs() + 1000);
    double rearm = 0.0;
    if (EnvDoubleOr(env::kDeepRearmAt, &rearm)) rule.rearm_threshold = rearm;
    rule.max_windows = static_cast<int>(EnvIntOr(env::kDeepMaxWindows, 3));

    rule.window.max_duration_ms = EnvIntOr(env::kDeepWindowMs, 0);
    rule.window.max_launches =
        static_cast<uint64_t>(EnvIntOr(env::kDeepWindowMaxLaunches, 0));
    rule.window.cooldown_ms = EnvIntOr(env::kDeepWindowCooldownMs, 0);
    rule.window.name = "deep_window";

    const RuleParseResult checked = validateRule(rule);
    if (!checked.ok()) {
        RefuseLocked(RuleOutcome::InvalidConfig,
                     checked.detail.empty() ? toString(checked.error)
                                            : checked.detail);
        return;
    }

    g_rule_id = RuleId(CanonicalConfig(rule));
    g_feeds = std::make_unique<MetricFeeds>();
    g_feeds->seedStartup(detail::GetTimestampNs());
    g_source = std::make_unique<MetricSource>(rule.metric, rule.timing,
                                              g_feeds.get(),
                                              ActiveCounterProvider());
    g_eval = std::make_unique<RuleEvaluator>(rule, g_rule_id, QueryCapabilities(),
                                             g_source.get(),
                                             RuleEvaluator::liveHooks());
    g_installed = true;
    // The launch feed costs an atomic per launch, so only a rule that reads it
    // turns it on.
    g_wants_launch_feed.store(
        rule.metric.kind == MetricKind::KernelLaunchRate ||
            rule.metric.kind == MetricKind::RecentKernelMs,
        std::memory_order_release);

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
    if (!WantsLaunchFeed()) return;
    std::lock_guard lk(g_mu);
    if (g_feeds) g_feeds->noteKernelLaunch(ts_ns);
}

void DeepWindowRules::NoteKernelDuration(const int64_t ts_ns, const double ms) {
    if (!WantsLaunchFeed()) return;
    std::lock_guard lk(g_mu);
    if (g_feeds) g_feeds->noteKernelDuration(ts_ns, ms);
}

void DeepWindowRules::NoteDeviceSample(const DeviceSample& sample,
                                       const int64_t ts_ns) {
    std::lock_guard lk(g_mu);
    if (g_feeds) g_feeds->noteDeviceSample(sample, ts_ns);
}

void DeepWindowRules::Service() {
    std::lock_guard lk(g_mu);
    if (!g_eval) return;
    g_eval->poll(detail::GetTimestampNs());
}

void DeepWindowRules::Finish() {
    RuleSummary summary;
    {
        std::lock_guard lk(g_mu);
        if (!g_installed || g_finished) return;
        g_finished = true;
        if (g_refused) {
            summary = *g_refused;
        } else if (g_eval) {
            summary = g_eval->finish(detail::GetTimestampNs());
        } else {
            return;
        }
    }

    const Runtime* rt = runtime();
    if (!rt || !rt->logger) {
        // Nothing to write to. Logged rather than dropped silently, since this
        // is the record that explains why no window ever appeared.
        GFL_LOG_ERROR("[DeepWindowRule] no logger at shutdown; summary lost: ",
                      toString(summary.outcome), " ", summary.reason);
        return;
    }

    DeepWindowRuleSummaryEvent ev;
    ev.pid            = detail::GetPid();
    ev.app            = rt->app_name;
    ev.session_id     = rt->session_id;
    ev.rule_id        = summary.rule_id;
    ev.expression     = g_expression;
    ev.state          = toString(summary.state);
    ev.outcome        = toString(summary.outcome);
    ev.reason         = summary.reason;
    ev.metric_state   = toString(summary.last_metric_state);
    ev.samples_seen   = summary.samples_seen;
    ev.windows_opened = summary.windows_opened;
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
                  " reason=", ev.reason);
}

void DeepWindowRules::ResetForTesting() {
    std::lock_guard lk(g_mu);
    g_installed = false;
    g_finished = false;
    g_expression.clear();
    g_rule_id.clear();
    g_eval.reset();
    g_source.reset();
    g_feeds.reset();
    g_refused.reset();
    g_wants_launch_feed.store(false, std::memory_order_release);
}

}  // namespace gpufl::detail

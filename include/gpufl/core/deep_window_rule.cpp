#include "gpufl/core/deep_window_rule.hpp"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <utility>

namespace gpufl::detail {
namespace {

/** Largest window a rule may ask for. Property 4: nothing is unbounded. */
constexpr int64_t  kMaxWindowDurationMs = 60 * 1000;
constexpr uint64_t kMaxWindowLaunches   = 5000000;
constexpr int      kMaxWindowsHardLimit = 64;

std::string trim(const std::string& s) {
    const size_t b = s.find_first_not_of(" \t");
    if (b == std::string::npos) return {};
    const size_t e = s.find_last_not_of(" \t");
    return s.substr(b, e - b + 1);
}

/** Parse "2s" / "500ms" / "2000" (bare = milliseconds). Negative on failure. */
int64_t parseDurationMs(const std::string& text) {
    const std::string s = trim(text);
    if (s.empty()) return -1;

    size_t digits = 0;
    while (digits < s.size() && s[digits] >= '0' && s[digits] <= '9') ++digits;
    if (digits == 0) return -1;

    errno = 0;
    const long long value = std::strtoll(s.substr(0, digits).c_str(), nullptr, 10);
    if (errno != 0 || value < 0) return -1;

    const std::string unit = trim(s.substr(digits));
    if (unit.empty() || unit == "ms") return value;
    if (unit == "s") {
        if (value > kMaxWindowDurationMs) return -1;   // also guards the *1000
        return value * 1000;
    }
    return -1;
}

bool parseDouble(const std::string& text, double* out) {
    const std::string s = trim(text);
    if (s.empty()) return false;
    char* end = nullptr;
    errno = 0;
    const double v = std::strtod(s.c_str(), &end);
    if (errno != 0 || end == s.c_str() || *end != '\0') return false;
    if (!std::isfinite(v)) return false;
    *out = v;
    return true;
}

}  // namespace

const char* toString(const Comparison op) {
    return op == Comparison::LessThan ? "<" : ">";
}

const char* toString(const RuleError e) {
    switch (e) {
        case RuleError::None:                 return "ok";
        case RuleError::BadMetric:            return "bad_metric";
        case RuleError::BadTiming:            return "bad_timing";
        case RuleError::ThresholdNotFinite:   return "threshold_not_finite";
        case RuleError::RearmWrongSide:       return "rearm_wrong_side";
        case RuleError::MaxWindowsOutOfRange: return "max_windows_out_of_range";
        case RuleError::WindowBoundsMissing:  return "window_bounds_missing";
        case RuleError::WindowBoundsTooLarge: return "window_bounds_too_large";
        case RuleError::Unparsable:           return "unparsable";
        case RuleError::DuplicateRule:        return "duplicate_rule";
    }
    return "unknown";
}

const char* toString(const RuleState s) {
    switch (s) {
        case RuleState::Inactive:        return "inactive";
        case RuleState::WarmingUp:       return "warming_up";
        case RuleState::Armed:           return "armed";
        case RuleState::Pending:         return "pending";
        case RuleState::Opening:         return "opening";
        case RuleState::Blackout:        return "blackout";
        case RuleState::Recovery:        return "recovery";
        case RuleState::WaitingForRearm: return "waiting_for_rearm";
    }
    return "unknown";
}

const char* toString(const RuleOutcome o) {
    switch (o) {
        case RuleOutcome::None:          return "none";
        case RuleOutcome::NeverTrue:     return "never_true";
        case RuleOutcome::Fired:         return "fired";
        case RuleOutcome::Exhausted:     return "exhausted";
        case RuleOutcome::Unsupported:   return "unsupported";
        case RuleOutcome::InvalidConfig: return "invalid_config";
    }
    return "unknown";
}

const char* toString(const RuleGate g) {
    switch (g) {
        case RuleGate::Ok:                 return "ok";
        case RuleGate::MetricUnavailable:  return "metric_unavailable";
        case RuleGate::CountersNotShared:  return "counters_not_shared";
        case RuleGate::NoDeepEngine:       return "no_deep_engine";
        case RuleGate::WindowsUnsupported: return "windows_unsupported";
    }
    return "unknown";
}

// ------------------------------------------------------------------- parsing

RuleParseResult parseRuleExpression(const std::string& text) {
    RuleParseResult out;

    const size_t op_at = text.find_first_of("<>");
    if (op_at == std::string::npos) {
        out.error = RuleError::Unparsable;
        out.detail = "expected '<' or '>' in \"" + text + "\"";
        return out;
    }
    out.rule.op = text[op_at] == '<' ? Comparison::LessThan : Comparison::GreaterThan;

    const MetricParseResult metric = parseMetric(trim(text.substr(0, op_at)));
    if (!metric.ok()) {
        out.error = RuleError::BadMetric;
        out.metric_error = metric.error;
        out.detail = toString(metric.error);
        return out;
    }
    out.rule.metric = metric.id;

    std::string rest = text.substr(op_at + 1);
    // " for <duration>" is optional; without it the rule fires on the first
    // fresh true reading.
    int64_t sustained_ms = 0;
    if (const size_t f = rest.find(" for "); f != std::string::npos) {
        sustained_ms = parseDurationMs(rest.substr(f + 5));
        if (sustained_ms < 0) {
            out.error = RuleError::Unparsable;
            out.detail = "bad duration in \"" + text + "\"";
            return out;
        }
        rest = rest.substr(0, f);
    }

    if (!parseDouble(rest, &out.rule.threshold)) {
        out.error = RuleError::Unparsable;
        out.detail = "bad threshold in \"" + text + "\"";
        return out;
    }

    out.rule.timing.sustained_ms = sustained_ms;
    // Plain "condition false" until an explicit hysteresis value is supplied.
    out.rule.rearm_threshold = out.rule.threshold;
    return out;
}

RuleParseResult validateRule(const DeepWindowRule& rule) {
    RuleParseResult out;
    out.rule = rule;

    if (!std::isfinite(rule.threshold) || !std::isfinite(rule.rearm_threshold)) {
        // NaN makes every comparison false, so the rule would sit armed forever
        // while looking perfectly healthy.
        out.error = RuleError::ThresholdNotFinite;
        return out;
    }

    // A rearm on the wrong side of the operator can never be reached, so the
    // rule fires exactly once and then waits for a condition that cannot occur.
    const bool rearm_ok = rule.op == Comparison::LessThan
                              ? rule.rearm_threshold >= rule.threshold
                              : rule.rearm_threshold <= rule.threshold;
    if (!rearm_ok) {
        out.error = RuleError::RearmWrongSide;
        out.detail = std::string("rearm ") + toString(rule.op) +
                     " rule needs rearm on the other side of the threshold";
        return out;
    }

    if (rule.max_windows < 1 || rule.max_windows > kMaxWindowsHardLimit) {
        out.error = RuleError::MaxWindowsOutOfRange;
        return out;
    }

    if (rule.window.max_duration_ms <= 0 && rule.window.max_launches == 0) {
        // A window with neither bound never closes on its own, which turns a
        // bounded-cost feature into an always-on one.
        out.error = RuleError::WindowBoundsMissing;
        return out;
    }
    if (rule.window.max_duration_ms > kMaxWindowDurationMs ||
        rule.window.max_launches > kMaxWindowLaunches) {
        out.error = RuleError::WindowBoundsTooLarge;
        return out;
    }

    if (const ConfigError e = validate(rule.timing); e != ConfigError::None) {
        out.error = RuleError::BadTiming;
        out.config_error = e;
        out.detail = explain(rule.timing, e);
        return out;
    }

    return out;
}

// ----------------------------------------------------------------- evaluator

RuleEvaluator::Hooks RuleEvaluator::liveHooks() {
    Hooks h;
    h.request_open = [](void*, const DeepWindowSpec& spec) {
        return DeepWindow::RequestOpenTagged(spec);
    };
    h.window_active = [](void*) { return DeepWindow::Active(); };
    h.opens_completed = [](void*) { return DeepWindow::OpensCompleted(); };
    h.last_opened_token = [](void*) { return DeepWindow::LastOpenedToken(); };
    h.pending_open_token = [](void*) { return DeepWindow::PendingOpenToken(); };
    h.ctx = nullptr;
    return h;
}

RuleEvaluator::RuleEvaluator(DeepWindowRule rule, std::string rule_id,
                             const RuleCapabilities& caps, MetricSource* source,
                             Hooks hooks)
    : rule_(std::move(rule)),
      rule_id_(std::move(rule_id)),
      source_(source),
      hooks_(hooks) {
    // Two independent gates, each with its own reason. A rule can be perfectly
    // valid and still have nothing to watch, or nothing to arm.
    RuleGate gate = RuleGate::Ok;
    if (!caps.windows_supported) {
        gate = RuleGate::WindowsUnsupported;
    } else if (!caps.deep_engine_prepared) {
        // An enum check on the configured engine is not enough: without this a
        // rule spends its whole budget opening windows that arm nothing.
        gate = RuleGate::NoDeepEngine;
    } else if (rule_.metric.resolvesLazily() && caps.multi_module &&
               !caps.counters_shared) {
        // The target ticks one registry and this evaluator reads another, so
        // the counter can never be seen. Refusing here is the honest answer;
        // the alternative is reporting a counter that is being ticked as
        // Missing for the whole run.
        gate = RuleGate::CountersNotShared;
    } else if (caps.device_count > 0 && !rule_.metric.resolvesLazily() &&
               rule_.metric.shape() == MetricShape::Gauge &&
               rule_.metric.device_index >= caps.device_count) {
        // Built-in metrics are decided eagerly - unlike a custom counter, the
        // answer cannot change later.
        gate = RuleGate::MetricUnavailable;
    }

    if (gate != RuleGate::Ok) {
        state_ = RuleState::Inactive;
        terminal_ = RuleOutcome::Unsupported;
        reason_ = toString(gate);
    }
}

DeepWindowSpec RuleEvaluator::specWithTrigger(const MetricSample& sample) const {
    DeepWindowSpec spec = rule_.window;
    spec.trigger.present         = true;
    spec.trigger.rule_id         = rule_id_;
    spec.trigger.metric          = rule_.metric.canonical;
    spec.trigger.op              = toString(rule_.op);
    spec.trigger.threshold       = rule_.threshold;
    spec.trigger.rearm_threshold = rule_.rearm_threshold;
    spec.trigger.observed        = sample.value;
    spec.trigger.rate_window_ms  = rule_.timing.rate_window_ms;
    spec.trigger.sustained_ms    = rule_.timing.sustained_ms;
    spec.trigger.first_true_ns   = first_true_observed_ns_.value_or(sample.observed_ns);
    spec.trigger.fired_ns        = sample.observed_ns;
    return spec;
}

bool RuleEvaluator::conditionHolds(const double value) const {
    return rule_.op == Comparison::LessThan ? value < rule_.threshold
                                            : value > rule_.threshold;
}

bool RuleEvaluator::rearmHolds(const double value) const {
    // One predicate, not "false or threshold". With rearm == threshold this is
    // exactly "condition false", which is the no-hysteresis default.
    return rule_.op == Comparison::LessThan ? value >= rule_.rearm_threshold
                                            : value <= rule_.rearm_threshold;
}

void RuleEvaluator::toArmed() {
    state_ = RuleState::Armed;
    first_true_observed_ns_.reset();
    ++state_sequence_;
}

void RuleEvaluator::enterBlackout(const int64_t) {
    state_ = RuleState::Blackout;
    first_true_observed_ns_.reset();
    ++state_sequence_;
}

void RuleEvaluator::poll(const int64_t now_ns) {
    if (state_ == RuleState::Inactive) return;

    const bool active = hooks_.window_active(hooks_.ctx);

    // A window can open and close entirely between two beats - a launch-bounded
    // one over a busy loop routinely does. The counter catches that; a boolean
    // would not, and the samples taken during it would feed the rule as clean.
    const uint64_t opens = hooks_.opens_completed != nullptr
                               ? hooks_.opens_completed(hooks_.ctx) : 0;
    const bool missed_window = have_opens_ && opens != opens_seen_ && !active;
    opens_seen_ = opens;
    have_opens_ = true;

    // 1. Confirm a requested open before anything else. The window we caused
    //    also puts us into blackout, so checking blackout first would lose the
    //    only chance to count it.
    if (state_ == RuleState::Opening) {
        if (pending_token_ != 0 &&
            hooks_.last_opened_token(hooks_.ctx) == pending_token_) {
            pending_token_ = 0;
            ++windows_opened_;
            if (windows_opened_ >= static_cast<uint32_t>(rule_.max_windows)) {
                // Marked at the open that reaches the limit, not later: a run
                // that crashes during this window still explains itself.
                terminal_ = RuleOutcome::Exhausted;
            }
            enterBlackout(now_ns);
            window_was_active_ = true;
            return;
        }
        else if (hooks_.pending_open_token(hooks_.ctx) != pending_token_) {
            // No longer queued and never opened: the coordinator turned it down
            // when it got round to it. No budget is consumed - the budget
            // bounds what the rule COST, and this cost nothing.
            //
            // Checked against the queue rather than "is a window open yet",
            // because an open is serviced on a later beat and treating "not
            // yet" as "never" would abandon a window about to open.
            pending_token_ = 0;
            reason_ = "open_request_dropped";
            toArmed();
        }
    }

    // 2. Any open window contaminates, whoever opened it. Contamination does
    //    not care who asked.
    if (active) {
        if (state_ != RuleState::Blackout) enterBlackout(now_ns);
        window_was_active_ = true;
        return;
    }

    // 3. A window just closed - either one we watched, or one that came and
    //    went between beats. Blackout and recovery are distinct: blackout is
    //    "discard everything", recovery is "refill the clean epoch". Merging
    //    them would let contaminated samples prove the workload recovered.
    if (window_was_active_ || missed_window) {
        window_was_active_ = false;
        source_->resetEpoch(now_ns);
        if (terminal_ == RuleOutcome::Exhausted) {
            state_ = RuleState::Inactive;
            ++state_sequence_;
            return;
        }
        state_ = RuleState::Recovery;
        ++state_sequence_;
    }

    const MetricSample sample = source_->poll(now_ns);
    last_metric_state_ = sample.state;

    // Staleness is checked BEFORE the repeat filter, deliberately. A source
    // that dies stops advancing its sequence, so a rule that only looked at new
    // samples would never notice - it would sit in Pending on a reading nobody
    // is taking any more, which is the exact failure the two-timestamp design
    // exists to prevent.
    if (sample.state != MetricState::Fresh) {
        // Not evidence the condition stopped holding, but not evidence it
        // continued either, so the run is broken rather than completed.
        if (state_ == RuleState::Pending) {
            first_true_observed_ns_.reset();
            state_ = RuleState::Armed;
            ++state_sequence_;
        }
        return;
    }

    // Only a NEW fresh reading is evidence. The evaluator runs 100-2000x faster
    // than a metric publishes, so counting repeats would let a single reading
    // satisfy any sustained_ms on its own.
    const bool is_new = !have_sequence_ || sample.sequence != last_sequence_;
    if (!is_new) return;
    last_sequence_ = sample.sequence;
    have_sequence_ = true;

    ++samples_seen_;
    last_value_ = sample.value;
    last_observed_ns_ = sample.observed_ns;

    switch (state_) {
        case RuleState::WarmingUp:
        case RuleState::Recovery:
            // A full window of clean data has now arrived, since a Fresh sample
            // implies it.
            state_ = state_ == RuleState::Recovery ? RuleState::WaitingForRearm
                                                   : RuleState::Armed;
            ++state_sequence_;
            if (state_ == RuleState::Armed) break;
            [[fallthrough]];

        case RuleState::WaitingForRearm:
            if (rearmHolds(sample.value)) {
                toArmed();
            }
            break;

        case RuleState::Armed:
            if (conditionHolds(sample.value)) {
                first_true_observed_ns_ = sample.observed_ns;
                state_ = RuleState::Pending;
                ++state_sequence_;
                // sustained_ms == 0 fires on this same reading, handled below.
                if (rule_.timing.sustained_ms == 0) {
                    pending_token_ = hooks_.request_open(hooks_.ctx, specWithTrigger(sample));
                    if (pending_token_ != 0) {
                        state_ = RuleState::Opening;
                        ++state_sequence_;
                    } else {
                        reason_ = "open_refused";
                        toArmed();
                    }
                }
            }
            break;

        case RuleState::Pending: {
            if (!conditionHolds(sample.value)) {
                toArmed();
                break;
            }
            if (!first_true_observed_ns_.has_value()) {
                first_true_observed_ns_ = sample.observed_ns;
                break;
            }
            // A span between two observations, not an accumulation - so one
            // stale reading re-read many times can never satisfy it.
            const int64_t held_ns = sample.observed_ns - *first_true_observed_ns_;
            if (held_ns < rule_.timing.sustained_ms * 1000000) break;

            pending_token_ = hooks_.request_open(hooks_.ctx, specWithTrigger(sample));
            if (pending_token_ != 0) {
                state_ = RuleState::Opening;
                ++state_sequence_;
            } else {
                // Refused by cooldown. Back to armed rather than holding
                // pending, which would fire the instant the cooldown lapsed on
                // evidence gathered long before.
                reason_ = "open_refused";
                toArmed();
            }
            break;
        }

        case RuleState::Opening:
        case RuleState::Blackout:
        case RuleState::Inactive:
            break;
    }
}

RuleSummary RuleEvaluator::snapshot(const int64_t now_ns) const {
    RuleSummary s;
    s.rule_id = rule_id_;
    s.state = state_;
    s.outcome = terminal_;
    s.samples_seen = samples_seen_;
    s.windows_opened = windows_opened_;
    s.last_value = last_value_;
    s.last_observed_ns = last_observed_ns_;
    s.last_metric_state = last_metric_state_;
    s.reason = reason_;
    s.state_sequence = state_sequence_;
    s.emitted_ns = now_ns;
    return s;
}

RuleSummary RuleEvaluator::finish(const int64_t now_ns) {
    RuleSummary s = snapshot(now_ns);

    // Precedence, highest first. Several can apply at once and the most
    // specific has to win, or a rule that was refused outright would report as
    // one that simply never matched.
    if (terminal_ == RuleOutcome::InvalidConfig ||
        terminal_ == RuleOutcome::Unsupported) {
        s.outcome = terminal_;
    } else if (terminal_ == RuleOutcome::Exhausted) {
        s.outcome = RuleOutcome::Exhausted;
    } else if (windows_opened_ > 0) {
        s.outcome = RuleOutcome::Fired;
    } else {
        s.outcome = RuleOutcome::NeverTrue;
        if (s.reason.empty()) {
            // Distinguish "watched it and it never happened" from "never had
            // anything to watch"; both leave windows_opened at 0.
            if (last_metric_state_ == MetricState::Missing) {
                s.reason = "custom_metric_never_registered";
            } else if (samples_seen_ == 0) {
                // Watched, but the source never produced a usable reading -
                // a device that does not exist, or a sampler that never ran.
                s.reason = "metric_source_never_reported";
            } else {
                s.reason = "condition_never_held";
            }
        }
    }

    s.state_sequence = ++state_sequence_;
    return s;
}

RuleSummary RuleEvaluator::refused(std::string rule_id, const RuleOutcome outcome,
                                   std::string reason, const int64_t now_ns) {
    RuleSummary s;
    s.rule_id = std::move(rule_id);
    // The terminal outcomes leave the evaluator standing in none of the running
    // states, which is what `inactive` is for.
    s.state = RuleState::Inactive;
    s.outcome = outcome;
    s.reason = std::move(reason);
    s.emitted_ns = now_ns;
    s.state_sequence = 1;
    return s;
}

}  // namespace gpufl::detail

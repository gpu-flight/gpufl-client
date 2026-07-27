#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "gpufl/core/deep_window.hpp"
#include "gpufl/core/metric_id.hpp"
#include "gpufl/core/metric_registry.hpp"

namespace gpufl::detail {

enum class Comparison { LessThan, GreaterThan };

const char* toString(Comparison op);

/**
 * @brief A condition that opens a deep window.
 *
 * The predicate form already exists - it is the `if` a caller writes around
 * gpufl::deepWindow(). A rule earns its place on two counts only: metrics
 * gpufl has and the application does not, and thresholds that live in
 * configuration rather than in code.
 */
struct DeepWindowRule {
    MetricId       metric;
    Comparison     op = Comparison::LessThan;
    double         threshold = 0.0;
    /**
     * Value the metric must recover past before the rule may fire again.
     *
     * Equal to `threshold` by default, which degenerates to plain "condition
     * false" - the no-hysteresis case. A rule whose rearm sits on the wrong
     * side of the operator can never rearm, so the direction is validated.
     */
    double         rearm_threshold = 0.0;
    MetricWindowConfig timing;
    int            max_windows = 3;
    DeepWindowSpec window;
};

/**
 * @brief Why a rule was refused, or could not be honoured.
 *
 * Property 5 of the design: every reason a rule did not fire is recorded. A
 * rejected rule that leaves no trace is indistinguishable from one that was
 * simply never true, and the two call for opposite responses from the user.
 */
enum class RuleError {
    None,
    /// The metric name itself was refused; carries the parse reason.
    BadMetric,
    /// The timing combination cannot produce evidence; carries the config reason.
    BadTiming,
    ThresholdNotFinite,   ///< NaN makes every comparison false, silently
    RearmWrongSide,       ///< LessThan needs rearm >= threshold, and vice versa
    MaxWindowsOutOfRange,
    WindowBoundsMissing,  ///< a window with no bound at all never closes
    WindowBoundsTooLarge,
    Unparsable,           ///< the --deep-when expression did not parse
    DuplicateRule,        ///< MVP allows exactly one
};

const char* toString(RuleError e);

/** @brief Parse `<metric><op><threshold> for <duration>`, e.g. "x<1000 for 2s". */
struct RuleParseResult {
    DeepWindowRule rule;
    RuleError      error = RuleError::None;
    /// Set when error is BadMetric / BadTiming, so the reason is not lost.
    MetricParseError metric_error = MetricParseError::None;
    ConfigError      config_error = ConfigError::None;
    std::string      detail;

    bool ok() const { return error == RuleError::None; }
};

/**
 * @brief Parse the expression half of the config.
 *
 * Only the metric, operator, threshold and sustained duration come from the
 * expression; everything else is a separate option. Splitting them keeps each
 * refusal able to name the field it is about.
 */
RuleParseResult parseRuleExpression(const std::string& text);

/** @brief Check a fully assembled rule. Returns RuleError::None when usable. */
RuleParseResult validateRule(const DeepWindowRule& rule);

// ---------------------------------------------------------------------------

/**
 * @brief Where the evaluator is standing.
 *
 * Never a verdict on the session - see RuleOutcome for that. Collapsing the
 * two would make `armed` look like a conclusion, and force every reader to
 * guess which kind of answer it was holding.
 */
enum class RuleState {
    Inactive,          ///< terminal: invalid, unsupported, or budget spent
    WarmingUp,
    Armed,
    Pending,           ///< condition true, waiting out sustained_ms
    Opening,           ///< an open was requested; waiting to see it happen
    Blackout,          ///< a window is open; everything observed is contaminated
    Recovery,          ///< window closed; refilling the clean epoch
    WaitingForRearm,
};

const char* toString(RuleState s);

/** @brief What the session concluded about the rule. */
enum class RuleOutcome {
    None,
    NeverTrue,
    Fired,
    Exhausted,
    Unsupported,
    InvalidConfig,
};

const char* toString(RuleOutcome o);

/** @brief Which gate refused a rule that was otherwise well-formed. */
enum class RuleGate {
    Ok,
    /// The metric cannot be produced: no such device, sampler absent, or the
    /// base mode does not generate it.
    MetricUnavailable,
    /// Custom counters are not shared across modules, so a counter the target
    /// ticks is invisible to the evaluator that would read it.
    CountersNotShared,
    /// Windows are meaningless without an engine to arm inside them. An enum
    /// check on the configured engine is necessary but not sufficient: without
    /// this a rule burns its budget opening windows that arm nothing.
    NoDeepEngine,
    /// The base mode does not support bounded windows at all.
    WindowsUnsupported,
};

const char* toString(RuleGate g);

/**
 * @brief Everything the session learned about one rule.
 *
 * `state` and `outcome` are separate on purpose. A crash mid-run leaves a
 * meaningful state and no outcome; a clean run with a rule that never matched
 * leaves `armed` and `never_true`. One field cannot carry both.
 */
struct RuleSummary {
    std::string rule_id;
    RuleState   state = RuleState::Inactive;
    RuleOutcome outcome = RuleOutcome::None;
    uint64_t    samples_seen = 0;
    uint32_t    windows_opened = 0;
    /// Absent, not NaN: there is a real "no value yet", and NaN does not
    /// survive the JSON and DB boundaries cleanly.
    std::optional<double>  last_value;
    std::optional<int64_t> last_observed_ns;
    MetricState last_metric_state = MetricState::Missing;
    /**
     * Completed kernels discarded across the run.
     *
     * Reported so a conclusion drawn from a partial percentile is not
     * presented as one drawn from all of it. Counted rather than used to
     * suppress the metric: truncation starts at launch rates far below what
     * the workloads this feature targets actually reach.
     */
    uint64_t    truncated_samples = 0;
    std::string reason;
    uint64_t    state_sequence = 0;
    int64_t     emitted_ns = 0;
};

/**
 * @brief The gates a rule must pass before it can open anything.
 *
 * Supplied by the caller rather than queried here so the evaluator stays
 * testable without a GPU, and so the two gates are visibly independent - each
 * has its own recorded failure reason.
 */
struct RuleCapabilities {
    bool windows_supported = true;
    bool deep_engine_prepared = true;
    bool counters_shared = true;
    /// More than one module holds a copy of gpufl - injection. Only then does
    /// an unshared counter registry actually break anything.
    bool multi_module = false;
    int  device_count = 1;
};

/**
 * @brief Drives one rule from metric samples to a window request.
 *
 * Runs on the collector beat (~1ms) alongside serviceDeepWindow.
 */
class RuleEvaluator {
public:
    /** @brief Injected so tests drive the coordinator without a GPU. */
    struct Hooks {
        /// Ask for a window; returns a token, or 0 when refused outright.
        uint64_t (*request_open)(void* ctx, const DeepWindowSpec&) = nullptr;
        /// Is a window - any window, whoever opened it - currently open?
        bool (*window_active)(void* ctx) = nullptr;
        /// Monotonic count of windows that have opened. A short launch-bounded
        /// window can open AND close between two beats, so polling a boolean
        /// would miss it entirely and its contaminated samples would feed the
        /// rule as if profiling had never been on.
        uint64_t (*opens_completed)(void* ctx) = nullptr;
        /// Token of the most recent open, so a manual window is not mistaken
        /// for the one this rule asked for and charged to its budget.
        uint64_t (*last_opened_token)(void* ctx) = nullptr;
        /// Token of the request still queued, or 0. Without it "has not opened
        /// yet" and "will never open" are indistinguishable, and the evaluator
        /// abandons a window that was about to open.
        uint64_t (*pending_open_token)(void* ctx) = nullptr;
        void* ctx = nullptr;
    };

    /** @brief Hooks wired to the real DeepWindow coordinator. */
    static Hooks liveHooks();

    RuleEvaluator(DeepWindowRule rule, std::string rule_id,
                  const RuleCapabilities& caps, MetricSource* source, Hooks hooks);

    /** @brief Advance the machine. Call every collector beat. */
    void poll(int64_t now_ns);

    /** @brief Finalise at shutdown and return what the session concluded. */
    RuleSummary finish(int64_t now_ns);

    /** @brief Current summary without concluding, for a mid-run emit. */
    RuleSummary snapshot(int64_t now_ns) const;

    /**
     * @brief True once, when a terminal outcome is first reached.
     *
     * `exhausted` and `unsupported` are conclusions the run can reach long
     * before it ends. Holding them until shutdown means a process that crashes
     * afterwards explains nothing, and the session looks like one where the
     * rule simply never fired. Reported here so the caller can write it at the
     * transition; the shutdown summary still follows, with a higher sequence.
     */
    bool takeTerminalToEmit();

    RuleState state() const { return state_; }
    uint32_t windowsOpened() const { return windows_opened_; }

    /**
     * @brief Mark a rule that was refused before it could run.
     *
     * Kept separate from the constructor because an invalid rule must never
     * fail init(): configuration is parsed during init, and a hard failure
     * there would leave no session and no telemetry writer - nowhere to record
     * the very outcome that has to be reported.
     */
    static RuleSummary refused(std::string rule_id, RuleOutcome outcome,
                               std::string reason, int64_t now_ns);

private:
    /**
     * The window spec plus the comparison that produced it.
     *
     * Built at the request, not at the close: by the time the window closes the
     * reading that caused it is long gone, and a bare observed value with no
     * threshold beside it stops being readable the first time the rule changes.
     */
    DeepWindowSpec specWithTrigger(const MetricSample& sample) const;
    bool conditionHolds(double value) const;
    bool rearmHolds(double value) const;
    void enterBlackout(int64_t now_ns);
    void toArmed();

    DeepWindowRule rule_;
    std::string    rule_id_;
    MetricSource*  source_ = nullptr;
    Hooks          hooks_;

    RuleState   state_ = RuleState::WarmingUp;
    RuleOutcome terminal_ = RuleOutcome::None;
    std::string reason_;

    uint64_t samples_seen_ = 0;
    uint64_t last_sequence_ = 0;
    bool     have_sequence_ = false;
    uint32_t windows_opened_ = 0;
    uint64_t state_sequence_ = 0;

    /// Start of the current unbroken run of true readings. Compared against a
    /// later sample's timestamp, so sustained_ms is a span between two
    /// observations rather than an accumulation that one stale reading could
    /// satisfy on its own.
    std::optional<int64_t> first_true_observed_ns_;
    std::optional<double>  last_value_;
    std::optional<int64_t> last_observed_ns_;
    MetricState last_metric_state_ = MetricState::Missing;
    uint64_t    truncated_samples_ = 0;

    uint64_t pending_token_ = 0;
    bool     window_was_active_ = false;
    uint64_t opens_seen_ = 0;
    bool     have_opens_ = false;
    bool     terminal_emitted_ = false;
};

}  // namespace gpufl::detail

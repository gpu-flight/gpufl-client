#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace gpufl {

/**
 * One bounded deep-profiling window: the region between a
 * gpufl::deepWindow() trigger and the bound that closed it.
 *
 * Deep engines arm on open and disarm on close, so this is the only
 * record of what the window actually covered. Both the requested bounds
 * and the outcome are carried, because they routinely disagree: under
 * kernel replay a three-second window can cover a dozen launches, and
 * `close_reason` is what tells the reader that was the deadline expiring
 * rather than the profiler failing.
 */
/**
 * What a rule observed at the moment it asked for a window.
 *
 * Carried on the window rather than looked up later. A bare `trigger_value=842`
 * becomes unreadable the first time somebody edits the threshold, so the whole
 * comparison travels with the window it caused.
 *
 * `present` is false for a window nobody triggered - manual, scheduled, or the
 * launcher's --deep-after.
 */
struct DeepWindowTrigger {
    bool        present = false;
    std::string rule_id;
    std::string metric;          // canonical name, device index included
    std::string op;              // "<" or ">"
    double      threshold = 0.0;
    double      rearm_threshold = 0.0;
    double      observed = 0.0;
    int64_t     rate_window_ms = 0;
    int64_t     sustained_ms = 0;
    int64_t     first_true_ns = 0;   // start of the run of true readings
    int64_t     fired_ns = 0;
};

struct DeepWindowEvent {
    DeepWindowTrigger trigger;
    int pid = 0;
    std::string app;
    std::string session_id;
    std::string name;
    std::string close_reason;   // DeepWindowCloseName(): "deadline" | ...
    // Wire names of the deep engines this window actually armed. Empty means
    // the window opened but armed nothing, which is a real outcome worth
    // recording. Trace never appears here: it runs session-wide rather than
    // arming with the window.
    std::vector<std::string> engines;
    int64_t start_ns = 0;
    int64_t end_ns = 0;
    int64_t duration_ns = 0;
    uint64_t launches_covered = 0;
    // What the caller asked for, so a short window is self-explanatory.
    int64_t requested_duration_ms = 0;
    uint64_t requested_max_launches = 0;
};

/**
 * What the session concluded about a conditional rule.
 *
 * Emitted even when the rule never fired. A rule that leaves no record is
 * indistinguishable from one that was never true, and from a run that crashed
 * before it could report - three situations calling for different responses.
 *
 * `state` and `outcome` are separate: state is where the evaluator was
 * standing, outcome is the verdict. One field cannot carry both without making
 * `armed` look like a conclusion.
 */
struct DeepWindowRuleSummaryEvent {
    int pid = 0;
    std::string app;
    std::string session_id;
    std::string rule_id;
    std::string expression;      // the configured rule, as written
    std::string state;
    std::string outcome;
    std::string reason;
    std::string metric_state;
    uint64_t samples_seen = 0;
    uint32_t windows_opened = 0;
    // Absent rather than a sentinel: there is a real "no value yet", and NaN
    // does not survive the JSON and DB boundaries cleanly.
    bool     has_last_value = false;
    double   last_value = 0.0;
    int64_t  last_observed_ns = 0;
    /**
     * Completed kernels discarded before the percentile was computed.
     *
     * 0 for every metric that is not a percentile, and for a percentile that
     * kept everything. Non-zero says the conclusion rests on a subset - which
     * a value alone can never show.
     */
    uint64_t truncated_samples = 0;
    /// Rate windows THIS rule's metric discarded for failed reads (NVTX
    /// SAMPLE_UNAVAILABLE), and why the last one was. Per rule, never the
    /// session total - an unrelated counter's errors must not look like they
    /// broke this rule.
    uint64_t    metric_quality_resets = 0;
    std::string last_quality_reason;
    // Monotonic, so a redelivered or late record cannot overwrite a newer one.
    uint64_t state_sequence = 0;
    int64_t  emitted_ns = 0;
};

/**
 * Session-scoped data quality of application-fed counters.
 *
 * NOT capture capability: none of these say what the GPU or driver supports.
 * They say what the APPLICATION sent - refused registrations, samples for ids
 * nobody issued, reads the application itself failed, deltas that went
 * backwards - and how often the metric layer had to discard a rate window
 * because of it. Each field has ONE meaning; a combined tally is a number
 * nobody can act on.
 *
 * Values are this SESSION's, not the process totals: the tallies live for the
 * process (like counter slots) and an embedded host re-initialises, so raw
 * totals would re-report session one's problems as session two's.
 */
struct CounterDataQualitySummaryEvent {
    int pid = 0;
    std::string app;
    std::string session_id;
    /**
     * Which counter path these tallies observed. Only "nvtx" today: the
     * gpufl::counter() API's own rejections are not routed through the
     * bridge, and a generic-looking row would claim coverage it does not
     * have - a gpufl::counter registration failure beside
     * registration_rejected: 0 would read as "nothing went wrong".
     */
    std::string source = "nvtx";
    int schema_version = 1;
    /// Registration-table size at emit (process-lifetime context).
    uint64_t tracked_counters = 0;
    /// Valid samples THIS session - the denominator that distinguishes
    /// "0 failures out of many" from "0 out of 0".
    uint64_t samples_observed = 0;
    uint64_t registration_rejected = 0;
    uint64_t unknown_id_samples = 0;
    uint64_t unavailable_samples = 0;
    uint64_t negative_delta_samples = 0;
    uint64_t rate_windows_discarded = 0;
    int64_t  emitted_ns = 0;
};

}  // namespace gpufl

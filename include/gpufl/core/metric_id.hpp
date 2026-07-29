#pragma once

#include <cstdint>
#include <string>

namespace gpufl::detail {

/**
 * @brief What a rule is allowed to watch.
 *
 * Deliberately small. Every entry is something gpufl knows and the application
 * does not - otherwise the caller may as well write the `if` themselves, which
 * is the alternative this feature has to beat.
 */
enum class MetricKind {
    GpuUtilPct,        ///< gpu[N].util_pct
    GpuPowerMw,        ///< gpu[N].power_mw
    GpuSmClockMhz,     ///< gpu[N].sm_clock_mhz
    KernelLaunchRate,  ///< kernel_launch_rate, launches/s at the HOST API
    RecentKernelMs,    ///< recent_kernel_ms, p50 kernel duration over the window
    CustomRate,        ///< custom.<name>_rate, ticks/s
};

/** @brief How a metric behaves, which decides warm-up and empty-window rules. */
enum class MetricShape {
    /// Events per second. An empty window is a real 0.
    Rate,
    /// A percentile over samples. An empty window has no value at all.
    Percentile,
    /// Last successful measurement. Advances only on a real measurement.
    Gauge,
};

/**
 * @brief A parsed, canonicalised metric name.
 *
 * `canonical` is what the rule id hashes over, so `gpu[00].util_pct` and
 * `gpu[0].util_pct` cannot produce two different ids for one rule.
 */
struct MetricId {
    MetricKind  kind = MetricKind::KernelLaunchRate;
    int         device_index = 0;    ///< only meaningful for gpu[N].* metrics
    std::string custom_name;         ///< the <name> in custom.<name>_rate
    std::string canonical;

    MetricShape shape() const;
    /** @brief True when the metric cannot be resolved until the app registers it. */
    bool resolvesLazily() const { return kind == MetricKind::CustomRate; }
};

/**
 * @brief Why a metric name or a rule config was refused.
 *
 * Every value here surfaces as `outcome=invalid_config` with this string as the
 * reason. A rejected rule that leaves no trace is indistinguishable from one
 * that was simply never true, so nothing may fail silently.
 */
enum class MetricParseError {
    None,
    Empty,
    UnknownBuiltinMetric,   ///< looks built-in, is not: gpu[0].temperature_pct
    MissingCustomPrefix,    ///< bare name that is not built-in: tokne_rate
    MalformedDeviceIndex,   ///< gpu[].util_pct, gpu[-1].util_pct, gpu[x].util_pct
    MalformedCustomMetric,  ///< custom.<name>_rate with a bad or missing name
    CustomNameTooLong,
    CustomNameCharset,
};

const char* toString(MetricParseError e);
const char* toString(MetricKind k);

struct MetricParseResult {
    MetricId         id;
    MetricParseError error = MetricParseError::None;
    bool ok() const { return error == MetricParseError::None; }
};

/**
 * @brief Parse a metric name from config.
 *
 * The `custom.` prefix is what makes a typo detectable here rather than at
 * shutdown. Without it `tokne_rate` is syntactically indistinguishable from a
 * counter that has simply not registered yet, and the mistake only surfaces
 * once the run is over.
 */
MetricParseResult parseMetric(const std::string& text);

/**
 * @brief Timing configuration shared by the metric source and the rule.
 *
 * Held together because the three fields are not independently valid: a
 * combination that can never produce evidence is a config error even though
 * each field on its own looks reasonable.
 */
struct MetricWindowConfig {
    /**
     * Upper bound on the rate window.
     *
     * Property 4 of the design is that no setting is unbounded. The window also
     * sizes the bucket ring, so an unbounded value is an unbounded allocation
     * driven by a config string.
     */
    static constexpr int64_t kMaxRateWindowMs = 5 * 60 * 1000;   // 5 minutes

    int64_t rate_window_ms = 1000;
    int64_t sustained_ms   = 2000;
    int64_t stale_after_ms = 5000;

    /**
     * @brief Bucket width, derived so the validator and the producer agree.
     *
     * Internal and not configurable. Defined once here precisely because two
     * independent definitions would eventually disagree and the validator would
     * start accepting configurations the producer cannot satisfy.
     */
    int64_t bucketIntervalMs() const;

    /** @brief Number of buckets covering the rate window (>= 1). */
    int64_t bucketCount() const;
};

enum class ConfigError {
    None,
    RateWindowNotPositive,
    RateWindowTooLarge,
    SustainedNegative,
    StaleAfterNotPositive,
    /// stale_after_ms < rate_window + sustained + bucket: the rule goes stale
    /// before it can ever accumulate the evidence needed to fire.
    StaleBeforeEvidence,
};

const char* toString(ConfigError e);

/**
 * @brief Reject a configuration that cannot work, with the arithmetic named.
 *
 * The interesting case is StaleBeforeEvidence. A total stall only fires if the
 * metric stays Fresh long enough to accumulate `sustained_ms` of zero readings,
 * and the first zero does not appear until a full rate window has elapsed. The
 * weaker pairwise checks (`stale > window`, `stale >= sustained`) accept
 * `window=4s, sustained=4s, stale=5s`, which produces its first zero at t=4 and
 * goes stale at t=5 - four seconds short of ever firing.
 */
ConfigError validate(const MetricWindowConfig& cfg);

/** @brief Human-readable explanation including the numbers, for the summary. */
std::string explain(const MetricWindowConfig& cfg, ConfigError e);

}  // namespace gpufl::detail

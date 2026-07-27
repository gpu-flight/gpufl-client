#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "gpufl/abi/gpufl_counter_abi.h"
#include "gpufl/core/metric_id.hpp"

namespace gpufl {
struct DeviceSample;
}

namespace gpufl::detail {

/**
 * @brief Whether a reading may be used as evidence, and if not, why.
 *
 * The split between Fresh and Stale is what stops a rule satisfying
 * `sustained_ms` by re-reading one stale sample. System metrics publish every
 * 100-500ms while the evaluator runs at ~1ms, so without this the same reading
 * would be counted hundreds of times and every rule would fire on its first
 * true sample.
 */
enum class MetricState {
    Missing,     ///< custom counter never registered, or source unavailable
    WarmingUp,   ///< registered, but no full window of evidence yet
    Fresh,       ///< usable, INCLUDING a genuine 0
    Stale,       ///< the source stopped producing
};

const char* toString(MetricState s);

/**
 * @brief One reading.
 *
 * Two timestamps, not one. A zero-tick bucket still publishes a new sequence
 * and a new `observed_ns`, so publication time alone can never go stale - it
 * advances forever even during a total stall, which is exactly the condition a
 * rule most needs to detect. Staleness is therefore measured against the
 * source:
 *
 *     Stale  <=>  observed_ns - last_source_event_ns > stale_after_ms
 *
 * That lets zeros accumulate as fresh evidence and still detects a dead source.
 */
struct MetricSample {
    double      value = 0.0;
    int64_t     observed_ns = 0;
    int64_t     last_source_event_ns = 0;
    uint64_t    sequence = 0;
    MetricState state = MetricState::Missing;

    bool usable() const { return state == MetricState::Fresh; }
};

/**
 * @brief Feeds every metric reads from, owned by the runtime.
 *
 * Separate from MetricSource because the feeds are process-global while a
 * source belongs to one rule. Fed from the collector and sampler threads and
 * read from the evaluator, so everything here is under one mutex; the volumes
 * involved are per-kernel-launch at worst, not per-sample.
 */
class MetricFeeds {
public:
    /**
     * @brief A kernel launch was observed at the HOST launch API.
     *
     * Host launch rate, not GPU execution rate. Naming it `kernel_launch_rate`
     * rather than a bare "kernel rate" is deliberate: a launch storm and a slow
     * kernel look opposite here, and a caller who confuses the two writes a rule
     * that fires on the wrong condition.
     */
    void noteKernelLaunch(int64_t ts_ns);

    /** @brief A completed kernel's duration, for the recent_kernel_ms window. */
    void noteKernelDuration(int64_t ts_ns, double duration_ms);

    /** @brief A successful NVML/SMI measurement. Never called on polling. */
    void noteDeviceSample(const DeviceSample& sample, int64_t ts_ns);

    /**
     * @brief Seed the launch source's timestamp at runtime startup.
     *
     * The launch source exists from startup, so its zero is a legitimate Fresh
     * reading - but staleness needs something to measure against, and without a
     * seed a run with no launches yet would read Stale rather than "0 launches
     * per second", which is the opposite verdict.
     */
    void seedStartup(int64_t ts_ns);

    /** @brief How many GPUs the device collector reported. 0 until first seen. */
    int deviceCount() const;

    struct LaunchFeed {
        uint64_t count = 0;
        int64_t  last_event_ns = 0;
        bool     seeded = false;
    };
    struct DurationFeed {
        std::vector<double> samples;   // durations since the last drain
        int64_t last_event_ns = 0;
    };
    struct GaugeFeed {
        double   value = 0.0;
        int64_t  last_event_ns = 0;
        uint64_t measurements = 0;
    };

    LaunchFeed launchFeed() const;
    /**
     * @brief Take and clear the durations accumulated since the last call.
     *
     * Single-consumer: whoever drains gets the samples and the next caller sees
     * an empty feed. Fine while the MVP allows one rule; a second consumer of
     * recent_kernel_ms would silently starve the first.
     */
    DurationFeed drainDurations();
    /** @brief Last duration timestamp WITHOUT draining, for staleness checks. */
    int64_t durationsLastEventNs() const;
    GaugeFeed gaugeFeed(MetricKind kind, int device_index) const;

    void resetForTesting();

private:
    mutable std::mutex mu_;
    LaunchFeed   launch_;
    DurationFeed durations_;
    struct DeviceGauges {
        GaugeFeed util;
        GaugeFeed power;
        GaugeFeed sm_clock;
    };
    std::vector<DeviceGauges> devices_;
};

/**
 * @brief One metric, sampled on a rate window, for one rule.
 *
 * `poll()` is called from the collector loop at ~1ms and closes a bucket
 * whenever one is due. It publishes a new sequence on every closed bucket even
 * when nothing ticked - otherwise a real zero never accumulates as evidence and
 * a total stall would read as "no new data" rather than "the rate is 0", and
 * the rule that exists to catch stalls would be the one rule that cannot.
 */
class MetricSource {
public:
    MetricSource(MetricId id, MetricWindowConfig cfg, MetricFeeds* feeds,
                 const gpufl_counter_provider_v1* counters);

    /**
     * @brief Advance to @p now_ns and return the current reading.
     *
     * Cheap and idempotent within a bucket: repeated calls between bucket
     * boundaries return the same sequence, which is what lets the rule
     * evaluator ignore a repeat rather than counting it as new evidence.
     */
    MetricSample poll(int64_t now_ns);

    /**
     * @brief Discard accumulated evidence and start a fresh window.
     *
     * Called when a deep window closes. Buckets filled while profiling was
     * active describe a contaminated workload, and letting them prove the
     * workload recovered is how a rule ends up re-firing on its own overhead.
     */
    void resetEpoch(int64_t now_ns);

    const MetricId& id() const { return id_; }
    const MetricWindowConfig& config() const { return cfg_; }
    /** @brief True once a custom counter has been resolved to a live handle. */
    bool customResolved() const { return handle_ != nullptr; }

private:
    void closeBucket(int64_t boundary_ns);
    bool resolveCustomHandle();
    double windowRatePerSec() const;
    double windowPercentile() const;

    MetricId           id_;
    MetricWindowConfig cfg_;
    MetricFeeds*       feeds_ = nullptr;
    const gpufl_counter_provider_v1* counters_ = nullptr;

    gpufl_counter_handle handle_ = nullptr;   // custom metrics only

    int64_t bucket_ns_ = 0;
    int64_t window_ns_ = 0;
    int64_t stale_ns_  = 0;

    /** Cap on durations kept per bucket; a launch storm must not grow memory. */
    static constexpr size_t kMaxDurationsPerBucket = 4096;

    // Ring of per-bucket counts. Sized once from the validated config.
    std::vector<uint64_t> buckets_;
    // A percentile does not decompose across buckets, so durations are kept
    // per bucket and expire with it rather than being folded into a sum.
    std::vector<std::vector<double>> bucket_durations_;
    size_t   head_ = 0;
    /// Closes in the CURRENT epoch; decides whether the window is full.
    uint64_t buckets_closed_ = 0;
    /**
     * Closes over the whole process, never reset.
     *
     * The sequence has to be monotonic for the evaluator to tell new evidence
     * from a repeat. Deriving it from the per-epoch count would send it
     * backwards on every epoch reset, and the evaluator would then ignore real
     * samples whose numbers it had already seen.
     */
    uint64_t total_closes_ = 0;

    int64_t  next_boundary_ns_ = 0;
    uint64_t last_source_total_ = 0;   ///< counter/launch total at last close
    bool     baselined_ = false;
    bool     first_tick_seen_ = false;
    /**
     * When the custom counter last moved.
     *
     * The registry stores values, not times, so a custom metric's source
     * timestamp has to be observed here - at bucket close, whenever the delta
     * is non-zero. Without it there is nothing to measure staleness against and
     * a dead counter would read as a steady rate of 0 forever.
     */
    int64_t  last_tick_ns_ = 0;
    uint64_t last_published_close_ = 0;

    MetricSample current_;
    uint64_t     sequence_ = 0;
};

}  // namespace gpufl::detail

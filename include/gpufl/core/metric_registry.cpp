#include "gpufl/core/metric_registry.hpp"

#include <algorithm>
#include <utility>

#include "gpufl/core/events.hpp"

namespace gpufl::detail {
namespace {

constexpr int64_t kNsPerMs = 1000000;

}  // namespace

const char* toString(const MetricState s) {
    switch (s) {
        case MetricState::Missing:   return "missing";
        case MetricState::WarmingUp: return "warming_up";
        case MetricState::Fresh:     return "fresh";
        case MetricState::Stale:     return "stale";
    }
    return "unknown";
}

// ---------------------------------------------------------------- MetricFeeds

void MetricFeeds::noteKernelLaunch(const int64_t ts_ns) {
    // Lock-free: this is the application's launch path.
    launch_count_.fetch_add(1, std::memory_order_relaxed);
    launch_last_ns_.store(ts_ns, std::memory_order_relaxed);
    launch_seeded_.store(true, std::memory_order_release);
}

void MetricFeeds::noteKernelDuration(const int64_t ts_ns, const double duration_ms) {
    std::lock_guard lk(mu_);
    // Bounded here, at the push. Trimming after a drain left the buffer free to
    // grow without limit until that drain came - and the case where it grows
    // fastest is a stalled collector, which is also the case where the drain is
    // late.
    if (durations_.samples.size() >= kMaxPendingDurations) {
        ++durations_.dropped;
    } else {
        durations_.samples.push_back(DurationSample{ts_ns, duration_ms});
    }
    // Advances even when the sample is dropped: the SOURCE is alive, and
    // freezing this would report a busy workload as a dead one.
    durations_.last_event_ns = ts_ns;
}

void MetricFeeds::noteDeviceSample(const DeviceSample& sample, const int64_t ts_ns) {
    if (sample.device_id < 0) return;
    std::lock_guard lk(mu_);
    const auto index = static_cast<size_t>(sample.device_id);
    if (index >= devices_.size()) devices_.resize(index + 1);
    DeviceGauges& g = devices_[index];

    // The measurement timestamp advances only here, never on a poll. A metric
    // whose sequence advanced because someone asked about it could never be
    // detected as dead.
    g.util.value = static_cast<double>(sample.gpu_util);
    g.util.last_event_ns = ts_ns;
    ++g.util.measurements;

    g.power.value = static_cast<double>(sample.power_mw);
    g.power.last_event_ns = ts_ns;
    ++g.power.measurements;

    g.sm_clock.value = static_cast<double>(sample.clock_sm);
    g.sm_clock.last_event_ns = ts_ns;
    ++g.sm_clock.measurements;
}

void MetricFeeds::seedStartup(const int64_t ts_ns) {
    {
        std::lock_guard lk(mu_);
        durations_.last_event_ns = ts_ns;
    }
    if (launch_seeded_.load(std::memory_order_acquire)) return;
    launch_last_ns_.store(ts_ns, std::memory_order_relaxed);
    launch_seeded_.store(true, std::memory_order_release);
}

int MetricFeeds::deviceCount() const {
    std::lock_guard lk(mu_);
    return static_cast<int>(devices_.size());
}

MetricFeeds::LaunchFeed MetricFeeds::launchFeed() const {
    LaunchFeed out;
    // Seeded first: it is released last on the write side, so observing it
    // true means the count and timestamp beside it are already visible.
    out.seeded = launch_seeded_.load(std::memory_order_acquire);
    out.count = launch_count_.load(std::memory_order_relaxed);
    out.last_event_ns = launch_last_ns_.load(std::memory_order_relaxed);
    return out;
}

MetricFeeds::DurationFeed MetricFeeds::drainDurationsUpTo(const int64_t boundary_ns) {
    std::lock_guard lk(mu_);
    DurationFeed out;
    out.last_event_ns = durations_.last_event_ns;
    out.dropped = durations_.dropped;
    durations_.dropped = 0;

    // Kernels complete roughly in order, so a linear partition is enough; the
    // point is that a sample from after the boundary stays for the bucket it
    // belongs to rather than being counted in an earlier one.
    std::vector<DurationSample> keep;
    keep.reserve(durations_.samples.size());
    for (const DurationSample& s : durations_.samples) {
        if (s.ts_ns <= boundary_ns) {
            out.samples.push_back(s);
        } else {
            keep.push_back(s);
        }
    }
    durations_.samples.swap(keep);
    return out;
}

MetricFeeds::DurationFeed MetricFeeds::drainDurations() {
    std::lock_guard lk(mu_);
    DurationFeed out;
    out.samples = std::move(durations_.samples);
    out.last_event_ns = durations_.last_event_ns;
    out.dropped = durations_.dropped;
    durations_.samples.clear();
    durations_.dropped = 0;
    return out;
}

int64_t MetricFeeds::durationsLastEventNs() const {
    std::lock_guard lk(mu_);
    return durations_.last_event_ns;
}

MetricFeeds::GaugeFeed MetricFeeds::gaugeFeed(const MetricKind kind,
                                              const int device_index) const {
    std::lock_guard lk(mu_);
    if (device_index < 0 || static_cast<size_t>(device_index) >= devices_.size()) {
        return {};
    }
    const DeviceGauges& g = devices_[static_cast<size_t>(device_index)];
    switch (kind) {
        case MetricKind::GpuUtilPct:    return g.util;
        case MetricKind::GpuPowerMw:    return g.power;
        case MetricKind::GpuSmClockMhz: return g.sm_clock;
        default:                        return {};
    }
}

void MetricFeeds::resetForTesting() {
    std::lock_guard lk(mu_);
    durations_ = DurationFeed{};
    devices_.clear();
    launch_count_.store(0, std::memory_order_relaxed);
    launch_last_ns_.store(0, std::memory_order_relaxed);
    launch_seeded_.store(false, std::memory_order_release);
}

// --------------------------------------------------------------- MetricSource

MetricSource::MetricSource(MetricId id, MetricWindowConfig cfg, MetricFeeds* feeds,
                           const gpufl_counter_provider_v1* counters)
    : id_(std::move(id)), cfg_(cfg), feeds_(feeds), counters_(counters) {
    bucket_ns_ = cfg_.bucketIntervalMs() * kNsPerMs;
    const auto count = static_cast<size_t>(cfg_.bucketCount());
    buckets_.assign(count, 0);
    if (id_.shape() == MetricShape::Percentile) {
        bucket_durations_.resize(count);
        bucket_truncated_.assign(count, false);
    }
    // The ring, not the configured window, is what the rate divides by: the
    // bucket count rounds up, so the two differ and using the configured value
    // would report a rate the samples do not support.
    window_ns_ = static_cast<int64_t>(count) * bucket_ns_;
    stale_ns_  = cfg_.stale_after_ms * kNsPerMs;
}

bool MetricSource::resolveCustomHandle() {
    if (handle_ != nullptr) return true;
    if (counters_ == nullptr || counters_->lookup == nullptr) return false;
    // Lookup, never register. A rule naming a counter asks a question; creating
    // the counter here would answer it with itself.
    handle_ = counters_->lookup(id_.custom_name.c_str(), id_.custom_name.size());
    return handle_ != nullptr;
}

double MetricSource::windowRatePerSec() const {
    uint64_t total = 0;
    for (const uint64_t n : buckets_) total += n;
    const double seconds = static_cast<double>(window_ns_) / 1e9;
    if (seconds <= 0.0) return 0.0;
    return static_cast<double>(total) / seconds;
}

double MetricSource::windowPercentile() const {
    std::vector<double> all;
    for (const auto& bucket : bucket_durations_) {
        all.insert(all.end(), bucket.begin(), bucket.end());
    }
    if (all.empty()) return 0.0;
    const size_t mid = all.size() / 2;
    std::nth_element(all.begin(), all.begin() + mid, all.end());
    return all[mid];
}

void MetricSource::closeBucket(const int64_t boundary_ns) {
    head_ = (head_ + 1) % buckets_.size();
    ++buckets_closed_;
    ++total_closes_;

    switch (id_.shape()) {
        case MetricShape::Rate: {
            uint64_t total = 0;
            if (id_.kind == MetricKind::KernelLaunchRate) {
                total = feeds_->launchFeed().count;
            } else if (resolveCustomHandle()) {
                // Since the session baseline, not the raw lifetime total. A
                // permanent slot keeps its value across shutdown()/init(), so a
                // counter ticked by a PREVIOUS session would otherwise read as
                // already-ticked here and a stall rule would arm on evidence
                // this run never saw.
                total = counters_->load_since_baseline(handle_);
                if (total > 0) first_tick_seen_ = true;
            }
            // Unsigned delta: correct across a wrap, which is why the counter is
            // allowed to wrap rather than saturate.
            const uint64_t delta = baselined_ ? total - last_source_total_ : 0;
            last_source_total_ = total;
            baselined_ = true;
            buckets_[head_] = delta;
            if (delta > 0) last_tick_ns_ = boundary_ns;
            break;
        }
        case MetricShape::Percentile: {
            // Up to THIS boundary only. Draining everything would put a
            // catch-up's whole backlog into the oldest bucket and leave the
            // rest of the window empty.
            MetricFeeds::DurationFeed drained =
                feeds_->drainDurationsUpTo(boundary_ns);
            auto& slot = bucket_durations_[head_];
            slot.clear();
            bucket_truncated_[head_] = drained.dropped > 0;
            if (drained.samples.size() > kMaxDurationsPerBucket) {
                drained.samples.resize(kMaxDurationsPerBucket);
                bucket_truncated_[head_] = true;
            }
            slot.reserve(drained.samples.size());
            for (const MetricFeeds::DurationSample& s : drained.samples) {
                slot.push_back(s.ms);
            }
            if (drained.dropped > 0) durations_truncated_ += drained.dropped;
            break;
        }
        case MetricShape::Gauge:
            break;
    }

    current_.observed_ns = boundary_ns;
}

MetricSample MetricSource::poll(const int64_t now_ns) {
    if (next_boundary_ns_ == 0) {
        next_boundary_ns_ = now_ns + bucket_ns_;
        current_.observed_ns = now_ns;
    }

    // A collector that stalled longer than the whole window has no valid
    // evidence left, so jump rather than replaying thousands of empty buckets.
    // Replaying them would also be wrong: it would look like a run of measured
    // zeros and could fire a stall rule that nothing actually observed.
    if (now_ns - next_boundary_ns_ > window_ns_) {
        std::fill(buckets_.begin(), buckets_.end(), 0);
        for (auto& b : bucket_durations_) b.clear();
        // The FEED as well, not just the local buckets. Durations recorded
        // before the stall describe a workload from before the gap; letting the
        // next bucket inherit them would fire a rule on evidence that is
        // already older than the window it claims to cover.
        if (id_.shape() == MetricShape::Percentile) feeds_->drainDurations();
        buckets_closed_ = 0;
        baselined_ = false;
        next_boundary_ns_ = now_ns + bucket_ns_;
        current_.observed_ns = now_ns;
    }

    while (now_ns >= next_boundary_ns_) {
        closeBucket(next_boundary_ns_);
        next_boundary_ns_ += bucket_ns_;
    }

    const bool window_full = buckets_closed_ >= buckets_.size();

    switch (id_.shape()) {
        case MetricShape::Rate: {
            int64_t source_ns = 0;
            if (id_.kind == MetricKind::KernelLaunchRate) {
                const MetricFeeds::LaunchFeed feed = feeds_->launchFeed();
                if (!feed.seeded) {
                    // Nothing has seeded the launch source, so there is no
                    // baseline to measure staleness against yet.
                    current_.state = MetricState::WarmingUp;
                    current_.observed_ns = now_ns;
                    current_.sequence = total_closes_;
                    return current_;
                }
                source_ns = feed.last_event_ns;
            } else {
                if (!resolveCustomHandle()) {
                    // Not registered yet. Resolved lazily on purpose: an env
                    // rule is parsed during init(), long before application code
                    // reaches gpufl::counter(), so deciding this at install time
                    // would reject every counter rule before it could exist.
                    current_.state = MetricState::Missing;
                    current_.observed_ns = now_ns;
                    current_.sequence = total_closes_;
                    return current_;
                }
                if (!first_tick_seen_ &&
                    counters_->load_since_baseline(handle_) > 0) {
                    first_tick_seen_ = true;
                    // Seed the source clock at the moment the counter is first
                    // seen to move; ticks before this point have no timestamp.
                    last_tick_ns_ = now_ns;
                }
                if (!first_tick_seen_) {
                    // Registered but never ticked. Not Missing - the counter
                    // exists - and not Fresh 0 either, because a workload that
                    // has not started yet must not read as a stalled one.
                    current_.state = MetricState::WarmingUp;
                    current_.observed_ns = now_ns;
                    current_.sequence = total_closes_;
                    return current_;
                }
                source_ns = last_tick_ns_;
            }

            current_.value = windowRatePerSec();
            current_.last_source_event_ns = source_ns;
            // Every closed bucket is a publication, including one with no
            // ticks. That is what lets a genuine zero accumulate as evidence
            // instead of looking like "no new data" - without it, the stall a
            // rule most needs to catch is the one case it never could.
            current_.sequence = total_closes_;
            if (!window_full) {
                current_.state = MetricState::WarmingUp;
            } else if (now_ns - source_ns > stale_ns_) {
                current_.state = MetricState::Stale;
            } else {
                current_.state = MetricState::Fresh;
            }
            break;
        }
        case MetricShape::Percentile: {
            const bool have_samples =
                std::any_of(bucket_durations_.begin(), bucket_durations_.end(),
                            [](const std::vector<double>& b) { return !b.empty(); });
            if (!window_full) {
                current_.state = MetricState::WarmingUp;
                break;
            }
            if (!have_samples) {
                // An empty window has no percentile, so nothing is published:
                // neither a value nor a new sequence. Publishing 0 ms would read
                // as instantaneous kernels rather than no kernels, and a rule
                // watching for slow kernels would silently never fire.
                current_.state = MetricState::Stale;
                break;
            }
            if (total_closes_ != last_published_close_) {
                last_published_close_ = total_closes_;
                ++sequence_;
            }
            current_.value = windowPercentile();
            current_.last_source_event_ns = feeds_->durationsLastEventNs();
            current_.sequence = sequence_;
            current_.state = MetricState::Fresh;
            break;
        }
        case MetricShape::Gauge: {
            const MetricFeeds::GaugeFeed feed =
                feeds_->gaugeFeed(id_.kind, id_.device_index);
            if (feed.measurements == 0) {
                current_.state = MetricState::WarmingUp;
                break;
            }
            current_.value = feed.value;
            current_.last_source_event_ns = feed.last_event_ns;
            // Measurements, not polls. A sequence that advanced because someone
            // asked would let one reading satisfy sustained_ms on its own.
            current_.sequence = feed.measurements;
            current_.state = now_ns - feed.last_event_ns > stale_ns_
                                 ? MetricState::Stale
                                 : MetricState::Fresh;
            break;
        }
    }

    return current_;
}

void MetricSource::resetEpoch(const int64_t now_ns) {
    std::fill(buckets_.begin(), buckets_.end(), 0);
    for (auto& b : bucket_durations_) b.clear();
    buckets_closed_ = 0;
    baselined_ = false;
    next_boundary_ns_ = now_ns + bucket_ns_;
    current_.observed_ns = now_ns;
    current_.state = MetricState::WarmingUp;
    // Drop anything the feed accumulated during the window for the same reason
    // the buckets are cleared: it describes a contaminated workload.
    if (id_.shape() == MetricShape::Percentile) feeds_->drainDurations();
}

}  // namespace gpufl::detail

#include "gpufl/core/sampler.hpp"

#include "gpufl/core/deep_window_rules.hpp"

#include "gpufl/core/common.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/logger/logger.hpp"
#include "gpufl/core/model/batch_models.hpp"
#include "gpufl/core/runtime.hpp"
#include "gpufl/core/segment_runtime.hpp"

namespace gpufl {
Sampler::Sampler() = default;
Sampler::~Sampler() { shutdown(); }

void Sampler::configure(std::string appName, std::string sessionId,
                        std::shared_ptr<Logger> logger,
                        std::shared_ptr<ISystemCollector<DeviceSample>> collector,
                        const int sampleIntervalMs,
                        HostCollector* hostCollector) {
    auto fixed_context = std::make_shared<SegmentContext>(
        std::string(), std::move(sessionId), 0, 0, std::move(logger));
    auto fixed_runtime = std::make_shared<Runtime>();
    fixed_runtime->publishSegmentContext(fixed_context);
    configure(
        std::move(appName),
        [fixed_runtime]() {
            return fixed_runtime->acquireSegmentContext();
        },
        std::move(collector), sampleIntervalMs, hostCollector);
}

void Sampler::configure(
    std::string appName, SegmentProvider segmentProvider,
    std::shared_ptr<ISystemCollector<DeviceSample>> collector,
    const int sampleIntervalMs, HostCollector* hostCollector,
    RowObserver rowObserver) {
    std::lock_guard lk(mu_);
    appName_        = std::move(appName);
    segment_provider_ = std::move(segmentProvider);
    collector_      = std::move(collector);
    host_collector_ = hostCollector;
    row_observer_ = std::move(rowObserver);
    intervalMs_     = sampleIntervalMs;
}

void Sampler::activate() {
    std::lock_guard lk(mu_);
    const int prev = activations_.fetch_add(1, std::memory_order_acq_rel);
    if (prev == 0) {
        // 0 → 1 transition: start the worker thread if we're configured
        // enough to do useful work. If we're not configured, the counter
        // still increments - the next deactivate balances it. This keeps
        // the API safe to call before configure().
        if (segment_provider_ && collector_ && intervalMs_ > 0 &&
            !running_.load()) {
            startWorkerLocked_();
        }
    }
}

void Sampler::deactivate() {
    std::lock_guard lk(mu_);
    const int prev = activations_.fetch_sub(1, std::memory_order_acq_rel);
    if (prev <= 0) {
        // Clamp at zero and warn once. Negative activations means an
        // unbalanced call (deactivate without a matching activate); we
        // don't propagate it because the caller may be in a destructor
        // and surfacing exceptions there would be worse than the bug.
        activations_.store(0, std::memory_order_release);
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true)) {
            GFL_LOG_ERROR("[Sampler] deactivate() called more times than "
                          "activate() - clamping at zero. Check for "
                          "unbalanced systemStart/systemStop or leaked "
                          "ScopedMonitor destructors.");
        }
        return;
    }
    if (prev == 1) {
        // 1 → 0 transition: stop the worker.
        if (running_.load()) {
            stopWorkerLocked_();
        }
    }
}

void Sampler::shutdown() {
    std::lock_guard lk(mu_);
    activations_.store(0, std::memory_order_release);
    if (running_.load()) {
        stopWorkerLocked_();
    }
}

void Sampler::startWorkerLocked_() {
    batch_.clear();
    batch_context_.reset();
    batch_id_ = 0;
    host_batch_.clear();
    host_batch_id_ = 0;
    running_.store(true, std::memory_order_release);
    th_ = std::thread([this] { runLoop_(); });
}

void Sampler::stopWorkerLocked_() {
    running_.store(false, std::memory_order_release);
    std::thread toJoin = std::move(th_);
    // Releasing the lock during join would let another activate() race
    // ahead and start a second worker. Instead we join while holding
    // the lock; the worker doesn't touch mu_, so no deadlock.
    if (toJoin.joinable()) {
        if (toJoin.get_id() == std::this_thread::get_id()) {
            toJoin.detach();
        } else {
            toJoin.join();
        }
    }
}

void Sampler::runLoop_() {
    using clock = std::chrono::steady_clock;
    auto interval       = std::chrono::milliseconds(intervalMs_);
    auto next_wake_time = clock::now();
    int  samples_since_flush = 0;

    while (running_.load()) {
        next_wake_time += interval;
        const int64_t ts = detail::GetTimestampNs();
        auto context =
            segment_provider_ ? segment_provider_() : nullptr;
        if (!context || !context->logger) {
            std::this_thread::sleep_until(next_wake_time);
            continue;
        }
        if (batch_context_ && batch_context_.get() != context.get()) {
            if (!batch_.empty() || !host_batch_.empty()) {
                flushBatches_(batch_context_);
                samples_since_flush = 0;
            } else {
                batch_context_.reset();
            }
        }
        if (!batch_context_) batch_context_ = std::move(context);

        for (const DeviceSample& d : collector_->sampleAll()) {
            // A rule reads gauges from here, not by polling: the timestamp has
            // to be when the measurement was actually taken, or a metric could
            // never be detected as having stopped.
            detail::DeepWindowRules::NoteDeviceSample(d, ts);

            DeviceMetricBatchRow row;
            row.ts_ns            = ts;
            row.device_id        = d.device_id;
            row.gpu_util         = d.gpu_util;
            row.mem_util         = d.mem_util;
            row.temp_c           = d.temp_c;
            row.power_mw         = d.power_mw;
            row.used_mib         = d.used_mib;
            row.total_mib        = d.total_mib;
            row.clock_sm         = d.clock_sm;
            row.fan_speed_pct    = d.fan_speed_pct;
            row.temp_mem_c       = d.temp_mem_c;
            row.temp_junction_c  = d.temp_junction_c;
            row.voltage_mv       = d.voltage_mv;
            row.energy_uj        = d.energy_uj;
            row.clock_mem        = d.clock_mem;
            row.pcie_bw_bps      = d.pcie_rx_bps + d.pcie_tx_bps;
            row.ecc_corrected    = d.ecc_corrected;
            row.ecc_uncorrected  = d.ecc_uncorrected;
            batch_.push(row);
        }

        if (host_collector_) {
            const HostSample hs = host_collector_->sample();
            HostMetricBatchRow hrow;
            hrow.ts_ns         = ts;
            hrow.cpu_pct_x100  = static_cast<uint32_t>(hs.cpu_util_percent * 100.0);
            hrow.ram_used_mib  = hs.ram_used_mib;
            hrow.ram_total_mib = hs.ram_total_mib;
            host_batch_.push(hrow);
        }

        ++samples_since_flush;

        if (samples_since_flush >= kMetricBatchSize || batch_.needsFlush()) {
            flushBatches_(batch_context_);
            samples_since_flush = 0;
        }

        std::this_thread::sleep_until(next_wake_time);
    }

    // Flush any remaining samples accumulated before deactivation.
    flushBatches_(batch_context_);
}

void Sampler::flushBatches_(
    const SegmentWriteLease& context) {
    if (!context || !context->logger) return;
    uint64_t logical_rows = 0;
    if (!batch_.empty()) {
        logical_rows += batch_.rows().size();
        context->logger->write(model::DeviceMetricBatchModel(
            batch_, context->session_id, ++batch_id_));
        batch_.clear();
    }
    if (!host_batch_.empty()) {
        logical_rows += host_batch_.rows().size();
        context->logger->write(model::HostMetricBatchModel(
            host_batch_, context->session_id, ++host_batch_id_));
        host_batch_.clear();
    }
    if (logical_rows > 0 && row_observer_) {
        const int64_t steady_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count();
        row_observer_(context->segment_index, logical_rows, steady_ns,
                      detail::GetTimestampNs());
    }
    batch_context_.reset();
}

}  // namespace gpufl

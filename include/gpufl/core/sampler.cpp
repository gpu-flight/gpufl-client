#include "gpufl/core/sampler.hpp"

#include "gpufl/core/common.hpp"
#include "gpufl/core/logger/logger.hpp"
#include "gpufl/core/model/batch_models.hpp"

namespace gpufl {
Sampler::Sampler() = default;
Sampler::~Sampler() { stop(); }

void Sampler::start(std::string appName, std::string sessionId,
                    std::shared_ptr<Logger> logger,
                    std::shared_ptr<ISystemCollector<DeviceSample>> collector,
                    const int sampleIntervalMs, std::string name,
                    HostCollector* hostCollector) {
    stop();

    appName_        = std::move(appName);
    logger_         = std::move(logger);
    sessionId_      = std::move(sessionId);
    collector_      = std::move(collector);
    host_collector_ = hostCollector;
    intervalMs_     = sampleIntervalMs;
    name_           = std::move(name);
    batch_.clear();
    batch_id_      = 0;
    host_batch_.clear();
    host_batch_id_ = 0;

    if (!logger_ || !collector_ || intervalMs_ <= 0) return;

    running_.store(true);

    {
        std::lock_guard lk(mu_);
        th_ = std::thread([this] { runLoop_(); });
    }
}

void Sampler::stop() {
    running_.store(false, std::memory_order_release);

    std::thread toJoin;
    {
        std::lock_guard lk(mu_);
        toJoin = std::move(th_);
    }

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

        for (const DeviceSample& d : collector_->sampleAll()) {
            DeviceMetricBatchRow row;
            row.ts_ns     = ts;
            row.device_id = d.device_id;
            row.gpu_util  = d.gpu_util;
            row.mem_util  = d.mem_util;
            row.temp_c    = d.temp_c;
            row.power_mw  = d.power_mw;
            row.used_mib  = d.used_mib;
            row.clock_sm  = d.clock_sm;
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
            logger_->write(model::DeviceMetricBatchModel(
                batch_, sessionId_, ++batch_id_));
            batch_.clear();
            if (!host_batch_.empty()) {
                logger_->write(model::HostMetricBatchModel(
                    host_batch_, sessionId_, ++host_batch_id_));
                host_batch_.clear();
            }
            samples_since_flush = 0;
        }

        std::this_thread::sleep_until(next_wake_time);
    }

    // Flush any remaining samples accumulated before stop()
    if (!batch_.empty()) {
        logger_->write(
            model::DeviceMetricBatchModel(batch_, sessionId_, ++batch_id_));
        batch_.clear();
    }
    if (!host_batch_.empty()) {
        logger_->write(
            model::HostMetricBatchModel(host_batch_, sessionId_, ++host_batch_id_));
        host_batch_.clear();
    }
}

}  // namespace gpufl

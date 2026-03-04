#include "gpufl/core/sampler.hpp"

#include "gpufl/core/common.hpp"
#include "gpufl/core/logger.hpp"

namespace gpufl {
Sampler::Sampler() = default;
Sampler::~Sampler() { stop(); }

void Sampler::start(std::string appName, std::string sessionId,
                    std::shared_ptr<Logger> logger,
                    std::shared_ptr<ISystemCollector<DeviceSample>> collector,
                    const int sampleIntervalMs, std::string name) {
    stop();

    appName_ = std::move(appName);
    logger_ = std::move(logger);
    sessionId_ = std::move(sessionId);
    collector_ = std::move(collector);
    intervalMs_ = sampleIntervalMs;
    name_ = std::move(name);

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
        // Avoid joining self (causes "resource deadlock would occur")
        if (toJoin.get_id() == std::this_thread::get_id()) {
            // If stop() is called from within the sampler thread, detach
            // instead.
            toJoin.detach();
        } else {
            toJoin.join();
        }
    }
}

void Sampler::runLoop_() const {
    using clock = std::chrono::steady_clock;
    auto interval = std::chrono::milliseconds(intervalMs_);
    auto next_wake_time = clock::now();
    while (running_.load()) {
        next_wake_time += interval;
        const int64_t ts = detail::GetTimestampNs();
        SystemSampleEvent e;
        e.pid = detail::GetPid();
        e.app = appName_;
        e.session_id = sessionId_;
        e.name = name_;
        e.ts_ns = ts;
        e.devices = collector_->sampleAll();
        logger_->logSystemSample(e);

        std::this_thread::sleep_until(next_wake_time);
    }
}
}  // namespace gpufl

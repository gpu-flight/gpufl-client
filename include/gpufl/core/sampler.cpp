#include "gpufl/core/sampler.hpp"
#include "gpufl/core/logger.hpp"
#include "gpufl/core/common.hpp"

namespace gpufl {
    Sampler::Sampler() = default;
    Sampler::~Sampler() { stop(); }

    void Sampler::start(std::string appName,
                        std::shared_ptr<Logger> logger,
                        std::shared_ptr<ISystemCollector> collector,
                        const int sampleIntervalMs,
                        std::string name) {
        stop();

        appName_ = std::move(appName);
        logger_ = std::move(logger);
        collector_ = std::move(collector);
        intervalMs_ = sampleIntervalMs;
        name_ = std::move(name);

        if  (!logger_ || !collector_ || intervalMs_ <= 0) return;

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
                // If stop() is called from within the sampler thread, detach instead.
                toJoin.detach();
            } else {
                toJoin.join();
            }
        }
    }

    void Sampler::runLoop_() const {
        while (running_.load()) {
            const int64_t ts = detail::getTimestampNs();
            SystemSampleEvent e;
            e.pid = detail::getPid();
            e.app = appName_;
            e.name = name_;
            e.tsNs = ts;
            e.devices = collector_->sampleAll();
            logger_->logSystemSample(e);


            std::this_thread::sleep_for(std::chrono::milliseconds(intervalMs_));
        }
    }
}

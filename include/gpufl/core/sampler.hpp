#pragma once
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "gpufl/backends/host_collector.hpp"
#include "gpufl/core/batch_buffer.hpp"
#include "gpufl/core/events.hpp"

namespace gpufl {
class Logger;

template <typename T>
class ISystemCollector {
   public:
    virtual ~ISystemCollector() = default;

    virtual std::vector<T> sampleAll() = 0;
};

class Sampler {
   public:
    Sampler();
    ~Sampler();

    void start(std::string appName, std::string sessionId,
               std::shared_ptr<Logger> logger,
               std::shared_ptr<ISystemCollector<DeviceSample>> collector,
               int sampleIntervalMs, std::string name,
               HostCollector* hostCollector = nullptr);

    void stop();

    bool running() const { return running_.load(); }

   private:
    static constexpr int kMetricBatchSize = 4;  // flush every N samples

    void runLoop_();

    std::atomic<bool> running_{false};
    std::mutex mu_;
    std::thread th_;

    std::string appName_;
    std::string sessionId_;
    std::shared_ptr<Logger> logger_;
    std::shared_ptr<ISystemCollector<DeviceSample>> collector_;
    HostCollector* host_collector_{nullptr};  // non-owning
    std::string name_;
    int intervalMs_{0};

    BatchBuffer<DeviceMetricBatchRow> batch_;
    uint64_t batch_id_ = 0;

    BatchBuffer<HostMetricBatchRow> host_batch_;
    uint64_t host_batch_id_ = 0;
};
}  // namespace gpufl
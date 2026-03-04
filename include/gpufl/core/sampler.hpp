#pragma once
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

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
               int sampleIntervalMs, std::string name);

    void stop();

    bool running() const { return running_.load(); }

   private:
    void runLoop_() const;

    std::atomic<bool> running_{false};
    std::mutex mu_;
    std::thread th_;

    std::string appName_;
    std::string sessionId_;
    std::shared_ptr<Logger> logger_;
    std::shared_ptr<ISystemCollector<DeviceSample>> collector_;
    std::string name_;
    int intervalMs_{0};
};
}  // namespace gpufl
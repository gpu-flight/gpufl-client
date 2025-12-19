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

    class ISystemCollector {
    public:
        virtual ~ISystemCollector() = default;

        virtual std::vector<DeviceSample> sampleAll() = 0;
    };

    class Sampler {
    public:
        Sampler();
        ~Sampler();

        void start(std::string appName,
                   std::shared_ptr<Logger> logger,
                   std::shared_ptr<ISystemCollector> collector,
                   int sampleIntervalMs,
                   std::string name);

        void stop();

        bool running() const { return running_.load(); }
    private:
        void runLoop_() const;

        std::atomic<bool> running_{false};
        std::mutex mu_;
        std::thread th_;

        std::string appName_;
        std::shared_ptr<Logger> logger_;
        std::shared_ptr<ISystemCollector> collector_;
        std::string name_;
        int intervalMs_{0};
    };
}
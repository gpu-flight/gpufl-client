#pragma once
#include <memory>
#include <mutex>
#include <string>

#include "gpufl/backends/host_collector.hpp"
#include "gpufl/core/sampler.hpp"

namespace gpufl {
class Logger;

struct Runtime {
    std::string app_name;
    std::string session_id;
    std::shared_ptr<Logger> logger;
    std::shared_ptr<ISystemCollector<DeviceSample>> collector;
    std::unique_ptr<HostCollector> host_collector;
    std::unique_ptr<ISystemCollector<CudaStaticDeviceInfo>> cuda_collector;

    // background system sampling
    std::atomic<bool> system_sampling{false};
    Sampler sampler;
    std::mutex system_mu;
    std::thread system_thread;
    int system_interval_ms{0};
};

Runtime* runtime();
void set_runtime(std::unique_ptr<Runtime> rt);
}  // namespace gpufl

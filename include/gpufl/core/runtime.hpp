#pragma once
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include "gpufl/backends/host_collector.hpp"
#include "gpufl/core/backend_interfaces.hpp"
#include "gpufl/core/sampler.hpp"

namespace gpufl {
class Logger;

struct Runtime {
    std::string app_name;
    std::string session_id;
    // Present only for launcher-owned long-running segmentation. The
    // coordinator is introduced in a later slice; segment zero still carries
    // these values in job_start so the wire contract is testable end to end.
    std::string run_id;
    uint32_t segment_index = 0;
    int64_t segment_every_ms = 0;
    uint64_t segment_max_rows = 0;
    std::shared_ptr<Logger> logger;
    std::shared_ptr<IUnifiedGpuCollector> unified_gpu_collector;
    std::shared_ptr<ISystemCollector<DeviceSample>> collector;
    std::unique_ptr<HostCollector> host_collector;
    std::shared_ptr<IGpuStaticInfoCollector> static_info_collector;

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

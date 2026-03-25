#pragma once

#include <memory>
#include <vector>

#include "gpufl/core/events.hpp"
#include "gpufl/core/sampler.hpp"

namespace gpufl {

class IGpuStaticInfoCollector {
   public:
    virtual ~IGpuStaticInfoCollector() = default;
    virtual std::vector<GpuStaticDeviceInfo> sampleAll() = 0;
};

struct BackendCollectors {
    std::shared_ptr<ISystemCollector<DeviceSample>> telemetry_collector;
    std::unique_ptr<IGpuStaticInfoCollector> static_info_collector;
};

}  // namespace gpufl

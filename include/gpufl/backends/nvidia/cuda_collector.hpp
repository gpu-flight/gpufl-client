#pragma once

#include <vector>

#include "gpufl/core/events.hpp"
#include "gpufl/core/sampler.hpp"

namespace gpufl::nvidia {
class CudaCollector : public ISystemCollector<GpuStaticDeviceInfo> {
   public:
    CudaCollector();
    ~CudaCollector() override;

    std::vector<GpuStaticDeviceInfo> sampleAll() override;
};
}  // namespace gpufl::nvidia

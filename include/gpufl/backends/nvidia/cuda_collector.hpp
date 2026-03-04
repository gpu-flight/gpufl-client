#pragma once

#include <vector>

#include "gpufl/core/events.hpp"
#include "gpufl/core/sampler.hpp"

namespace gpufl::nvidia {
class CudaCollector : public ISystemCollector<CudaStaticDeviceInfo> {
   public:
    CudaCollector();
    ~CudaCollector() override;

    std::vector<CudaStaticDeviceInfo> sampleAll() override;
};
}  // namespace gpufl::nvidia

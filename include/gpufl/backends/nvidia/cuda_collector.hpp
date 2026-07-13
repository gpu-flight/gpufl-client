#pragma once

#include <vector>

#include "gpufl/core/events.hpp"
#include "gpufl/core/sampler.hpp"

namespace gpufl::nvidia {
struct CudaRuntimeVersions {
    int driver = 0;
    int runtime = 0;
};

class CudaCollector : public ISystemCollector<GpuStaticDeviceInfo> {
   public:
    CudaCollector();
    ~CudaCollector() override;

    std::vector<GpuStaticDeviceInfo> sampleAll() override;
};

// Numeric CUDA versions use CUDA's native encoding (for example 13030 for
// CUDA 13.3). A zero field means the corresponding query was unavailable.
CudaRuntimeVersions QueryCudaRuntimeVersions();
}  // namespace gpufl::nvidia

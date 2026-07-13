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

// Adds capabilities that require the CUDA Driver API. This is intentionally
// separate from sampleAll() so injection-time telemetry remains Runtime-API
// only; gpufl info is the expected caller.
void EnrichCudaInfoCapabilities(std::vector<GpuStaticDeviceInfo>& devices);

// Numeric CUDA versions use CUDA's native encoding (for example 13030 for
// CUDA 13.3). A zero field means the corresponding query was unavailable.
CudaRuntimeVersions QueryCudaRuntimeVersions();
}  // namespace gpufl::nvidia

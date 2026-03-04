#pragma once
#include <string>
#include <vector>

#include "gpufl/core/sampler.hpp"

namespace gpufl::amd {

// Future: implement via ROCm SMI / rocm_smi_lib.
class RocmCollector : public ISystemCollector<DeviceSample> {
   public:
    RocmCollector();
    ~RocmCollector() override;

    std::vector<DeviceSample> sampleAll() override;
};

}  // namespace gpufl::amd

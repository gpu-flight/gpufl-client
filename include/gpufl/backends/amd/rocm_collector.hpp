#pragma once
#include "gpufl/core/sampler.hpp"

#include <string>
#include <vector>

namespace gpufl::amd {

    // Future: implement via ROCm SMI / rocm_smi_lib.
    class RocmCollector : public ISystemCollector {
    public:
        RocmCollector();
        ~RocmCollector() override;

        std::string devicesInventoryJson() override;
        std::vector<DeviceSample> sampleAll() override;
    };

} // namespace gpufl::amd

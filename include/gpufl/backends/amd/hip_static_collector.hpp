#pragma once

#include <string>
#include <vector>

#include "gpufl/core/events.hpp"
#include "gpufl/core/sampler.hpp"

namespace gpufl::amd {

class HipStaticCollector : public ISystemCollector<GpuStaticDeviceInfo> {
   public:
    HipStaticCollector();
    ~HipStaticCollector() override;

    std::vector<GpuStaticDeviceInfo> sampleAll() override;
    static bool IsAvailable(std::string* reason = nullptr);
};

}  // namespace gpufl::amd

#pragma once

#include <string>
#include <vector>

#include "gpufl/core/backend_interfaces.hpp"
#include "gpufl/core/events.hpp"

namespace gpufl::metal {

class MetalCollector : public IUnifiedGpuCollector {
   public:
    MetalCollector();
    ~MetalCollector() override;

    std::vector<DeviceSample> sampleAll() override;
    std::vector<GpuStaticDeviceInfo> sampleStaticInfo() override;

    bool canSampleTelemetry() const override { return available_; }
    bool canSampleStaticInfo() const override { return available_; }

    static bool IsAvailable(std::string* reason = nullptr);

   private:
    bool available_ = false;
};

}  // namespace gpufl::metal

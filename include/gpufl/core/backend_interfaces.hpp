#pragma once

#include <memory>
#include <vector>

#include "gpufl/core/events.hpp"
#include "gpufl/core/sampler.hpp"

namespace gpufl {

class IGpuStaticInfoCollector {
   public:
    virtual ~IGpuStaticInfoCollector() = default;
    virtual std::vector<GpuStaticDeviceInfo> sampleStaticInfo() = 0;
};

class IUnifiedGpuCollector : public ISystemCollector<DeviceSample>,
                             public IGpuStaticInfoCollector {
   public:
    ~IUnifiedGpuCollector() override = default;
    virtual bool canSampleTelemetry() const = 0;
    virtual bool canSampleStaticInfo() const = 0;
};

struct BackendCollectors {
    std::shared_ptr<IUnifiedGpuCollector> unified_collector;
    std::shared_ptr<ISystemCollector<DeviceSample>> telemetry_collector;
    std::shared_ptr<IGpuStaticInfoCollector> static_info_collector;
};

}  // namespace gpufl

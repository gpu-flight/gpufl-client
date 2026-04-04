#pragma once
#include <string>
#include <unordered_map>
#include <vector>

#include "gpufl/core/backend_interfaces.hpp"
#include "gpufl/core/events.hpp"
#include "gpufl/core/sampler.hpp"

namespace gpufl::amd {

class RocmCollector : public IUnifiedGpuCollector {
   public:
    RocmCollector();
    ~RocmCollector() override;

    std::vector<DeviceSample> sampleAll() override;
    std::vector<GpuStaticDeviceInfo> sampleStaticInfo() override;

    bool canSampleTelemetry() const override { return telemetry_initialized_; }
    bool canSampleStaticInfo() const override { return static_info_available_; }

    static bool IsAvailable(std::string* reason = nullptr);
    static bool IsTelemetryAvailable(std::string* reason = nullptr);
    static bool IsStaticInfoAvailable(std::string* reason = nullptr);

   private:
    void resolveDeviceNames();

    bool telemetry_initialized_ = false;
    bool static_info_available_ = false;
    uint32_t telemetry_device_count_ = 0;
    int static_device_count_ = 0;
    std::unordered_map<int, std::string> hip_device_names_;  // device_id -> full name from HIP
};

}  // namespace gpufl::amd

#pragma once
#include "gpufl/core/sampler.hpp"
#include "gpufl/core/events.hpp"
#include <string>
#include <vector>

#if GPUFL_ENABLE_NVIDIA && GPUFL_HAS_NVML
#include <nvml.h>

namespace gpufl::nvidia {
    class NvmlCollector : public ISystemCollector {
    public:
        NvmlCollector();
        ~NvmlCollector() override;

        std::string devicesInventoryJson() override;
        std::vector<DeviceSample> sampleAll() override;
        static bool isAvailable(std::string* reason = nullptr);

    private:
        bool initialized_ = false;
        unsigned int deviceCount_ = 0;

        static std::string nvmlErrorToString(nvmlReturn_t r);
        static unsigned long long toMiB(unsigned long long bytes);
    };
}
#else
namespace gpufl::nvidia {
    class NvmlCollector;
}
#endif

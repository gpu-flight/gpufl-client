#pragma once
#include <string>
#include <vector>

#include "gpufl/core/events.hpp"
#include "gpufl/core/sampler.hpp"

#if GPUFL_ENABLE_NVIDIA && GPUFL_HAS_NVML
#include <nvml.h>

namespace gpufl::nvidia {
struct NvLinkState {
    unsigned long long lastRxTotal = 0;
    unsigned long long lastTxTotal = 0;
    std::chrono::steady_clock::time_point lastTime;
    bool initialized = false;
};
class NvmlCollector : public ISystemCollector<DeviceSample> {
   public:
    NvmlCollector();
    ~NvmlCollector() override;

    std::vector<DeviceSample> sampleAll() override;
    static bool IsAvailable(std::string* reason = nullptr);

   private:
    bool initialized_ = false;
    unsigned int deviceCount_ = 0;

    static std::string NvmlErrorToString(nvmlReturn_t r);
    static unsigned long long ToMiB(unsigned long long bytes);

#ifdef _WIN32
    // PDH fallback for GPU utilization on WDDM where NVML returns 0%.
    void* pdh_query_ = nullptr;
    void* pdh_gpu_counter_ = nullptr;
    bool  pdh_available_ = false;
    void  initPdh_();
    void  cleanupPdh_();
    unsigned int sampleGpuUtilPdh_();
#endif
};
}  // namespace gpufl::nvidia
#else
namespace gpufl::nvidia {
class NvmlCollector;
}
#endif

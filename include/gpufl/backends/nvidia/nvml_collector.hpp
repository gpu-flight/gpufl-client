#pragma once
#include <string>
#include <vector>

#include "gpufl/core/events.hpp"
#include "gpufl/core/sampler.hpp"

#if GPUFL_ENABLE_NVIDIA && GPUFL_HAS_NVML
#include <nvml.h>

#if defined(_WIN32) && GPUFL_HAS_NVAPI
#include <nvapi.h>
#include <unordered_map>
#endif

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

#if defined(_WIN32) && GPUFL_HAS_NVAPI
    // NVAPI fallback for GPU + memory(FB) utilization on WDDM, where NVML's
    // util rates read 0%. FB (memory-controller) util has no PDH counter, so
    // NVAPI is the only source for mem_util on Windows consumer GPUs.
    bool nvapi_available_ = false;
    std::unordered_map<unsigned int, NvPhysicalGpuHandle> nvapi_by_bus_;  // PCI bus -> handle
    void initNvapi_();
    void cleanupNvapi_();
    // Fills gpu/mem (0-100) for the GPU at pciBus via NVAPI dynamic pstates.
    // Returns false if NVAPI has no data for that bus.
    bool sampleUtilNvapi_(int pciBus, unsigned int& gpu, unsigned int& mem);
#endif
};
}  // namespace gpufl::nvidia
#else
namespace gpufl::nvidia {
class NvmlCollector;
}
#endif

#include "gpufl/core/monitor_adapter.hpp"

#if GPUFL_ENABLE_NVIDIA && GPUFL_HAS_CUDA
#include "gpufl/backends/nvidia/monitor_adapter_nvidia.hpp"
#endif

namespace gpufl {

std::unique_ptr<IMonitorAdapter> CreateMonitorAdapter() {
#if GPUFL_ENABLE_NVIDIA && GPUFL_HAS_CUDA
    return std::make_unique<nvidia::NvidiaMonitorAdapter>();
#else
    return nullptr;
#endif
}

}  // namespace gpufl

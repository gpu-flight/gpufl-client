#include "gpufl/core/monitor_adapter.hpp"

#if GPUFL_ENABLE_AMD && GPUFL_HAS_ROCPROFILER_SDK
#include "gpufl/backends/amd/monitor_adapter_amd.hpp"
#endif

#if GPUFL_ENABLE_NVIDIA && GPUFL_HAS_CUDA
#include "gpufl/backends/nvidia/monitor_adapter_nvidia.hpp"
#endif

namespace gpufl {

std::unique_ptr<IMonitorAdapter> CreateMonitorAdapter(const MonitorOptions& opts) {
    switch (opts.backend_kind) {
        case MonitorBackendKind::Nvidia:
#if GPUFL_ENABLE_NVIDIA && GPUFL_HAS_CUDA
            return std::make_unique<nvidia::NvidiaMonitorAdapter>();
#else
            return nullptr;
#endif
        case MonitorBackendKind::Amd:
#if GPUFL_ENABLE_AMD && GPUFL_HAS_ROCPROFILER_SDK
            return std::make_unique<amd::AmdMonitorAdapter>();
#else
            return nullptr;
#endif
        case MonitorBackendKind::None:
            return nullptr;
        case MonitorBackendKind::Auto:
        default:
#if GPUFL_ENABLE_NVIDIA && GPUFL_HAS_CUDA
            return std::make_unique<nvidia::NvidiaMonitorAdapter>();
#elif GPUFL_ENABLE_AMD && GPUFL_HAS_ROCPROFILER_SDK
            return std::make_unique<amd::AmdMonitorAdapter>();
#else
            return nullptr;
#endif
    }
}

}  // namespace gpufl

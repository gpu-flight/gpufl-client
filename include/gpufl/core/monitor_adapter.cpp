#include "gpufl/core/monitor_adapter.hpp"

#if GPUFL_ENABLE_AMD && GPUFL_HAS_ROCPROFILER_SDK
#include "gpufl/backends/amd/monitor_adapter_amd.hpp"
#endif

#if GPUFL_ENABLE_NVIDIA && GPUFL_HAS_CUDA
#include "gpufl/backends/nvidia/monitor_adapter_nvidia.hpp"
#endif

namespace gpufl {

std::unique_ptr<IMonitorAdapter> CreateMonitorAdapter(const MonitorOptions& opts) {
    // ProfilingEngine::Monitor means telemetry-only - no CUPTI at all.
    // Return no adapter so Monitor never subscribes CUPTI callbacks or
    // enables activity kinds. Monitor::Start/Stop/Shutdown are all
    // null-adapter safe, and the collector thread + NVML telemetry
    // sampler are created independently of the adapter, so scopes and
    // GPU/host metrics still flow. This is the path that's immune to
    // every CUPTI kernel-path failure mode (per-launch overhead,
    // symbolName segfault, buffer leakage), which is why it's the default.
    if (opts.profiling_engine == ProfilingEngine::Monitor) {
        return nullptr;
    }
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

#include "gpufl/core/backend_factory.hpp"

#if GPUFL_HAS_CUDA
#include "gpufl/backends/nvidia/cuda_collector.hpp"
#endif

#if GPUFL_ENABLE_NVIDIA && GPUFL_HAS_NVML
#include "gpufl/backends/nvidia/nvml_collector.hpp"
#endif

#if GPUFL_ENABLE_AMD && (GPUFL_HAS_ROCM_SMI || GPUFL_HAS_HIP)
#include "gpufl/backends/amd/rocm_collector.hpp"
#endif

namespace gpufl {

#if GPUFL_HAS_CUDA
class NvidiaStaticInfoCollector final : public IGpuStaticInfoCollector {
   public:
    std::vector<GpuStaticDeviceInfo> sampleStaticInfo() override {
        return collector_.sampleAll();
    }

   private:
    nvidia::CudaCollector collector_;
};
#endif

BackendCollectors CreateBackendCollectors(const BackendKind backend,
                                          std::string* reasonOut) {
    if (reasonOut) reasonOut->clear();
    BackendCollectors out{};

    auto setReason = [&](const std::string& r) {
        if (reasonOut && reasonOut->empty()) *reasonOut = r;
    };

    auto tryNvml = [&]() -> std::shared_ptr<ISystemCollector<DeviceSample>> {
#if GPUFL_ENABLE_NVIDIA && GPUFL_HAS_NVML
        return std::make_shared<gpufl::nvidia::NvmlCollector>();
#else
        setReason(
            "NVIDIA telemetry not available (GPUFL_ENABLE_NVIDIA=OFF or NVML "
            "not found).");
        return nullptr;
#endif
    };

    auto tryAmdUnified = [&]() -> std::shared_ptr<IUnifiedGpuCollector> {
#if GPUFL_ENABLE_AMD && (GPUFL_HAS_ROCM_SMI || GPUFL_HAS_HIP)
        std::string reason;
        if (!gpufl::amd::RocmCollector::IsAvailable(&reason)) {
            setReason("AMD backend unavailable: " + reason);
            return nullptr;
        }
        return std::make_shared<gpufl::amd::RocmCollector>();
#else
        setReason(
            "AMD backend not available (GPUFL_ENABLE_AMD=OFF or ROCm/HIP not "
            "found).");
        return nullptr;
#endif
    };

    auto tryNvidiaStatic = [&]() -> std::shared_ptr<IGpuStaticInfoCollector> {
#if GPUFL_HAS_CUDA
        return std::make_shared<NvidiaStaticInfoCollector>();
#else
        return nullptr;
#endif
    };

    switch (backend) {
        case BackendKind::None:
            break;

        case BackendKind::Nvidia:
            out.telemetry_collector = tryNvml();
            if (!out.telemetry_collector) {
                setReason("Requested backend=nvidia but NVML is unavailable.");
            }
            out.static_info_collector = tryNvidiaStatic();
            break;

        case BackendKind::Amd:
            out.unified_collector = tryAmdUnified();
            if (!out.unified_collector) {
                setReason("Requested backend=amd but no AMD capability is available.");
            } else {
                if (out.unified_collector->canSampleTelemetry()) {
                    out.telemetry_collector = std::static_pointer_cast<
                        ISystemCollector<DeviceSample>>(out.unified_collector);
                }
                if (out.unified_collector->canSampleStaticInfo()) {
                    out.static_info_collector = std::static_pointer_cast<
                        IGpuStaticInfoCollector>(out.unified_collector);
                }
            }
            break;

        case BackendKind::Auto:
        default:
            out.telemetry_collector = tryNvml();
            if (out.telemetry_collector) {
                out.static_info_collector = tryNvidiaStatic();
                break;
            }

            out.unified_collector = tryAmdUnified();
            if (out.unified_collector) {
                if (out.unified_collector->canSampleTelemetry()) {
                    out.telemetry_collector = std::static_pointer_cast<
                        ISystemCollector<DeviceSample>>(out.unified_collector);
                }
                if (out.unified_collector->canSampleStaticInfo()) {
                    out.static_info_collector = std::static_pointer_cast<
                        IGpuStaticInfoCollector>(out.unified_collector);
                }
                break;
            }

            if (!out.telemetry_collector) {
                setReason(
                    "No GPU backend available (NVML/ROCm not compiled in or "
                    "not available).");
            }
            out.static_info_collector = tryNvidiaStatic();
            if (!out.static_info_collector && out.unified_collector &&
                out.unified_collector->canSampleStaticInfo()) {
                out.static_info_collector = std::static_pointer_cast<
                    IGpuStaticInfoCollector>(out.unified_collector);
            }
            break;
    }

    return out;
}

}  // namespace gpufl

#include "gpufl/core/backend_factory.hpp"

#if GPUFL_HAS_CUDA
#include "gpufl/backends/nvidia/cuda_collector.hpp"
#endif

#if GPUFL_ENABLE_NVIDIA && GPUFL_HAS_NVML
#include "gpufl/backends/nvidia/nvml_collector.hpp"
#endif

#if GPUFL_ENABLE_AMD && GPUFL_HAS_ROCM
#include "gpufl/backends/amd/rocm_collector.hpp"
#endif

#if GPUFL_ENABLE_AMD && GPUFL_HAS_HIP
#include "gpufl/backends/amd/hip_static_collector.hpp"
#endif

namespace gpufl {

#if GPUFL_HAS_CUDA
class NvidiaStaticInfoCollector final : public IGpuStaticInfoCollector {
   public:
    std::vector<GpuStaticDeviceInfo> sampleAll() override {
        return collector_.sampleAll();
    }

   private:
    nvidia::CudaCollector collector_;
};
#endif

#if GPUFL_ENABLE_AMD && GPUFL_HAS_HIP
class AmdStaticInfoCollector final : public IGpuStaticInfoCollector {
   public:
    std::vector<GpuStaticDeviceInfo> sampleAll() override {
        return collector_.sampleAll();
    }

   private:
    amd::HipStaticCollector collector_;
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

    auto tryRocm = [&]() -> std::shared_ptr<ISystemCollector<DeviceSample>> {
#if GPUFL_ENABLE_AMD && GPUFL_HAS_ROCM
        std::string reason;
        if (!gpufl::amd::RocmCollector::IsAvailable(&reason)) {
            setReason("AMD telemetry unavailable: " + reason);
            return nullptr;
        }
        return std::make_shared<gpufl::amd::RocmCollector>();
#else
        setReason(
            "AMD telemetry not available (GPUFL_ENABLE_AMD=OFF or ROCm not "
            "found).");
        return nullptr;
#endif
    };

    auto tryNvidiaStatic = [&]() -> std::unique_ptr<IGpuStaticInfoCollector> {
#if GPUFL_HAS_CUDA
        return std::make_unique<NvidiaStaticInfoCollector>();
#else
        return nullptr;
#endif
    };

    auto tryAmdStatic = [&]() -> std::unique_ptr<IGpuStaticInfoCollector> {
#if GPUFL_ENABLE_AMD && GPUFL_HAS_HIP
        std::string reason;
        if (!gpufl::amd::HipStaticCollector::IsAvailable(&reason)) {
            setReason("AMD static inventory unavailable: " + reason);
            return nullptr;
        }
        return std::make_unique<AmdStaticInfoCollector>();
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
            out.telemetry_collector = tryRocm();
            if (!out.telemetry_collector) {
                setReason("Requested backend=amd but ROCm is unavailable.");
            }
            out.static_info_collector = tryAmdStatic();
            break;

        case BackendKind::Auto:
        default:
            out.telemetry_collector = tryNvml();
            if (out.telemetry_collector) {
                out.static_info_collector = tryNvidiaStatic();
                break;
            }

            out.telemetry_collector = tryRocm();
            if (out.telemetry_collector) {
                out.static_info_collector = tryAmdStatic();
                break;
            }

            if (!out.telemetry_collector) {
                setReason(
                    "No GPU backend available (NVML/ROCm not compiled in or "
                    "not available).");
            }
            out.static_info_collector = tryNvidiaStatic();
            if (!out.static_info_collector) {
                out.static_info_collector = tryAmdStatic();
            }
            break;
    }

    return out;
}

}  // namespace gpufl

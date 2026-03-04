#include "gpufl/backends/nvidia/cuda_collector.hpp"

#include "gpufl/core/common.hpp"

#if GPUFL_HAS_CUDA || defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

namespace gpufl::nvidia {
CudaCollector::CudaCollector() : ISystemCollector() {}
CudaCollector::~CudaCollector() = default;

std::vector<gpufl::CudaStaticDeviceInfo> CudaCollector::sampleAll() {
    std::vector<CudaStaticDeviceInfo> devices;

#if GPUFL_HAS_CUDA || defined(__CUDACC__)
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);

    if (err == cudaSuccess && count > 0) {
        for (int i = 0; i < count; ++i) {
            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
                CudaStaticDeviceInfo info;
                info.id = i;
                info.name = prop.name;
                info.uuid = detail::UuidToString(prop.uuid.bytes);
                info.compute_major = prop.major;
                info.compute_minor = prop.minor;
                info.l2_cache_size = prop.l2CacheSize;
                info.shared_mem_per_block = prop.sharedMemPerBlock;
                info.regs_per_block = prop.regsPerBlock;
                info.multi_processor_count = prop.multiProcessorCount;
                info.warp_size = prop.warpSize;

                devices.push_back(info);
            }
        }
    }
#endif
    return devices;
}
}  // namespace gpufl::nvidia
#include "gpufl/backends/nvidia/cuda_collector.hpp"

#include "gpufl/core/common.hpp"

#if GPUFL_HAS_CUDA || defined(__CUDACC__)
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace gpufl::nvidia {
CudaCollector::CudaCollector() : ISystemCollector() {}
CudaCollector::~CudaCollector() = default;

std::vector<GpuStaticDeviceInfo> CudaCollector::sampleAll() {
    std::vector<GpuStaticDeviceInfo> devices;

#if GPUFL_HAS_CUDA || defined(__CUDACC__)
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    const bool driver_ready = cuInit(0) == CUDA_SUCCESS;

    if (err == cudaSuccess && count > 0) {
        for (int i = 0; i < count; ++i) {
            cudaDeviceProp prop{};
            if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
                GpuStaticDeviceInfo info{};
                info.id = i;
                info.name = prop.name;
                info.uuid = detail::UuidToString(prop.uuid.bytes);
                info.vendor = "NVIDIA";
                info.compute_major = prop.major;
                info.compute_minor = prop.minor;
                info.l2_cache_size = prop.l2CacheSize;
                info.shared_mem_per_block = prop.sharedMemPerBlock;
                info.regs_per_block = prop.regsPerBlock;
                info.multi_processor_count = prop.multiProcessorCount;
                info.warp_size = prop.warpSize;
                info.total_global_mem = prop.totalGlobalMem;
                info.total_const_mem = prop.totalConstMem;
                info.shared_mem_per_block_optin =
                    static_cast<int>(prop.sharedMemPerBlockOptin);
                info.shared_mem_per_multiprocessor =
                    static_cast<int>(prop.sharedMemPerMultiprocessor);
                info.regs_per_multiprocessor = prop.regsPerMultiprocessor;
                info.max_threads_per_block = prop.maxThreadsPerBlock;
                info.max_threads_per_multiprocessor =
                    prop.maxThreadsPerMultiProcessor;
                info.max_blocks_per_multiprocessor =
                    prop.maxBlocksPerMultiProcessor;
                for (int dim = 0; dim < 3; ++dim) {
                    info.max_threads_dim[dim] = prop.maxThreadsDim[dim];
                    info.max_grid_size[dim] = prop.maxGridSize[dim];
                }
                // CUDA 13 removed clockRate/memoryClockRate from
                // cudaDeviceProp; device attributes remain the supported API.
                (void)cudaDeviceGetAttribute(
                    &info.clock_rate_khz, cudaDevAttrClockRate, i);
                (void)cudaDeviceGetAttribute(
                    &info.memory_clock_rate_khz,
                    cudaDevAttrMemoryClockRate, i);
                (void)cudaDeviceGetAttribute(
                    &info.memory_bus_width_bits,
                    cudaDevAttrGlobalMemoryBusWidth, i);
                info.async_engine_count = prop.asyncEngineCount;
                info.concurrent_kernels = prop.concurrentKernels != 0;
                info.cooperative_launch = prop.cooperativeLaunch != 0;
                info.unified_addressing = prop.unifiedAddressing != 0;
                info.managed_memory = prop.managedMemory != 0;
                info.memory_pools_supported = prop.memoryPoolsSupported != 0;
                info.cluster_launch = prop.clusterLaunch != 0;

                if (driver_ready) {
                    CUdevice device{};
                    int tensor_map_supported = 0;
                    if (cuDeviceGet(&device, i) == CUDA_SUCCESS &&
                        cuDeviceGetAttribute(
                            &tensor_map_supported,
                            CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED,
                            device) == CUDA_SUCCESS) {
                        info.tensor_map_access_supported =
                            tensor_map_supported != 0;
                    }
                }

                devices.push_back(info);
            }
        }
    }
#endif
    return devices;
}

CudaRuntimeVersions QueryCudaRuntimeVersions() {
    CudaRuntimeVersions versions{};
#if GPUFL_HAS_CUDA || defined(__CUDACC__)
    (void)cudaDriverGetVersion(&versions.driver);
    (void)cudaRuntimeGetVersion(&versions.runtime);
#endif
    return versions;
}
}  // namespace gpufl::nvidia

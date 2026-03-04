#include "gpufl/backends/nvidia/cupti_utils.hpp"

#include <cuda_runtime.h>
#include <cupti.h>

#include <mutex>
#include <unordered_map>

namespace gpufl {

SmProps GetSMProps(int deviceId) {
    static std::mutex mu;
    static std::unordered_map<int, SmProps> cache;

    std::lock_guard<std::mutex> lock(mu);
    if (cache.find(deviceId) == cache.end()) {
        cudaDeviceProp prop{};
        SmProps props{};
        if (cudaGetDeviceProperties(&prop, deviceId) == cudaSuccess) {
            props.maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
            props.warpSize = prop.warpSize;
            props.regsPerSM = prop.regsPerMultiprocessor;
            props.sharedMemPerSM =
                static_cast<int>(prop.sharedMemPerMultiprocessor);
            props.maxBlocksPerSM = prop.maxBlocksPerMultiProcessor;
        } else {
            // Fallback for modern architectures (Ampere/Hopper)
            props.maxThreadsPerSM = 2048;
            props.warpSize = 32;
            props.regsPerSM = 65536;
            props.sharedMemPerSM = 49152;
            props.maxBlocksPerSM = 32;
        }
        cache[deviceId] = props;
    }
    return cache[deviceId];
}

int GetMaxThreadsPerSM(int deviceId) {
    return GetSMProps(deviceId).maxThreadsPerSM;
}

void CalculateOccupancy(LaunchMeta& meta, const void* funcPtr) {
    if (!funcPtr) return;

    int deviceId = 0;
    cudaGetDevice(&deviceId);

    int numBlocks = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks, funcPtr, meta.block_x * meta.block_y * meta.block_z,
        meta.dyn_shared);

    if (err == cudaSuccess) {
        meta.max_active_blocks = numBlocks;
        int maxThreadsPerSM = GetSMProps(deviceId).maxThreadsPerSM;
        if (maxThreadsPerSM > 0) {
            meta.occupancy =
                static_cast<float>(numBlocks *
                                   (meta.block_x * meta.block_y * meta.block_z)) /
                static_cast<float>(maxThreadsPerSM);
        }
    }
}

bool IsContextValid(CUcontext ctx) {
    if (!ctx) return false;
    CUcontext current = nullptr;
    if (cuCtxGetCurrent(&current) != CUDA_SUCCESS) return false;
    return (current == ctx);
}

}  // namespace gpufl

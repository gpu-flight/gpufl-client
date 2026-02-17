#include "gpufl/backends/nvidia/cupti_utils.hpp"
#include <cuda_runtime.h>
#include <cupti.h>
#include <mutex>
#include <unordered_map>

namespace gpufl {

    int GetMaxThreadsPerSM(int deviceId) {
        static std::mutex mu;
        static std::unordered_map<int, int> cache;

        std::lock_guard<std::mutex> lock(mu);
        if (cache.find(deviceId) == cache.end()) {
            cudaDeviceProp prop{};
            if (cudaGetDeviceProperties(&prop, deviceId) == cudaSuccess) {
                cache[deviceId] = prop.maxThreadsPerMultiProcessor;
            } else {
                return 2048; // Fallback for most modern architecture
            }
        }
        return cache[deviceId];
    }

    void CalculateOccupancy(LaunchMeta& meta, const void* funcPtr) {
        if (!funcPtr) return;

        int deviceId = 0;
        cudaGetDevice(&deviceId);

        int numBlocks = 0;
        cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &numBlocks,
            funcPtr,
            meta.blockX * meta.blockY * meta.blockZ,
            meta.dynShared
        );

        if (err == cudaSuccess) {
            meta.maxActiveBlocks = numBlocks;
            int maxThreadsPerSM = GetMaxThreadsPerSM(deviceId);
            if (maxThreadsPerSM > 0) {
                meta.occupancy = static_cast<float>(numBlocks * (meta.blockX * meta.blockY * meta.blockZ)) / 
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

} // namespace gpufl

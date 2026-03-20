#include "gpufl/backends/nvidia/cupti_utils.hpp"

#include <cuda_runtime.h>
#include <cupti.h>
#include <cupti_target.h>

#include <mutex>
#include <unordered_map>

#include "gpufl/core/debug_logger.hpp"

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

bool EnsureCudaContext(CUcontext* ctx) {
    if (!ctx) return false;


    if (*ctx && IsContextValid(*ctx)) {
        return true;
    }

    CUcontext current = nullptr;
    if (cuCtxGetCurrent(&current) == CUDA_SUCCESS && current) {
        *ctx = current;
        return true;
    }

    // Force runtime initialization so a context may be created.
    cudaFree(nullptr);
    current = nullptr;
    if (cuCtxGetCurrent(&current) == CUDA_SUCCESS && current) {
        *ctx = current;
        return true;
    }

    CUdevice dev;
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) {
        return false;
    }

    CUcontext primary = nullptr;
    if (cuDevicePrimaryCtxRetain(&primary, dev) != CUDA_SUCCESS) {
        return false;
    }
    if (cuCtxPushCurrent(primary) != CUDA_SUCCESS) {
        cuDevicePrimaryCtxRelease(dev);
        return false;
    }

    *ctx = primary;
    return true;
}

std::string GetCurrentDeviceName() {
    CUdevice dev;
    if (cuCtxGetDevice(&dev) != CUDA_SUCCESS) {
        return "Unknown Device";
    }
    char nameBuf[256]{};
    if (cuDeviceGetName(nameBuf, sizeof(nameBuf), dev) != CUDA_SUCCESS) {
        return "Unknown Device";
    }
    return std::string(nameBuf);
}

bool LogCuptiErrorIfFailedImpl(const char* scope, const char* op,
                               CUptiResult err, const char* file, int line) {
    if (err == CUPTI_SUCCESS) return false;
    const char* errStr = nullptr;
    cuptiGetResultString(err, &errStr);
    DebugLogger::error("[GPUFL-ERROR] ", (file ? file : "unknown"), ":", line,
                       ": [", (scope ? scope : "CUPTI"), "] ",
                       (op ? op : "operation"), " failed: ",
                       (errStr ? errStr : "unknown"), " (",
                       static_cast<int>(err), ")");
    return true;
}

const char* getChipName(uint32_t deviceId) {
    CUpti_Device_GetChipName_Params chipNameParams = {
        CUpti_Device_GetChipName_Params_STRUCT_SIZE};
    chipNameParams.deviceIndex = deviceId;
    CUptiResult res = cuptiDeviceGetChipName(&chipNameParams);
    if (LogCuptiErrorIfFailed("CUPTI", "cuptiDeviceGetChipName", res)) {
        return "";
    }
    return chipNameParams.pChipName;
}

}  // namespace gpufl

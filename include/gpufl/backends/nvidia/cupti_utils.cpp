#include "gpufl/backends/nvidia/cupti_utils.hpp"

#include <cuda_runtime.h>
#include <cupti.h>
#include <cupti_target.h>

#include <mutex>
#include <unordered_map>

#include "gpufl/core/debug_logger.hpp"

namespace gpufl {

SmProps GetSMProps(int deviceId) {
    // Process-lifetime caches avoid teardown-order races with CUPTI/injection
    // shutdown paths that can still query device properties during atexit.
    static auto* mu = new std::mutex;
    static auto* cache = new std::unordered_map<int, SmProps>;

    std::lock_guard lock(*mu);
    if (cache->find(deviceId) == cache->end()) {
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
        (*cache)[deviceId] = props;
    }
    return (*cache)[deviceId];
}

int GetMaxThreadsPerSM(const int deviceId) {
    return GetSMProps(deviceId).maxThreadsPerSM;
}

ComputeCapability GetComputeCapability(int deviceId) {
    // Same process-lifetime cache rationale as GetSMProps.
    static auto* mu = new std::mutex;
    static auto* cache = new std::unordered_map<int, ComputeCapability>;

    std::lock_guard lock(*mu);
    auto it = cache->find(deviceId);
    if (it != cache->end()) return it->second;

    ComputeCapability cc{};
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, deviceId) == cudaSuccess) {
        cc.major = prop.major;
        cc.minor = prop.minor;
    }
    (*cache)[deviceId] = cc;
    return cc;
}

uint32_t GetCuptiVersion() {
    static std::once_flag once;
    static uint32_t version = 0;
    std::call_once(once, [] {
        uint32_t v = 0;
        if (cuptiGetVersion(&v) == CUPTI_SUCCESS) version = v;
    });
    return version;
}

void CalculateOccupancy(LaunchMeta& meta, const void* funcPtr) {
    if (!funcPtr) return;

    int deviceId = 0;
    cudaGetDevice(&deviceId);

    int numBlocks = 0;
    const cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks, funcPtr, meta.block_x * meta.block_y * meta.block_z,
        meta.dyn_shared);

    if (err == cudaSuccess) {
        meta.max_active_blocks = numBlocks;
        const int maxThreadsPerSM = GetSMProps(deviceId).maxThreadsPerSM;
        if (maxThreadsPerSM > 0) {
            meta.occupancy =
                static_cast<float>(numBlocks *
                                   (meta.block_x * meta.block_y * meta.block_z)) /
                static_cast<float>(maxThreadsPerSM);
        }
    }
}

bool IsContextValid(const CUcontext ctx) {
    if (!ctx) return false;
    CUcontext current = nullptr;
    if (cuCtxGetCurrent(&current) != CUDA_SUCCESS) return false;
    return current == ctx;
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
    DebugLogger::error("[GPUFL-ERROR] ", file ? file : "unknown", ":", line,
                       ": [", scope ? scope : "CUPTI", "] ",
                       op ? op : "operation", " failed: ",
                       errStr ? errStr : "unknown", " (",
                       static_cast<int>(err), ")");
    return true;
}

const char* getChipName(const uint32_t deviceId) {
    CUpti_Device_GetChipName_Params chipNameParams = {
        CUpti_Device_GetChipName_Params_STRUCT_SIZE};
    chipNameParams.deviceIndex = deviceId;
    if (const CUptiResult res = cuptiDeviceGetChipName(&chipNameParams);
        LogCuptiErrorIfFailed("CUPTI", "cuptiDeviceGetChipName", res)) {
        return "";
    }
    return chipNameParams.pChipName;
}

}  // namespace gpufl

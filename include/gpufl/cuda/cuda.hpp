#pragma once

#include <string>
#include <cstdint>

#if GPUFL_HAS_CUDA
#include <cuda_runtime.h>
#include <mutex>
#include <sstream>
#include "gpufl/gpufl.hpp"

namespace gpufl::cuda {

    std::string dim3ToString(dim3 v);

    const char* getCudaErrorString(cudaError_t error);

    struct PendingKernel {
        std::string name;
        std::string tag;

        // Launch Params (for logging later)
        std::string grid;
        std::string block;
        int numRegs = 0;
        size_t staticShared = 0;
        int dynShared = 0;
        float occupancy = 0.0f;
        int maxActiveBlocks = 0;
        int deviceId = 0;

        std::size_t localBytes = 0;
        std::size_t constBytes = 0;

        // Timing & Hardware Handles
        int64_t cpuDispatchNs = 0; // When we called the macro
        cudaEvent_t startEvent = nullptr;
        cudaEvent_t stopEvent = nullptr;
    };

    void initEventPool(size_t initialSize = 128);
    void freeEventPool();

    std::pair<cudaEvent_t, cudaEvent_t> getEventPair();
    void returnEventPair(cudaEvent_t start, cudaEvent_t stop);

    void pushPendingKernel(PendingKernel&& k);

    void processPendingKernels();

    const cudaDeviceProp& getDevicePropsCached(int deviceId);

}
#else
#endif

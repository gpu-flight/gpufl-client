#pragma once
#include <gtest/gtest.h>

#if GPUFL_HAS_CUDA
#include <cuda_runtime.h>
#endif

// Helper to check if we are on an NVIDIA machine
inline bool isNvidiaGpuAvailable() {
#if GPUFL_HAS_CUDA
    int deviceCount = 0;
    // We use cudaGetDeviceCount because it's lightweight and standard
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    // If CUDA fails or no devices found, return false
    if (error != cudaSuccess || deviceCount == 0) {
        // Optional: Reset error so it doesn't pollute logs
        cudaGetLastError();
        return false;
    }
    return true;
#else
    return false;
#endif
}

// Macro to skip test if GPU is missing
#define SKIP_IF_NO_CUDA()                                                 \
    if (!isNvidiaGpuAvailable()) {                                        \
        GTEST_SKIP() << "No NVIDIA GPU detected. Skipping backend test."; \
    }

#pragma once
#include <gtest/gtest.h>

#if GPUFL_HAS_CUDA
#include <cuda_runtime.h>
#endif

#if GPUFL_ENABLE_NVIDIA && GPUFL_HAS_CUPTI
#include "gpufl/backends/nvidia/cupti_utils.hpp"
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

#if GPUFL_ENABLE_NVIDIA && GPUFL_HAS_CUPTI
/** Returns the compute capability of device 0, or {0, 0} if unavailable. */
inline gpufl::ComputeCapability GetTestDeviceCC() {
    return gpufl::GetComputeCapability(0);
}

/**
 * Skip the test if the attached GPU is below the given compute capability.
 * Safe to call after SKIP_IF_NO_CUDA() — relies on the CUDA runtime being up.
 */
#define SKIP_IF_CC_BELOW(major, minor)                                         \
    do {                                                                       \
        auto _cc = GetTestDeviceCC();                                          \
        if (!_cc.atLeast((major), (minor))) {                                  \
            GTEST_SKIP() << "Skipping: requires sm_" << (major) << (minor)     \
                         << " (got sm_" << _cc.major << _cc.minor << ")";      \
        }                                                                      \
    } while (0)
#endif  // GPUFL_ENABLE_NVIDIA && GPUFL_HAS_CUPTI

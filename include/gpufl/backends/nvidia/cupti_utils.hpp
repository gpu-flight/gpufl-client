#pragma once

#include <cuda_runtime.h>

#include "gpufl/backends/nvidia/cupti_common.hpp"

namespace gpufl {

struct SmProps {
    int maxThreadsPerSM;  // prop.maxThreadsPerMultiProcessor
    int warpSize;         // prop.warpSize (always 32 for NVIDIA)
    int regsPerSM;        // prop.regsPerMultiprocessor
    int sharedMemPerSM;   // prop.sharedMemPerMultiprocessor
    int maxBlocksPerSM;   // prop.maxBlocksPerMultiProcessor
};

/**
 * @brief Gets SM properties for a given device (cached).
 */
SmProps GetSMProps(int deviceId);

/**
 * @brief Gets the maximum number of threads per SM for a given device.
 */
int GetMaxThreadsPerSM(int deviceId);

/**
 * @brief Calculates the occupancy for a given kernel launch.
 */
void CalculateOccupancy(LaunchMeta& meta, const void* funcPtr);

/**
 * @brief Checks if a CUDA context is valid.
 */
bool IsContextValid(CUcontext ctx);

}  // namespace gpufl

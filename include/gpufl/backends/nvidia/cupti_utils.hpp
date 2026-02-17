#pragma once

#include <cuda_runtime.h>
#include "gpufl/backends/nvidia/cupti_common.hpp"

namespace gpufl {

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

} // namespace gpufl

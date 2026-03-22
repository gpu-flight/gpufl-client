#pragma once

#include <cuda_runtime.h>

#include <string>

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

/**
 * @brief Ensures a valid CUDA context exists and stores it in @p ctx.
 * Attempts current context, runtime init, then primary context fallback.
 */
bool EnsureCudaContext(CUcontext* ctx);

/**
 * @brief Returns the current CUDA device name, or "Unknown Device" on failure.
 */
std::string GetCurrentDeviceName();

/**
 * @brief Logs a formatted CUPTI error when err != CUPTI_SUCCESS.
 * @return true if an error was logged.
 */
bool LogCuptiErrorIfFailedImpl(const char* scope, const char* op,
                               CUptiResult err, const char* file, int line);

#define LogCuptiErrorIfFailed(scope, op, err) \
    ::gpufl::LogCuptiErrorIfFailedImpl((scope), (op), (err), __FILE__, __LINE__)

/**
 * get chipName
 * @param deviceId
 * @return
 */
const char* getChipName(uint32_t deviceId);

}  // namespace gpufl

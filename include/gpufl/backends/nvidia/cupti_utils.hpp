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
 * CUDA compute capability (e.g. {8, 6} for Ampere RTX 3090, {12, 0} for
 * Blackwell RTX 5060). Used to gate profiling features that have very
 * different overhead characteristics across GPU generations.
 */
struct ComputeCapability {
    int major = 0;
    int minor = 0;
    /** Returns true if this GPU is at least (majorNeeded, minorNeeded). */
    bool atLeast(int majorNeeded, int minorNeeded) const {
        if (major != majorNeeded) return major > majorNeeded;
        return minor >= minorNeeded;
    }
    bool valid() const { return major > 0; }
};

/**
 * @brief Returns the compute capability of the given CUDA device (cached).
 * Returns {0, 0} if the query fails.
 */
ComputeCapability GetComputeCapability(int deviceId);

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

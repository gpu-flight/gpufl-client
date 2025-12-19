#pragma once

#include <string>
#include <cstdint>

#if GPUFL_HAS_CUDA
#include <cuda_runtime.h>
#include <sstream>
#include "gpufl/gpufl.hpp"

namespace gpufl::cuda {
    const cudaDeviceProp& getDevicePropsCached(int deviceId);

    std::string dim3ToString(dim3 v);

    const char* getCudaErrorString(cudaError_t error);

    template <typename T> inline const cudaFuncAttributes& get_kernel_static_attrs(T kernel) {
        static const cudaFuncAttributes attrs = [kernel](){
            cudaFuncAttributes a = {};
            cudaFuncGetAttributes(&a, kernel);
            return a;
        }();
        return attrs;
    }

    class KernelMonitor {
    public:
        KernelMonitor(std::string name,
                               std::string tag = "",
                               std::string grid = "",
                               std::string block = "",
                               int dynShared = 0,
                               int numRegs = 0,
                               size_t staticShared = 0,
                               size_t localBytes = 0,
                               size_t constBytes = 0,
                               float occupancy = 0.0f,
                               int maxActiveBlocks = 0);

        ~KernelMonitor();

        void setError(std::string err) {
            error_ = std::move(err);
        }

        static const char* getCudaErrorString(const cudaError_t error) {
            return ::cudaGetErrorString(error);
        }

        KernelMonitor(const KernelMonitor&) = delete;
        KernelMonitor& operator=(const KernelMonitor&) = delete;

    private:
        std::string name_;
        std::string tag_;
        int pid_;
        int64_t startTs_;
        std::string error_ = "Success";
    };

}
#else
#endif

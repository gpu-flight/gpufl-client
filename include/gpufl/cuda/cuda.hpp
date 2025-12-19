#pragma once

#include <string>
#include <cstdint>

#if GPUFL_HAS_CUDA
#include <cuda_runtime.h>
#include <sstream>
#include "gpufl/gpufl.hpp"

namespace gpufl::cuda {

    std::string dim3ToString(dim3 v);

    void logKernelEvent(const std::string& kernelName,
                        int64_t ts_start_ns,
                        int64_t ts_end_ns,
                        dim3 grid,
                        dim3 block,
                        int dyn_shared_bytes,
                        const std::string& cuda_error,
                        const cudaFuncAttributes& attrs,
                        const std::string& tag = "");

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
                               size_t constBytes = 0);

        ~KernelMonitor();

        void setError(std::string err) {
            error_ = std::move(err);
        }

        static std::string dim3ToString(dim3 v) {
            std::ostringstream oss;
            oss << "(" << v.x << "," << v.y << "," << v.z << ")";
            return oss.str();
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

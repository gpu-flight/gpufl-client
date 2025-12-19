#pragma once

#include <string>
#include <cstdint>

#if GPUFL_HAS_CUDA
#include <cuda_runtime.h>

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

}
#else
#endif

#include "common/test_kernel.hpp"

#include <cuda_runtime.h>

namespace {

/**
 * FMA-bound kernel. The compiler cannot constant-fold this because the
 * accumulator feeds back into itself and the final value is written to
 * device memory.
 */
__global__ void GpuflTestKernel(float* out, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = static_cast<float>(tid) * 1e-7f;
    #pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        acc = __fmaf_rn(acc, 1.0000001f, 1e-7f);
    }
    out[tid] = acc;
}

}  // namespace

namespace gpufl::test {

void RunTestKernel(int iters, int blocks, int threads) {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        cudaGetLastError();  // clear
        return;
    }

    const size_t n = static_cast<size_t>(blocks) * static_cast<size_t>(threads);
    float* d_out = nullptr;
    if (cudaMalloc(&d_out, n * sizeof(float)) != cudaSuccess) {
        cudaGetLastError();
        return;
    }

    GpuflTestKernel<<<blocks, threads>>>(d_out, iters);
    cudaDeviceSynchronize();
    cudaFree(d_out);
}

}  // namespace gpufl::test

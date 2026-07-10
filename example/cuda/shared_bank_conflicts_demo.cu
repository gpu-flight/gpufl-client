// GPUFlight shared-memory bank-conflict demonstration.
//
// RangeProfilerKernelReplay reports each template instantiation separately:
//   STRIDE=1  -> adjacent lanes use adjacent banks (conflict-free)
//   STRIDE=32 -> every lane uses a different word in the same bank (32-way)

#include <cstdio>

#include <cuda_runtime.h>

#include "gpufl/gpufl.hpp"

template <int STRIDE>
__global__ void sharedBankAccess(float* output, int iterations) {
    extern __shared__ volatile float tile[];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warpBase = warp * 1024;
    const int sharedIndex = warpBase + lane * STRIDE;

    tile[sharedIndex] = static_cast<float>(lane + 1);
    __syncthreads();

    float sum = 0.0f;
    for (int i = 0; i < iterations; ++i) {
        sum += tile[sharedIndex];
    }
    output[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

int main() {
    gpufl::InitOptions opts;
    opts.app_name = "shared_bank_conflicts_demo";
    opts.log_path = "shared_bank_conflicts_demo";
    opts.enable_debug_output = true;
    opts.profiling_engine = gpufl::ProfilingEngine::RangeProfilerKernelReplay;
    if (!gpufl::init(opts)) {
        std::fprintf(stderr, "GPUFlight initialization failed\n");
        return 1;
    }

    constexpr int blocks = 128;
    constexpr int threads = 256;
    constexpr int iterations = 256;
    constexpr size_t sharedBytes = 8 * 1024 * sizeof(float);

    float* output = nullptr;
    cudaMalloc(&output, blocks * threads * sizeof(float));

    sharedBankAccess<1><<<blocks, threads, sharedBytes>>>(output, iterations);
    sharedBankAccess<32><<<blocks, threads, sharedBytes>>>(output, iterations);
    std::printf("Kernel replay range 0: stride 1 (conflict-free)\n");
    std::printf("Kernel replay range 1: stride 32 (deliberate 32-way conflict)\n");
    const cudaError_t result = cudaDeviceSynchronize();

    cudaFree(output);
    gpufl::shutdown();
    gpufl::generateReport();

    if (result != cudaSuccess) {
        std::fprintf(stderr, "CUDA execution failed: %s\n",
                     cudaGetErrorString(result));
        return 1;
    }
    return 0;
}

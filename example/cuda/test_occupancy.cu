#include <iostream>
#include <cuda_runtime.h>
#include "gpufl/gpufl.hpp"
#include "gpufl/core/common.hpp"

// A simple kernel to test
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    // [2] Initialize GPUFL
    if (!gpufl::init({"OccupancyTest", "gpufl_test.log", 10})) {
        std::cerr << "ERROR: Failed to initialize GPUFL" << std::endl;
        return 1;
    }
    std::cout << "GPUFL initialized successfully" << std::endl;
    gpufl::systemStart();
    const int N = 1000000;
    const int bytes = N * sizeof(float);

    // Allocate Memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // --- Test 1: High Occupancy (Standard Block Size) ---
    // 256 threads is usually a "sweet spot" for occupancy
    {
        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        std::cout << "Launching Kernel with 256 threads (Expect High Occupancy)..." << std::endl;

        // Use standard <<<>>>
        vector_add<<<blocks, threads>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
    }

    // --- Test 2: Low Occupancy (Tiny Block Size) ---
    // 32 threads (1 warp) per block often wastes SM resources due to block limit caps
    {
        int threads = 32;
        int blocks = (N + threads - 1) / threads;

        std::cout << "Launching Kernel with 32 threads (Expect Lower Occupancy)..." << std::endl;

        vector_add<<<blocks, threads>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
    }

    // [3] Shutdown
    gpufl::shutdown();

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    std::cout << "Done! Check gpufl_test.log for 'occupancy' fields." << std::endl;
    return 0;
}
#include <iostream>
#include <cuda_runtime.h>
#include "gpufl/gpufl.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/monitor.hpp"

__global__
void vectorAdd(int* a, int* b, int* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__
void vectorMul(int* a, int* b, int* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

__global__
void vectorScale(int* a, int scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int val = a[idx];
        for (int i = 0; i < 2048; ++i) {
            val = val * scale + i;
        }
        a[idx] = val;
    }
}

int main() {
    // Initialize GFL
    gpufl::InitOptions opts;
    opts.appName = "block_style_demo";
    opts.logPath = "gfl_block.log";
    opts.systemSampleRateMs = 10;
    opts.enableKernelDetails = true;
    opts.samplingAutoStart = true;
    opts.enableDebugOutput = true;
    if (!gpufl::init(opts)) {
        std::cerr << "Failed to initialize gpufl" << std::endl;
        return 1;
    }

    std::cout << "=== GPUFl Block-Style API Demo ===" << std::endl;
    std::cout << "Logs: " << opts.logPath << "\n" << std::endl;

    const int n = 1 << 22; // 4M elements
    const size_t bytes = n * sizeof(int);

    // Allocate memory
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    int* h_a = new int[n];
    int* h_b = new int[n];

    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    dim3 grid(4);
    dim3 block(256);

    std::cout << "Running heavy monitored scope..." << std::endl;
    GFL_SCOPE("heavy-scope") {
        for (int i = 0; i < 2000; ++i) {
            vectorScale<<<grid, block>>>(d_a, 3, n);
        }
        cudaDeviceSynchronize();
    }

    // Cleanup
    delete[] h_a;
    delete[] h_b;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    gpufl::shutdown();

    std::cout << "\n=== Demo Complete ===" << std::endl;
    std::cout << "Check " << opts.logPath << " for detailed logs" << std::endl;

    return 0;
}

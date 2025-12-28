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
        a[idx] *= scale;
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
    if (!gpufl::init(opts)) {
        std::cerr << "Failed to initialize gpufl" << std::endl;
        return 1;
    }

    std::cout << "=== GPUFl Block-Style API Demo ===" << std::endl;
    std::cout << "Logs: " << opts.logPath << "\n" << std::endl;

    const int n = 1024 * 1024;
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

    // ========================================================================
    // GFL_SCOPE - Block-style (most like Scala)
    // ========================================================================
    std::cout << "Using GFL_SCOPE (block-style)..." << std::endl;

    GFL_SCOPE("computation-phase-1") {
        // You can have multiple kernel launches inside the scope
        vectorAdd<<<grid, block>>>(d_a, d_b, d_c, n);
        cudaDeviceSynchronize();

        vectorMul<<<grid, block>>>(d_a, d_b, d_c, n);
        cudaDeviceSynchronize();
    }
    // Automatically logs scope_end when block exits

    std::cout << "   ✓ Scope automatically closed\n" << std::endl;

    // ========================================================================
    // ScopedRange RAII object
    // ========================================================================
    std::cout << "Using ScopedRange (RAII object)..." << std::endl;

    {
        gpufl::ScopedRange range("training-epoch-1");

        vectorAdd<<<grid, block>>>(d_a, d_b, d_c, n);
        cudaDeviceSynchronize();

        vectorMul<<<grid, block>>>(d_a, d_b, d_c, n);
        cudaDeviceSynchronize();

        // Range automatically ends when going out of scope
    }

    std::cout << "   ✓ Range destroyed, scope logged\n" << std::endl;

    // ========================================================================
    // Method 4: Lambda-based functional style
    // ========================================================================
    std::cout << "4. Using monitor() lambda wrapper..." << std::endl;

    gpufl::monitor("functional-style", [&]() {
        vectorAdd<<<grid, block>>>(d_a, d_b, d_c, n);
        cudaDeviceSynchronize();
    });

    std::cout << "   ✓ Lambda executed and monitored\n" << std::endl;

    // ========================================================================
    // Method 5: Automatic Kernel Monitoring (CUPTI)
    // ========================================================================
    std::cout << "5. Automatic Kernel Monitoring (CUPTI)..." << std::endl;
    std::cout << "   (No macros needed, just standard kernel launches)" << std::endl;

    vectorAdd<<<grid, block>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    std::cout << "   ✓ Kernel launched and automatically timed by CUPTI\n" << std::endl;

    // ========================================================================
    // Method 6: Nested scopes
    // ========================================================================
    std::cout << "6. Using nested scopes..." << std::endl;

    GFL_SCOPE("outer-scope") {
        vectorScale<<<grid, block>>>(d_a, 3, n);
        cudaDeviceSynchronize();

        GFL_SCOPE("inner-scope") {
            vectorAdd<<<grid, block>>>(d_a, d_b, d_c, n);
            cudaDeviceSynchronize();
        }

        vectorMul<<<grid, block>>>(d_a, d_b, d_c, n);
        cudaDeviceSynchronize();
    }

    std::cout << "   ✓ Nested scopes properly tracked\n" << std::endl;

    // ========================================================================
    // Cleanup
    // ========================================================================
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

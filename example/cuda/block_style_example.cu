#include <gpumon/gpumon.hpp>
#include <iostream>
#include <cuda_runtime.h>

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
    // Initialize GPUmon
    gpumon::InitOptions opts;
    opts.appName = "block_style_demo";
    opts.logPath = "gpumon_block.log";
    opts.sampleIntervalMs = 5;
    
    if (!gpumon::init(opts)) {
        std::cerr << "Failed to initialize gpumon" << std::endl;
        return 1;
    }
    
    std::cout << "=== GPUmon Block-Style API Demo ===" << std::endl;
    std::cout << "Logs: " << opts.logPath << "\n" << std::endl;
    
    const int n = 1024;
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
    // GPUMON_SCOPE - Block-style (most like Scala)
    // ========================================================================
    std::cout << "Using GPUMON_SCOPE (block-style)..." << std::endl;
    
    GPUMON_SCOPE("computation-phase-1") {
        // You can have multiple kernel launches inside the scope
        vectorAdd<<<grid, block>>>(d_a, d_b, d_c, n);
        cudaDeviceSynchronize();
        
        vectorMul<<<grid, block>>>(d_a, d_b, d_c, n);
        cudaDeviceSynchronize();
    }
    // Automatically logs scope_end when block exits
    
    std::cout << "   ✓ Scope automatically closed\n" << std::endl;

    // ========================================================================
    // ScopedMonitor RAII object
    // ========================================================================
    std::cout << "Using ScopedMonitor (RAII object)..." << std::endl;
    
    {
        gpumon::ScopedMonitor monitor("training-epoch-1");
        
        vectorAdd<<<grid, block>>>(d_a, d_b, d_c, n);
        cudaDeviceSynchronize();
        
        vectorMul<<<grid, block>>>(d_a, d_b, d_c, n);
        cudaDeviceSynchronize();
        
        // Monitor automatically ends when going out of scope
    }
    
    std::cout << "   ✓ Monitor destroyed, scope logged\n" << std::endl;
    
    // ========================================================================
    // Method 4: Lambda-based functional style
    // ========================================================================
    std::cout << "4. Using monitor() lambda wrapper..." << std::endl;
    
    gpumon::monitor("functional-style", [&]() {
        vectorAdd<<<grid, block>>>(d_a, d_b, d_c, n);
        cudaDeviceSynchronize();
    });
    
    std::cout << "   ✓ Lambda executed and monitored\n" << std::endl;
    
    // ========================================================================
    // Method 5: Traditional GPUMON_LAUNCH (auto-sync, auto-time)
    // ========================================================================
    std::cout << "5. Using GPUMON_LAUNCH (traditional)..." << std::endl;
    
    GPUMON_LAUNCH(vectorAdd, grid, block, 0, 0, d_a, d_b, d_c, n);
    
    std::cout << "   ✓ Kernel launched with automatic timing\n" << std::endl;
    
    // ========================================================================
    // Method 6: Nested scopes
    // ========================================================================
    std::cout << "6. Using nested scopes..." << std::endl;
    
    GPUMON_SCOPE("outer-scope") {
        vectorScale<<<grid, block>>>(d_a, 3, n);
        cudaDeviceSynchronize();
        
        GPUMON_SCOPE("inner-scope") {
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
    
    gpumon::shutdown();
    
    std::cout << "\n=== Demo Complete ===" << std::endl;
    std::cout << "Check " << opts.logPath << " for detailed logs" << std::endl;
    
    return 0;
}

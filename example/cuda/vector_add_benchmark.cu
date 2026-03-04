#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

__global__ void vectorAddGPU(const int* a, const int* b, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void vectorAddCPU(const int* a, const int* b, int* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int n = 1 << 24; // 16M elements
    const size_t size = n * sizeof(int);

    std::cout << "Vector Addition Benchmark (n = " << n << " elements)" << std::endl;
    std::cout << "Data size: " << (double)size / (1024 * 1024) << " MB per vector" << std::endl;

    int *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    h_a = (int*)malloc(size);
    h_b = (int*)malloc(size);
    h_c_cpu = (int*)malloc(size);
    h_c_gpu = (int*)malloc(size);

    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // CPU Benchmark
    auto start_cpu = std::chrono::high_resolution_clock::now();
    vectorAddCPU(h_a, h_b, h_c_cpu, n);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_time = end_cpu - start_cpu;
    std::cout << "CPU time: " << cpu_time.count() << " ms" << std::endl;

    // GPU Benchmark
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    auto start_total = std::chrono::high_resolution_clock::now();
    
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventRecord(start_event);
    vectorAddGPU<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop_event);

    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
    
    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> gpu_total_time = end_total - start_total;

    cudaEventSynchronize(stop_event);
    float kernel_time = 0;
    cudaEventElapsedTime(&kernel_time, start_event, stop_event);

    std::cout << "GPU Kernel time: " << kernel_time << " ms" << std::endl;
    std::cout << "GPU Total time (including H2D and D2H): " << gpu_total_time.count() << " ms" << std::endl;

    std::cout << "Speedup (CPU / GPU Kernel): " << cpu_time.count() / kernel_time << "x" << std::endl;
    std::cout << "Speedup (CPU / GPU Total): " << cpu_time.count() / gpu_total_time.count() << "x" << std::endl;

    // Verification
    bool passed = true;
    for (int i = 0; i < n; i++) {
        if (h_c_cpu[i] != h_c_gpu[i]) {
            passed = false;
            break;
        }
    }
    std::cout << "Verification: " << (passed ? "PASSED" : "FAILED") << std::endl;

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);

    return 0;
}

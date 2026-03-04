#include <gpufl/gpufl.hpp>
#include <thread>
#include <chrono>
#include <iostream>


__global__
void vectorAdd(int* a, int* b, int* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(int argc, char** argv) {
    // 1. Configure for System Monitoring
    gpufl::InitOptions opts;
    opts.app_name = "SystemMonitor";

    // Optional: Set a specific log path or let it default
    opts.log_path = "gpufl_system.log";
    opts.system_sample_rate_ms = 10;
    opts.sampling_auto_start = true;
    gpufl::init(opts);
    // GFL_SYSTEM_START("system");

    const int n = 2048 * 2048;
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

    vectorAdd<<<grid, block>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    std::cout << "Starting GPU System Monitor (Ctrl+C to stop)..." << std::endl;

    // GFL_SYSTEM_STOP("system");
    gpufl::shutdown();
    return 0;
}
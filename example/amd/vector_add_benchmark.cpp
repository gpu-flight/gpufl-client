#include <chrono>
#include <cstdlib>
#include <iostream>

#include <hip/hip_runtime.h>

static bool checkHip(const hipError_t status, const char* what) {
    if (status == hipSuccess) return true;
    std::cerr << what << " failed: " << hipGetErrorString(status) << "\n";
    return false;
}

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
    const int n = 1 << 24;
    const size_t size = static_cast<size_t>(n) * sizeof(int);

    std::cout << "HIP Vector Addition Benchmark (n = " << n << " elements)\n";
    std::cout << "Data size: " << static_cast<double>(size) / (1024 * 1024)
              << " MB per vector\n";

    int* h_a = static_cast<int*>(std::malloc(size));
    int* h_b = static_cast<int*>(std::malloc(size));
    int* h_c_cpu = static_cast<int*>(std::malloc(size));
    int* h_c_gpu = static_cast<int*>(std::malloc(size));

    if (!h_a || !h_b || !h_c_cpu || !h_c_gpu) {
        std::cerr << "Failed to allocate host memory\n";
        return 1;
    }

    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    auto start_cpu = std::chrono::high_resolution_clock::now();
    vectorAddCPU(h_a, h_b, h_c_cpu, n);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_time = end_cpu - start_cpu;
    std::cout << "CPU time: " << cpu_time.count() << " ms\n";

    int* d_a = nullptr;
    int* d_b = nullptr;
    int* d_c = nullptr;
    if (!checkHip(hipMalloc(&d_a, size), "hipMalloc(d_a)") ||
        !checkHip(hipMalloc(&d_b, size), "hipMalloc(d_b)") ||
        !checkHip(hipMalloc(&d_c, size), "hipMalloc(d_c)")) {
        return 1;
    }

    hipEvent_t start_event{};
    hipEvent_t stop_event{};
    if (!checkHip(hipEventCreate(&start_event), "hipEventCreate(start)") ||
        !checkHip(hipEventCreate(&stop_event), "hipEventCreate(stop)")) {
        return 1;
    }

    auto start_total = std::chrono::high_resolution_clock::now();

    if (!checkHip(hipMemcpy(d_a, h_a, size, hipMemcpyHostToDevice),
                  "hipMemcpy H2D a") ||
        !checkHip(hipMemcpy(d_b, h_b, size, hipMemcpyHostToDevice),
                  "hipMemcpy H2D b")) {
        return 1;
    }

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    if (!checkHip(hipEventRecord(start_event), "hipEventRecord(start)")) {
        return 1;
    }
    hipLaunchKernelGGL(vectorAddGPU, dim3(blocksPerGrid), dim3(threadsPerBlock),
                       0, 0, d_a, d_b, d_c, n);
    if (!checkHip(hipGetLastError(), "vectorAddGPU launch") ||
        !checkHip(hipEventRecord(stop_event), "hipEventRecord(stop)")) {
        return 1;
    }

    if (!checkHip(hipMemcpy(h_c_gpu, d_c, size, hipMemcpyDeviceToHost),
                  "hipMemcpy D2H c")) {
        return 1;
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> gpu_total_time =
        end_total - start_total;

    float kernel_time = 0.0f;
    if (!checkHip(hipEventSynchronize(stop_event), "hipEventSynchronize(stop)") ||
        !checkHip(hipEventElapsedTime(&kernel_time, start_event, stop_event),
                  "hipEventElapsedTime")) {
        return 1;
    }

    std::cout << "GPU Kernel time: " << kernel_time << " ms\n";
    std::cout << "GPU Total time (including H2D and D2H): "
              << gpu_total_time.count() << " ms\n";
    std::cout << "Speedup (CPU / GPU Kernel): "
              << cpu_time.count() / kernel_time << "x\n";
    std::cout << "Speedup (CPU / GPU Total): "
              << cpu_time.count() / gpu_total_time.count() << "x\n";

    bool passed = true;
    for (int i = 0; i < n; i++) {
        if (h_c_cpu[i] != h_c_gpu[i]) {
            passed = false;
            break;
        }
    }
    std::cout << "Verification: " << (passed ? "PASSED" : "FAILED") << "\n";

    (void)hipFree(d_a);
    (void)hipFree(d_b);
    (void)hipFree(d_c);
    (void)hipEventDestroy(start_event);
    (void)hipEventDestroy(stop_event);
    std::free(h_a);
    std::free(h_b);
    std::free(h_c_cpu);
    std::free(h_c_gpu);

    return passed ? 0 : 1;
}

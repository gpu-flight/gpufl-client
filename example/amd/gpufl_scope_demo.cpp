#include <cstdlib>
#include <iostream>

#include <hip/hip_runtime.h>

#include "gpufl/gpufl.hpp"

static bool checkHip(const hipError_t status, const char* what) {
    if (status == hipSuccess) return true;
    std::cerr << what << " failed: " << hipGetErrorString(status) << "\n";
    return false;
}

__global__ void vectorAddKernel(const int* a, const int* b, int* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void scaleKernel(int* data, int factor, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int value = data[idx];
        for (int i = 0; i < 1024; ++i) {
            value = value * factor + i;
        }
        data[idx] = value;
    }
}

int main() {
    gpufl::InitOptions opts;
    opts.app_name = "amd_gpufl_scope_demo";
    opts.log_path = "gfl_amd_scope";
    opts.backend = gpufl::BackendKind::Amd;
    opts.system_sample_rate_ms = 50;
    opts.kernel_sample_rate_ms = 0;
    opts.sampling_auto_start = true;
    opts.enable_kernel_details = false;
    opts.enable_debug_output = true;
    opts.enable_stack_trace = false;
    opts.profiling_engine = gpufl::ProfilingEngine::None;

    if (!gpufl::init(opts)) {
        std::cerr << "Failed to initialize gpufl for AMD backend\n";
        return 1;
    }

    std::cout << "=== GPUFL AMD Scope Demo ===\n";
    std::cout << "Backend: AMD telemetry + kernel tracing\n";
    std::cout << "Logs: " << opts.log_path << "\n";
    std::cout << "Note: with ROCprofiler-SDK available, this demo records HIP\n"
                 "kernel dispatches, memcpy activity, system telemetry, and user scopes.\n\n";

    const int n = 1 << 22;
    const size_t bytes = static_cast<size_t>(n) * sizeof(int);

    int* h_a = static_cast<int*>(std::malloc(bytes));
    int* h_b = static_cast<int*>(std::malloc(bytes));
    int* h_c = static_cast<int*>(std::malloc(bytes));
    if (!h_a || !h_b || !h_c) {
        std::cerr << "Failed to allocate host buffers\n";
        gpufl::shutdown();
        return 1;
    }

    for (int i = 0; i < n; ++i) {
        h_a[i] = i;
        h_b[i] = i * 3;
    }

    int* d_a = nullptr;
    int* d_b = nullptr;
    int* d_c = nullptr;
    if (!checkHip(hipMalloc(&d_a, bytes), "hipMalloc(d_a)") ||
        !checkHip(hipMalloc(&d_b, bytes), "hipMalloc(d_b)") ||
        !checkHip(hipMalloc(&d_c, bytes), "hipMalloc(d_c)")) {
        gpufl::shutdown();
        return 1;
    }

    if (!checkHip(hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice),
                  "hipMemcpy H2D a") ||
        !checkHip(hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice),
                  "hipMemcpy H2D b")) {
        gpufl::shutdown();
        return 1;
    }

    const dim3 block(256);
    const dim3 grid((n + block.x - 1) / block.x);

    GFL_SCOPE("hip_vector_add") {
        hipLaunchKernelGGL(vectorAddKernel, grid, block, 0, 0, d_a, d_b, d_c, n);
        if (!checkHip(hipGetLastError(), "vectorAddKernel launch") ||
            !checkHip(hipDeviceSynchronize(), "hipDeviceSynchronize(vectorAdd)")) {
            gpufl::shutdown();
            return 1;
        }
    }

    GFL_SCOPE("hip_scale_loop") {
        for (int iter = 0; iter < 50; ++iter) {
            hipLaunchKernelGGL(scaleKernel, grid, block, 0, 0, d_c, 2, n);
            if (!checkHip(hipGetLastError(), "scaleKernel launch")) {
                gpufl::shutdown();
                return 1;
            }
        }
        if (!checkHip(hipDeviceSynchronize(), "hipDeviceSynchronize(scaleLoop)")) {
            gpufl::shutdown();
            return 1;
        }
    }

    if (!checkHip(hipMemcpy(h_c, d_c, bytes, hipMemcpyDeviceToHost),
                  "hipMemcpy D2H c")) {
        gpufl::shutdown();
        return 1;
    }

    std::cout << "Sample output values: " << h_c[0] << ", " << h_c[1] << ", "
              << h_c[2] << "\n";

    (void)hipFree(d_a);
    (void)hipFree(d_b);
    (void)hipFree(d_c);
    std::free(h_a);
    std::free(h_b);
    std::free(h_c);

    gpufl::shutdown();

    std::cout << "\nDemo complete. Inspect logs with prefix " << opts.log_path
              << "\n";
    return 0;
}

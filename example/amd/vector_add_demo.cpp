#include <hip/hip_runtime.h>

#include <iostream>
#include <vector>

#include "gpufl/gpufl.hpp"

namespace {

constexpr int kElementCount = 1 << 20;
constexpr int kBlockSize = 256;

bool CheckHip(const hipError_t status, const char* what) {
    if (status == hipSuccess) return true;
    std::cerr << what << " failed: " << hipGetErrorString(status) << "\n";
    return false;
}

}  // namespace

__global__ void vectorAdd(const float* a, const float* b, float* c, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) c[i] = a[i] + b[i];
}

int main() {
    const size_t bytes = static_cast<size_t>(kElementCount) * sizeof(float);
    std::vector<float> a(kElementCount), b(kElementCount), c(kElementCount);
    for (int i = 0; i < kElementCount; ++i) {
        // Small integer-valued floats make exact result verification possible.
        a[i] = static_cast<float>(i % 1024);
        b[i] = static_cast<float>((i % 256) * 2);
    }

    gpufl::InitOptions opts;
    opts.app_name = "amd_vector_add_demo";
    opts.log_path = "gfl_amd_vector_add";
    opts.backend = gpufl::BackendKind::Amd;
    opts.profiling_engine = gpufl::ProfilingEngine::Trace;
    opts.enable_memory_tracking = true;
    opts.enable_synchronization = true;
    opts.continuous_system_sampling = true;
    opts.system_sample_rate_ms = 50;
    opts.enable_stack_trace = false;
    if (!gpufl::init(opts)) {
        std::cerr << "Failed to initialize GPUFlight for AMD vector addition\n";
        return 1;
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    bool ok = CheckHip(hipMalloc(&d_a, bytes), "hipMalloc(a)") &&
              CheckHip(hipMalloc(&d_b, bytes), "hipMalloc(b)") &&
              CheckHip(hipMalloc(&d_c, bytes), "hipMalloc(c)");
    if (ok) {
        ok = CheckHip(hipMemcpy(d_a, a.data(), bytes, hipMemcpyHostToDevice),
                      "H2D a") &&
             CheckHip(hipMemcpy(d_b, b.data(), bytes, hipMemcpyHostToDevice),
                      "H2D b");
    }
    if (ok) {
        GFL_SCOPE("vector-addition-scope") {
            const dim3 block(kBlockSize);
            const dim3 grid((kElementCount + kBlockSize - 1) / kBlockSize);
            hipLaunchKernelGGL(vectorAdd, grid, block, 0, 0,
                               d_a, d_b, d_c, kElementCount);
            ok = CheckHip(hipGetLastError(), "vectorAdd launch") &&
                 CheckHip(hipDeviceSynchronize(), "vectorAdd synchronize");
        }
    }
    if (ok) {
        ok = CheckHip(hipMemcpy(c.data(), d_c, bytes, hipMemcpyDeviceToHost),
                      "D2H c");
    }
    if (ok) {
        for (int i = 0; i < kElementCount; ++i) {
            if (c[i] != a[i] + b[i]) {
                std::cerr << "Verification failed at element " << i << "\n";
                ok = false;
                break;
            }
        }
    }

    // Always release successful allocations, including after a partial failure.
    // Keep the profiler running until frees have been captured.
    if (d_c) ok = CheckHip(hipFree(d_c), "hipFree(c)") && ok;
    if (d_b) ok = CheckHip(hipFree(d_b), "hipFree(b)") && ok;
    if (d_a) ok = CheckHip(hipFree(d_a), "hipFree(a)") && ok;
    gpufl::shutdown();
    gpufl::generateReport();
    if (!ok) return 2;

    std::cout << "\nPASS: vectorAdd verified all " << kElementCount
              << " elements.\nLogs: " << opts.log_path << "\n";
    return 0;
}

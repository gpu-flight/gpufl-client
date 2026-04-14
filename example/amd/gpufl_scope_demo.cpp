#include <hip/hip_runtime.h>

#include <cstdlib>
#include <iostream>

#include "../../include/gpufl/core/monitor.hpp"
#include "gpufl/gpufl.hpp"

static bool checkHip(const hipError_t status, const char* what) {
    if (status == hipSuccess) return true;
    std::cerr << what << " failed: " << hipGetErrorString(status) << "\n";
    return false;
}

// ---------------------------------------------------------------------------
// Kernel 1: Simple vector add — baseline, 100% occupancy expected
// ---------------------------------------------------------------------------
__global__ void vectorAddKernel(const int* a, const int* b, int* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: Shared-memory reduction — exercises LDS, register pressure from
// large block, and demonstrates < 100% occupancy via shared memory limit.
// Uses 32 KB shared memory per block to force occupancy below 100%.
// ---------------------------------------------------------------------------
constexpr int kReduceBlockSize = 256;
constexpr int kSharedBytes = 32 * 1024;  // 32 KB LDS per block
constexpr int kSharedInts = kSharedBytes / sizeof(int);

__global__ void sharedMemReduceKernel(const int* __restrict__ input,
                                      int* __restrict__ output,
                                      int n) {
    __shared__ int smem[kSharedInts];

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;

    // Load input into shared memory (with padding to fill the buffer)
    for (int i = tid; i < kSharedInts; i += blockDim.x) {
        const int src = blockIdx.x * kSharedInts + i;
        smem[i] = (src < n) ? input[src] : 0;
    }
    __syncthreads();

    // Parallel reduction in shared memory
    for (int stride = kSharedInts / 2; stride > 0; stride >>= 1) {
        for (int i = tid; i < stride; i += blockDim.x) {
            smem[i] += smem[i + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: Register-heavy computation — unrolled multiply chain uses many
// VGPRs, pushing register occupancy below 100%.
// Private (local) array forces scratch memory usage.
// ---------------------------------------------------------------------------
constexpr int kHeavyBlockSize = 128;

__global__ void registerHeavyKernel(float* __restrict__ data, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Private array forces scratch (local) memory allocation
    float local_buf[64];
    for (int i = 0; i < 64; ++i) {
        local_buf[i] = static_cast<float>(idx + i) * 0.01f;
    }

    // Heavy computation to drive up register usage
    float a = data[idx];
    float b = a * 1.01f, c = a * 0.99f, d = a + 0.5f;
    float e = b * c, f = d * a, g = e + f, h = g * 0.7f;

    for (int iter = 0; iter < 256; ++iter) {
        a = a * b + c;
        b = b * d + e;
        c = c * f + g;
        d = d * h + a;
        e = e * a + b;
        f = f * b + c;
        g = g * c + d;
        h = h * d + e;
        local_buf[iter & 63] = a + b + c + d + e + f + g + h;
    }

    float sum = 0.0f;
    for (int i = 0; i < 64; ++i) sum += local_buf[i];

    data[idx] = sum + a + b + c + d + e + f + g + h;
}

// ---------------------------------------------------------------------------
// Kernel 4: Dynamic shared memory — demonstrates runtime-specified shared mem
// ---------------------------------------------------------------------------
__global__ void dynamicSharedKernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int n) {
    extern __shared__ float dyn_smem[];

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;

    dyn_smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    // Simple smoothing: average with neighbors
    float val = dyn_smem[tid];
    if (tid > 0) val += dyn_smem[tid - 1];
    if (tid < blockDim.x - 1) val += dyn_smem[tid + 1];
    val /= 3.0f;

    if (gid < n) output[gid] = val;
}

int main() {
    gpufl::InitOptions opts;
    opts.app_name = "amd_gpufl_scope_demo";
    opts.log_path = "gfl_amd_scope";
    opts.backend = gpufl::BackendKind::Amd;
    opts.system_sample_rate_ms = 50;
    opts.kernel_sample_rate_ms = 0;
    opts.sampling_auto_start = true;
    opts.enable_kernel_details = true;
    opts.enable_debug_output = true;
    opts.enable_stack_trace = false;
    opts.profiling_engine = gpufl::ProfilingEngine::SassMetrics;

    if (!gpufl::init(opts)) {
        std::cerr << "Failed to initialize gpufl for AMD backend\n";
        return 1;
    }

    std::cout << "=== GPUFL AMD Scope Demo ===\n";
    std::cout << "Kernels: vectorAdd, sharedMemReduce, registerHeavy, dynamicShared\n\n";

    const int n = 1 << 20;  // 1M elements
    const size_t int_bytes = static_cast<size_t>(n) * sizeof(int);
    const size_t float_bytes = static_cast<size_t>(n) * sizeof(float);

    // ── Allocate ─────────────────────────────────────────────────────────
    int* h_a = static_cast<int*>(std::malloc(int_bytes));
    int* h_b = static_cast<int*>(std::malloc(int_bytes));
    int* h_c = static_cast<int*>(std::malloc(int_bytes));
    float* h_f = static_cast<float*>(std::malloc(float_bytes));
    if (!h_a || !h_b || !h_c || !h_f) {
        std::cerr << "Host alloc failed\n";
        gpufl::shutdown();
        return 1;
    }
    for (int i = 0; i < n; ++i) {
        h_a[i] = i;
        h_b[i] = i * 3;
        h_f[i] = static_cast<float>(i) * 0.001f;
    }

    int *d_a = nullptr, *d_b = nullptr, *d_c = nullptr, *d_reduce_out = nullptr;
    float *d_f = nullptr, *d_fo = nullptr;
    if (!checkHip(hipMalloc(&d_a, int_bytes), "hipMalloc(d_a)") ||
        !checkHip(hipMalloc(&d_b, int_bytes), "hipMalloc(d_b)") ||
        !checkHip(hipMalloc(&d_c, int_bytes), "hipMalloc(d_c)") ||
        !checkHip(hipMalloc(&d_reduce_out, int_bytes), "hipMalloc(d_reduce_out)") ||
        !checkHip(hipMalloc(&d_f, float_bytes), "hipMalloc(d_f)") ||
        !checkHip(hipMalloc(&d_fo, float_bytes), "hipMalloc(d_fo)")) {
        gpufl::shutdown();
        return 1;
    }

    checkHip(hipMemcpy(d_a, h_a, int_bytes, hipMemcpyHostToDevice), "H2D a");
    checkHip(hipMemcpy(d_b, h_b, int_bytes, hipMemcpyHostToDevice), "H2D b");
    checkHip(hipMemcpy(d_f, h_f, float_bytes, hipMemcpyHostToDevice), "H2D f");

    // ── Kernel 1: Vector add (baseline) ─────────────────────────────────
    std::cout << "  [1/4] vectorAdd — simple, full occupancy\n";
    GFL_SCOPE("vector_add") {
        const dim3 block(256);
        const dim3 grid((n + block.x - 1) / block.x);
        hipLaunchKernelGGL(vectorAddKernel, grid, block, 0, 0, d_a, d_b, d_c, n);
        checkHip(hipDeviceSynchronize(), "sync(vectorAdd)");
    }

    // ── Kernel 2: Shared-memory reduction (LDS-limited occupancy) ───────
    std::cout << "  [2/4] sharedMemReduce — 32KB shared mem, LDS-limited\n";
    GFL_SCOPE("shared_mem_reduce") {
        const int reduce_blocks = (n + kSharedInts - 1) / kSharedInts;
        for (int iter = 0; iter < 10; ++iter) {
            hipLaunchKernelGGL(sharedMemReduceKernel,
                               dim3(reduce_blocks), dim3(kReduceBlockSize),
                               0, 0, d_a, d_reduce_out, n);
        }
        checkHip(hipDeviceSynchronize(), "sync(sharedMemReduce)");
    }

    // ── Kernel 3: Register-heavy (register-limited occupancy) ───────────
    std::cout << "  [3/4] registerHeavy — many VGPRs + scratch, reg-limited\n";
    GFL_SCOPE("register_heavy") {
        const dim3 block(kHeavyBlockSize);
        const dim3 grid((n + block.x - 1) / block.x);
        for (int iter = 0; iter < 5; ++iter) {
            hipLaunchKernelGGL(registerHeavyKernel, grid, block, 0, 0, d_f, n);
        }
        checkHip(hipDeviceSynchronize(), "sync(registerHeavy)");
    }

    // ── Kernel 4: Dynamic shared memory ─────────────────────────────────
    std::cout << "  [4/4] dynamicShared — runtime shared mem allocation\n";
    GFL_SCOPE("dynamic_shared") {
        const dim3 block(256);
        const dim3 grid((n + block.x - 1) / block.x);
        const size_t dyn_shared = block.x * sizeof(float);  // 1 KB
        for (int iter = 0; iter < 10; ++iter) {
            hipLaunchKernelGGL(dynamicSharedKernel, grid, block, dyn_shared, 0,
                               d_f, d_fo, n);
        }
        checkHip(hipDeviceSynchronize(), "sync(dynamicShared)");
    }

    // ── Copy back & verify ──────────────────────────────────────────────
    checkHip(hipMemcpy(h_c, d_c, int_bytes, hipMemcpyDeviceToHost), "D2H c");
    std::cout << "\nSample output: c[0]=" << h_c[0] << " c[1]=" << h_c[1] << "\n";

    hipFree(d_a); hipFree(d_b); hipFree(d_c); hipFree(d_reduce_out);
    hipFree(d_f); hipFree(d_fo);
    std::free(h_a); std::free(h_b); std::free(h_c); std::free(h_f);

    gpufl::shutdown();
    gpufl::generateReport();

    std::cout << "\nDemo complete. Inspect logs with prefix " << opts.log_path << "\n";
    return 0;
}

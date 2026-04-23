// sass_divergence_demo.cu
//
// Demonstrates thread divergence patterns detectable via SASS metrics.
// Run with GPU Flight's SassMetrics engine, then inspect the scope log
// to see per-instruction warp vs thread execution counts.
//
// Kernels:
//   1. uniformWork       — no divergence (baseline)
//   2. branchByWarpLane  — classic if/else on threadIdx.x % 2
//   3. branchByWarpQuad  — only 1 in 4 threads takes the hot path
//   4. earlyExit         — variable-length work, some threads bail early
//   5. indirectBranch    — data-dependent branching (random input)

#include <cuda_runtime.h>

#include <iostream>

#include "gpufl/core/common.hpp"
#include "gpufl/core/monitor.hpp"
#include "gpufl/gpufl.hpp"
static bool CheckCuda(cudaError_t err, const char* call, const char* file,
                      int line) {
    if (err == cudaSuccess) return true;
    std::cerr << "[CUDA ERROR] " << file << ":" << line << " " << call
              << " failed: " << cudaGetErrorString(err) << " ("
              << static_cast<int>(err) << ")" << std::endl;
    return false;
}

#define CHECK_CUDA(call)                                             \
    do {                                                             \
        if (!CheckCuda((call), #call, __FILE__, __LINE__)) return 2; \
    } while (0)

// ---------------------------------------------------------------------------
// Kernel 1: No divergence — all threads do the same work.
// Expected: Active/32 = 32.0 everywhere
// ---------------------------------------------------------------------------
__global__ void uniformWork(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        for (int i = 0; i < 512; ++i) {
            val = val * 1.01f + 0.001f;
        }
        out[idx] = val;
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: Even/odd divergence — half the warp takes each path.
// Expected: Active/32 = 16.0 inside each branch
// ---------------------------------------------------------------------------
__global__ void branchByWarpLane(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        if (threadIdx.x % 2 == 0) {
            // Even threads: heavy multiply chain
            for (int i = 0; i < 512; ++i) {
                val = val * 1.01f + 0.001f;
            }
        } else {
            // Odd threads: heavy add chain (different work, same cost)
            for (int i = 0; i < 512; ++i) {
                val = val + 0.001f * (float)i;
            }
        }
        out[idx] = val;
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: Quad divergence — only 1 in 4 threads does real work.
// Expected: Active/32 = 8.0 in the hot path
// ---------------------------------------------------------------------------
__global__ void branchByWarpQuad(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        if (threadIdx.x % 4 == 0) {
            // Only every 4th thread enters — 8 out of 32
            for (int i = 0; i < 2048; ++i) {
                val = val * 1.001f + 0.0001f;
            }
        }
        out[idx] = val;
    }
}

// ---------------------------------------------------------------------------
// Kernel 4: Early exit — threads with small values bail out early.
// Divergence depends on data: threads that exit skip the heavy loop.
// ---------------------------------------------------------------------------
__global__ void earlyExit(float* out, const float* in, float threshold, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        // Threads below threshold skip the expensive work
        if (val < threshold) {
            out[idx] = val;
            return;  // early exit — this thread goes idle
        }
        // Remaining threads do the heavy computation
        for (int i = 0; i < 1024; ++i) {
            val = val * 1.01f - 0.005f;
        }
        out[idx] = val;
    }
}

// ---------------------------------------------------------------------------
// Kernel 5: Data-dependent branching — random input drives the branch.
// Divergence is unpredictable and varies per warp.
// ---------------------------------------------------------------------------
__global__ void indirectBranch(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        // Branch depends on data — some warps will diverge, some won't
        int category = (int)(val * 4.0f) % 4;
        switch (category) {
            case 0:
                for (int i = 0; i < 256; ++i) val = val * 1.01f;
                break;
            case 1:
                for (int i = 0; i < 256; ++i) val = val + 0.01f;
                break;
            case 2:
                for (int i = 0; i < 256; ++i) val = val - 0.005f;
                break;
            case 3:
                for (int i = 0; i < 256; ++i) val = val * 0.99f;
                break;
        }
        out[idx] = val;
    }
}

// ---------------------------------------------------------------------------

int main() {
    gpufl::InitOptions opts;
    opts.app_name = "sass_divergence_demo";
    opts.log_path = "sass_divergence";
    opts.system_sample_rate_ms = 10;
    opts.enable_kernel_details = true;
    opts.enable_debug_output = true;
    opts.sampling_auto_start = true;
    opts.enable_stack_trace = true;
    opts.profiling_engine = gpufl::ProfilingEngine::PcSampling;

    if (!gpufl::init(opts)) {
        std::cerr << "Failed to initialize gpufl" << std::endl;
        return 1;
    }

    std::cout << "=== SASS Divergence Demo ===" << std::endl;

    const int n = 1 << 20;  // 1M elements
    const size_t bytes = n * sizeof(float);

    float* d_in;
    float* d_out;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));

    // Fill input with values in [0, 1)
    float* h_in = new float[n];
    srand(42);
    for (int i = 0; i < n; ++i) {
        h_in[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);

    // --- Kernel 1: Baseline (no divergence) ---
    std::cout << "  [1/5] uniformWork — no divergence" << std::endl;
    GFL_SCOPE("1_uniform_work") {
        uniformWork<<<grid, block>>>(d_out, d_in, n);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // --- Kernel 2: Even/odd split ---
    std::cout << "  [2/5] branchByWarpLane — 50/50 divergence" << std::endl;
    GFL_SCOPE("2_branch_by_lane") {
        branchByWarpLane<<<grid, block>>>(d_out, d_in, n);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // --- Kernel 3: Quad split ---
    std::cout << "  [3/5] branchByWarpQuad — 75% threads idle" << std::endl;
    GFL_SCOPE("3_branch_by_quad") {
        branchByWarpQuad<<<grid, block>>>(d_out, d_in, n);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // --- Kernel 4: Early exit (50% threshold) ---
    std::cout << "  [4/5] earlyExit — data-dependent early return" << std::endl;
    GFL_SCOPE("4_early_exit") {
        earlyExit<<<grid, block>>>(d_out, d_in, 0.5f, n);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // --- Kernel 5: Data-dependent switch ---
    std::cout << "  [5/5] indirectBranch — 4-way data-dependent" << std::endl;
    GFL_SCOPE("5_indirect_branch") {
        indirectBranch<<<grid, block>>>(d_out, d_in, n);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    delete[] h_in;

    gpufl::shutdown();
    gpufl::generateReport();

    std::cout << "\n=== Done ===" << std::endl;
    std::cout << "Logs: " << opts.log_path << ".scope.log" << std::endl;
    std::cout << "Analyze with:" << std::endl;
    std::cout
        << "  session = GpuFlightSession('.', log_prefix='sass_divergence')"
        << std::endl;
    std::cout << "  session.inspect_profile_samples()" << std::endl;

    return 0;
}

// occupancy_demo.cu
//
// Demonstrates how gpufl detects which SM resource is limiting occupancy,
// then shows the targeted fix that brings utilization to 100%.
//
// Scenario: parallel block reduction
//
//   block_reduce_naive     — 16 KB static shared memory per block.
//                            At most ~3 blocks fit on a typical SM (48 KB limit),
//                            leaving threads/warps idle.
//                            gpufl reports: limiting_resource = "shared_mem"
//
//   block_reduce_optimized — exact same algorithm; shared memory switched to
//                            dynamic allocation sized at launch (1 KB per block).
//                            The SM can now host many more concurrent blocks.
//                            gpufl reports: limiting_resource = "warps" (100%)
//
// Build: registered in example/cuda/CMakeLists.txt as occupancy_demo
// Run:   ./occupancy_demo
// Logs:  occupancy_demo.log  (kernel events with occupancy breakdown fields)

#include <cstdio>
#include <cuda_runtime.h>
#include "gpufl/gpufl.hpp"

// ─── Kernel 1: static shared-memory allocation (the bottleneck) ──────────────
//
// 4096 floats = 16 KB of shared memory reserved per block regardless of block
// size. On hardware with 48 KB shared memory per SM this limits concurrent
// blocks to floor(48 / 16) = 3, wasting the remaining warp slots.
__global__ void block_reduce_naive(const float* __restrict__ in,
                                    float* __restrict__       out,
                                    int                       n)
{
    __shared__ float smem[4096]; // 16 KB — always reserved, even if unused

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    smem[tid] = (gid < n) ? in[gid] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = smem[0];
}

// ─── Kernel 2: dynamic shared-memory allocation (the fix) ────────────────────
//
// Identical algorithm; smem is sized exactly to blockDim.x at launch.
// With BLOCK = 256 that is 256 * 4 = 1 KB per block — 48× smaller than v1.
// The SM can now schedule the hardware-maximum number of blocks simultaneously,
// achieving 100 % warp occupancy.
__global__ void block_reduce_optimized(const float* __restrict__ in,
                                        float* __restrict__       out,
                                        int                       n)
{
    extern __shared__ float smem[]; // sized at launch: blockDim.x * sizeof(float)

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    smem[tid] = (gid < n) ? in[gid] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = smem[0];
}

// ─── Main ────────────────────────────────────────────────────────────────────
int main()
{
    // ── gpufl initialisation ─────────────────────────────────────────────────
    gpufl::InitOptions opts;
    opts.app_name              = "occupancy_demo";
    opts.log_path              = "occupancy_demo.log";
    opts.enable_kernel_details = true;  // required for occupancy breakdown fields
    opts.sampling_auto_start   = true;
    opts.enable_debug_output   = false;

    if (!gpufl::init(opts)) {
        std::fprintf(stderr, "ERROR: gpufl::init failed\n");
        return 1;
    }
    std::printf("gpufl initialised. Logs -> %s\n\n", opts.log_path.c_str());

    // ── Device info ──────────────────────────────────────────────────────────
    int dev = 0;
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    std::printf("Device : %s\n", prop.name);
    std::printf("SM count       : %d\n",   prop.multiProcessorCount);
    std::printf("MaxThreads/SM  : %d\n",   prop.maxThreadsPerMultiProcessor);
    std::printf("Shared mem/SM  : %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
    std::printf("Max blocks/SM  : %d\n\n", prop.maxBlocksPerMultiProcessor);

    // ── Data allocation ───────────────────────────────────────────────────────
    const int N = 1 << 22; // 4 M elements
    float *d_in, *d_out;
    cudaMalloc(&d_in,  N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemset(d_in, 0, N * sizeof(float));

    const int BLOCK = 256;
    const int GRID  = (N + BLOCK - 1) / BLOCK;
    const int ITERS = 200; // enough launches for gpufl to capture several samples

    // ── Phase 1: naive kernel ─────────────────────────────────────────────────
    // Static 16 KB smem per block → shared_mem is the limiting resource
    std::printf("=== Phase 1: naive (static 16 KB smem/block) ===\n");
    std::printf("Expected gpufl report: limiting_resource = \"shared_mem\"\n\n");

    GFL_SCOPE("naive") {
        for (int i = 0; i < ITERS; ++i) {
            block_reduce_naive<<<GRID, BLOCK>>>(d_in, d_out, N);
        }
        cudaDeviceSynchronize();
    }

    // ── Phase 2: optimised kernel ─────────────────────────────────────────────
    // Dynamic 1 KB smem per block → warp count becomes the only limiter (100%)
    std::printf("=== Phase 2: optimised (dynamic 1 KB smem/block) ===\n");
    std::printf("Expected gpufl report: limiting_resource = \"warps\"\n\n");

    GFL_SCOPE("optimized") {
        const size_t smem_bytes = BLOCK * sizeof(float); // 1 KB
        for (int i = 0; i < ITERS; ++i) {
            block_reduce_optimized<<<GRID, BLOCK, smem_bytes>>>(d_in, d_out, N);
        }
        cudaDeviceSynchronize();
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    cudaFree(d_in);
    cudaFree(d_out);

    gpufl::shutdown();

    std::printf("Done.\n\n");
    std::printf("Inspect the log with the gpufl analyzer:\n");
    std::printf("  from gpufl.analyzer import GpuFlightSession\n");
    std::printf("  s = GpuFlightSession('.', log_prefix='occupancy_demo')\n");
    std::printf("  s.inspect_hotspots()\n\n");
    std::printf("Look for 'limiting_resource' and the per-resource occupancy\n");
    std::printf("columns (reg_occupancy, smem_occupancy, warp_occupancy,\n");
    std::printf("block_occupancy) to see exactly what changed between the\n");
    std::printf("two kernel variants.\n");

    return 0;
}

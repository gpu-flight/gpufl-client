// memory_coalescing_demo.cu — GPUFlight example
//
// Demonstrates how memory access patterns affect GPU performance
// by comparing two simple matrix multiplication strategies:
//   1) Row-per-thread: each thread computes one row of C (uncoalesced B access)
//   2) Col-per-thread: each thread computes one column of C (coalesced B access)
//
// Run with GPUFlight to see the difference in memory efficiency,
// stall distribution, and SASS instruction-level profiling.

#include <iostream>
#include <random>

#include "gpufl/gpufl.hpp"

// Each thread computes one full ROW of C.
// Adjacent threads (lane 0, lane 1, ...) access rows 0, 1, ... of A — which
// are contiguous — but access COLUMNS 0, 1, ... of B for the same loop
// iteration.  Since B is stored row-major, B[i*N + 0] and B[i*N + 1] are
// adjacent, but each thread reads B[i*N + col] where col varies across the
// OUTER loop, not across threads.  The inner-loop reads of A[row*K + i] have
// row varying across threads, causing stride-K access → poor coalescing.
__global__ void matmul_row_per_thread(const float* __restrict__ A,
                                      const float* __restrict__ B, float* C,
                                      int M, int K, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    for (int col = 0; col < N; col++) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Each thread computes one full COLUMN of C.
// Adjacent threads access col, col+1, col+2, ... so the read B[i*N + col]
// hits consecutive addresses within a warp → perfectly coalesced.
// The write C[row*N + col] is also coalesced for the same reason.
__global__ void matmul_col_per_thread(const float* __restrict__ A,
                                      const float* __restrict__ B, float* C,
                                      int M, int K, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;

    for (int row = 0; row < M; row++) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    // Initialize GPUFlight with PC Sampling + SASS metrics
    gpufl::InitOptions opts;
    opts.app_name = "memory_coalescing_demo";
    opts.log_path = "memory_coalescing_demo";
    opts.system_sample_rate_ms = 10;
    opts.kernel_sample_rate_ms = 10;
    opts.enable_kernel_details = true;
    opts.sampling_auto_start = true;
    opts.enable_debug_output = true;
    opts.profiling_engine = gpufl::ProfilingEngine::PcSamplingWithSass;
    gpufl::init(opts);

    constexpr int M = 512, K = 256, N = 512;
    constexpr size_t sizeA = M * K * sizeof(float);
    constexpr size_t sizeB = K * N * sizeof(float);
    constexpr size_t sizeC = M * N * sizeof(float);

    auto* h_A = static_cast<float*>(malloc(sizeA));
    auto* h_B = static_cast<float*>(malloc(sizeB));
    auto* h_C = static_cast<float*>(malloc(sizeC));

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    // Fill with random data
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (int i = 0; i < M * K; i++) h_A[i] = dis(gen);
    for (int i = 0; i < K * N; i++) h_B[i] = dis(gen);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // --- Uncoalesced version ---
    cudaEventRecord(start);
    GFL_SCOPE("row-per-thread") {
        matmul_row_per_thread<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);
        cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms1 = 0;
    cudaEventElapsedTime(&ms1, start, stop);
    std::cout << "Row-per-thread (uncoalesced): " << ms1 << " ms\n";

    // --- Coalesced version ---
    cudaEventRecord(start);
    GFL_SCOPE("col-per-thread") {
        matmul_col_per_thread<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);
        cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms2 = 0;
    cudaEventElapsedTime(&ms2, start, stop);
    std::cout << "Col-per-thread (coalesced):   " << ms2 << " ms\n";
    std::cout << "Speedup: " << ms1 / ms2 << "x\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    gpufl::shutdown();
    gpufl::generateReport();
    return 0;
}

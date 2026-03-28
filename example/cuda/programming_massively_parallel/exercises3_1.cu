#include <iostream>
#include <random>

#include "../../../include/gpufl/gpufl.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/monitor.hpp"
#include "gpufl/gpufl.hpp"

__global__ void matmul_row_per_thread(const float* __restrict__ A,
                                      const float* __restrict__ B, float* C,
                                      const int M, const int K, const int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int i = 0; i < K; i++) {
                sum += A[row * K + i] * B[i * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

__global__ void matmul_col_per_thread(const float* __restrict__ A,
                                      const float* __restrict__ B, float* C,
                                      const int M, const int K, const int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < N) {
        for (int row = 0; row < M; row++) {
            float sum = 0.0f;
            for (int i = 0; i < K; i++) {
                sum += A[row * K + i] * B[i * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

int main() {
    gpufl::InitOptions opts;
    opts.app_name = "exercise3_1";
    opts.log_path = "exercise3_1";
    opts.system_sample_rate_ms = 10;
    opts.kernel_sample_rate_ms = 10;
    opts.enable_kernel_details = true;
    opts.sampling_auto_start = true;
    opts.enable_debug_output = true;
    opts.profiling_engine = gpufl::ProfilingEngine::PcSamplingWithSass;
    gpufl::init(opts);

    constexpr int M = 512;
    constexpr int K = 256;
    constexpr int N = 512;

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

    std::random_device rd;

    std::mt19937 gen(rd());

    std::uniform_real_distribution<> dis(0.0f, 1.0f);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            h_A[i * K + j] = dis(gen);
        }
    }

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            h_B[i * N + j] = dis(gen);
        }
    }

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventRecord(start);
    GFL_SCOPE("row-base") {
        matmul_row_per_thread<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C,
                                                                  M, K, N);
        cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
    }

    cudaEventRecord(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Kernel Execution Time: " << milliseconds << " milliseconds"
              << std::endl;

    cudaEventRecord(start);
    GFL_SCOPE("col-base") {
        matmul_col_per_thread<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C,
                                                                  M, K, N);
        cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds2 = 0;
    cudaEventElapsedTime(&milliseconds2, start, stop);
    std::cout << "Kernel Execution Time: " << milliseconds2 << " milliseconds"
              << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    gpufl::shutdown();
    return 0;
}

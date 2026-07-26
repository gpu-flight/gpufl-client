// A CUDA workload that knows nothing about gpufl.
//
// Stands in for the case the deep window exists for: a long-running job whose
// source you can't edit, so it can never call gpufl::deepWindow() itself and
// the trigger has to come from `gpufl trace`. Deliberately NOT part of the
// example/ tree, which links gpufl - the point here is that it doesn't.
//
// Runs for a wall-clock duration and reports how many iterations it got
// through, so the same binary doubles as the throughput probe for measuring
// what an armed-but-idle session costs.
//
//   nvcc -O2 -lineinfo -o deep_window_target deep_window_target.cu
//   ./deep_window_target [seconds]

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

__global__ void computeHeavy(float* out, const float* in, int n, int iters) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = in[idx];
    for (int i = 0; i < iters; ++i) {
        val = val * 1.0009f + 0.0001f;
        val = fmaf(val, 0.9991f, 0.0002f);
    }
    out[idx] = val;
}

int main(int argc, char** argv) {
    const int seconds = argc > 1 ? std::atoi(argv[1]) : 12;
    const int n = 1 << 20;

    float* d_in = nullptr;
    float* d_out = nullptr;
    if (cudaMalloc(&d_in, n * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_out, n * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed\n");
        return 2;
    }
    cudaMemset(d_in, 0, n * sizeof(float));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    // One untimed pass so context creation, module load and clock ramp land
    // outside the measured region.
    computeHeavy<<<blocks, threads>>>(d_out, d_in, n, 4000);
    cudaDeviceSynchronize();

    const auto t0 = std::chrono::steady_clock::now();
    long iterations = 0;
    for (;;) {
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        if (elapsed >= seconds * 1000L) break;
        computeHeavy<<<blocks, threads>>>(d_out, d_in, n, 4000);
        cudaDeviceSynchronize();
        ++iterations;
    }
    const double secs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count() / 1000.0;

    // The throughput line the harness parses.
    std::printf("ITERATIONS=%ld SECONDS=%.3f ITERS_PER_SEC=%.2f\n",
                iterations, secs, iterations / secs);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}

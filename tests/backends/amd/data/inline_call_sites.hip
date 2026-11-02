// Fixture for the AMD ISA source-line tests in
// tests/core/test_source_capture_policy.cpp. Rebuild from the repository root
// with ROCm 7.2.0's clang:
//   clang++ -O2 -gline-tables-only -ffile-prefix-map=$PWD=/src -x hip \
//     --offload-arch=gfx942 --offload-device-only --no-gpu-bundle-output \
//     tests/backends/amd/data/inline_call_sites.hip \
//     -o tests/backends/amd/data/inline_call_sites.gfx942.co
// The line numbers below are the tests' expectations.
#include <hip/hip_runtime.h>

// Lab 0029's answer. __syncthreads(), expf, fmaxf, blockIdx and blockDim are
// inlined from HIP's and clang's headers.
__global__ void softmaxRows(const float* X, float* Y, int rows, int cols) {
    __shared__ float red[256];

    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    const float* src = X + row * cols;
    float* dst = Y + row * cols;

    float localMax = -3.402823466e+38f;
    for (int c = tid; c < cols; c += blockDim.x) localMax = fmaxf(localMax, src[c]);
    red[tid] = localMax;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) red[tid] = fmaxf(red[tid], red[tid + stride]);
        __syncthreads();
    }
    float m = red[0];
    __syncthreads();

    float localSum = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) localSum += expf(src[c] - m);
    red[tid] = localSum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) red[tid] += red[tid + stride];
        __syncthreads();
    }
    float denom = red[0];

    for (int c = tid; c < cols; c += blockDim.x) {
        dst[c] = expf(src[c] - m) / denom;
    }
}

// A helper of the program's own, inlined into its kernel: the barrier it
// calls belongs to its own __syncthreads() line.
__device__ float blockSum(float* red, float v) {
    red[threadIdx.x] = v;
    __syncthreads();
    float sum = 0.0f;
    for (int i = 0; i < 64; ++i) sum += red[i];
    return sum;
}

__global__ void normalize(float* data) {
    __shared__ float red[64];
    float v = data[threadIdx.x];
    data[threadIdx.x] = v / blockSum(red, v);
}

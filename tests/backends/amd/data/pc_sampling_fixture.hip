// Fixture for tests/backends/amd/test_amd_pc_sampling.cpp. Rebuild with:
//   clang++ -O2 -x hip --offload-arch=gfx942 --offload-device-only \
//     --no-gpu-bundle-output pc_sampling_fixture.hip \
//     -o pc_sampling_fixture.gfx942.co
// and its copy with DWARF line tables, from the repository root:
//   clang++ -O2 -gline-tables-only -ffile-prefix-map=$PWD=. -x hip \
//     --offload-arch=gfx942 --offload-device-only --no-gpu-bundle-output \
//     tests/backends/amd/data/pc_sampling_fixture.hip \
//     -o tests/backends/amd/data/pc_sampling_fixture.lines.gfx942.co
#include <hip/hip_runtime.h>

__device__ __noinline__ float scaleAndBias(float x, float s) {
    return x * s + 1.0f;
}

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void scaled(const float* a, float* c, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = scaleAndBias(a[i], s);
}

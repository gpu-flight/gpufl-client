#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr std::size_t kElements = 1u << 20;
constexpr std::size_t kBytes = kElements * sizeof(float);

__global__ void AddOne(const float* input, float* output, std::size_t count) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = input[index] + 1.0f;
}

bool Check(cudaError_t result, const char* operation) {
    if (result == cudaSuccess) return true;
    std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(result));
    return false;
}

}  // namespace

int main() {
    std::vector<float> input(kElements, 2.0f);
    std::vector<float> output(kElements, 0.0f);
    float* device_input = nullptr;
    float* device_output = nullptr;

    // The ordering is load-bearing for the regression:
    // first allocation -> cuInit/injection callback -> deferred init;
    // second allocation -> immediate H2D before any launch/sync wrapper.
    if (!Check(cudaMalloc(&device_input, kBytes), "cudaMalloc(input)") ||
        !Check(cudaMalloc(&device_output, kBytes), "cudaMalloc(output)") ||
        !Check(cudaMemcpy(device_input, input.data(), kBytes,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(H2D)")) {
        cudaFree(device_input);
        cudaFree(device_output);
        return 1;
    }

    AddOne<<<(kElements + 255) / 256, 256>>>(
        device_input, device_output, kElements);
    if (!Check(cudaGetLastError(), "AddOne launch") ||
        !Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize") ||
        !Check(cudaMemcpy(output.data(), device_output, kBytes,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(D2H)")) {
        cudaFree(device_input);
        cudaFree(device_output);
        return 1;
    }

    const bool correct =
        std::fabs(output.front() - 3.0f) < 1e-6f &&
        std::fabs(output.back() - 3.0f) < 1e-6f;
    cudaFree(device_input);
    cudaFree(device_output);
    if (!correct) {
        std::fprintf(stderr, "VERIFY FAIL\n");
        return 2;
    }

    std::printf("VERIFY PASS bytes=%zu\n", kBytes);
    return 0;
}

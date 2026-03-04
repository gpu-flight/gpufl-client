#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        printf("CRITICAL: cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Found %d devices.\n", deviceCount);

    if (deviceCount > 0) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, 0); // Check Device 0
        if (err != cudaSuccess) {
            printf("CRITICAL: cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
            return 1;
        }
        printf("Success! Device 0: %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);
    }
    return 0;
}
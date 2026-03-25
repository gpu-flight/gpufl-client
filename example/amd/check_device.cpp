#include <cstdio>

#include <hip/hip_runtime.h>

int main() {
    int deviceCount = 0;
    hipError_t err = hipGetDeviceCount(&deviceCount);

    if (err != hipSuccess) {
        std::printf("CRITICAL: hipGetDeviceCount failed: %s\n",
                    hipGetErrorString(err));
        return 1;
    }

    std::printf("Found %d HIP devices.\n", deviceCount);

    if (deviceCount > 0) {
        hipDeviceProp_t prop{};
        err = hipGetDeviceProperties(&prop, 0);
        if (err != hipSuccess) {
            std::printf("CRITICAL: hipGetDeviceProperties failed: %s\n",
                        hipGetErrorString(err));
            return 1;
        }

        std::printf("Success! Device 0: %s (arch %s, capability %d.%d)\n",
                    prop.name, prop.gcnArchName, prop.major, prop.minor);
    }

    return 0;
}

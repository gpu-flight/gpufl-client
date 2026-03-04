#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include "gpufl/backends/nvidia/sampler/cupti_sass.hpp"
#include "gpufl/core/debug_logger.hpp"

using namespace gpufl::nvidia;

__global__ void simpleKernel(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2 + 1;
    }
}

int main() {
    gpufl::DebugLogger::setEnabled(true);

    if (cuInit(0) != CUDA_SUCCESS) {
        std::cerr << "cuInit failed" << std::endl;
        return 1;
    }

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found" << std::endl;
        return 1;
    }

    uint32_t deviceIndex = 0;
    cudaSetDevice(deviceIndex);

    // List of common SASS metrics to try (actual names depend on GPU architecture)
    // For many architectures, these are common:
    std::vector<std::string> metrics = {
        "smsp__sass_thread_inst_executed",
        "smsp__sass_thread_inst_executed_op_branch"
    };

    std::cout << "Configuring SASS metrics... with device index:" << deviceIndex << std::endl;
    if (!CuptiSass::setMetricsConfig(deviceIndex, metrics)) {
        std::cerr << "Failed to set metrics config. This might be because the metric names are not supported on this GPU." << std::endl;
        // Try to continue anyway or exit? Let's exit if we can't configure.
        return 1;
    }

    std::cout << "Enabling SASS metrics..." << std::endl;
    if (!CuptiSass::enableMetrics()) {
        std::cerr << "Failed to enable metrics" << std::endl;
        CuptiSass::unsetMetricsConfig(deviceIndex);
        return 1;
    }

    // Launch a simple kernel
    const int n = 1024;
    int *d_data;
    cudaMalloc(&d_data, n * sizeof(int));
    
    std::cout << "Launching kernel..." << std::endl;
    simpleKernel<<<1, 256>>>(d_data, n);
    cudaDeviceSynchronize();

    std::cout << "Flushing SASS metrics data..." << std::endl;
    auto data = CuptiSass::flushMetricsData();

    std::cout << "Disabling SASS metrics..." << std::endl;
    CuptiSass::disableMetrics();

    std::cout << "Collected " << data.size() << " SASS records." << std::endl;
    for (const auto& record : data) {
        std::cout << "Function: " << record.functionName 
                  << " PC Offset: 0x" << std::hex << record.pcOffset << std::dec << std::endl;
        for (const auto& val : record.values) {
            std::cout << "  Metric ID: " << val.metricId << " Value: " << val.value << std::endl;
        }
    }

    CuptiSass::unsetMetricsConfig(deviceIndex);
    cudaFree(d_data);

    std::cout << "Test completed." << std::endl;
    return 0;
}

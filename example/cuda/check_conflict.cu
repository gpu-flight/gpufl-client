#include <iostream>
#include <cuda_runtime.h>
#include <cupti.h>

#include "gpufl/core/debug_logger.hpp"

void CUPTIAPI BufferRequested(uint8_t **buffer, size_t *size,
                                                size_t *maxNumRecords) {
    *size = 64 * 1024;
    *buffer = static_cast<uint8_t *>(malloc(*size));
    *maxNumRecords = 0;
}

void CUPTIAPI BufferCompletedTest(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize) {
    CUpti_Activity *record = NULL;

    printf("called");
    while (cuptiActivityGetNextRecord(buffer, validSize, &record) == CUPTI_SUCCESS) {
        printf("hello");
        if (record->kind == CUPTI_ACTIVITY_KIND_MODULE) {
            CUpti_ActivityModule *module = (CUpti_ActivityModule *)record;

            // Accessing the data
            printf("Module Loaded:\n");
            printf("  Context ID: %u\n", module->contextId);
            printf("  Module ID: %u\n", module->id);
            printf("  Cubin Size: %llu bytes\n", (unsigned long long)module->cubinSize);
        }
    }
    free(buffer);
}

__global__
void vectorScale(int* a, int scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int val = a[idx];
        for (int i = 0; i < 2048; ++i) {
            val = val * scale + i;
        }
        a[idx] = val;
    }
}

int main() {
    std::cout << "--- STARTING CONFLICT CHECK ---" << std::endl;

    // 1. Force CUDA Initialization
    cudaFree(0);

    CUptiResult resCb = cuptiActivityRegisterCallbacks(BufferRequested, BufferCompletedTest);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MODULE);
    if (resCb != CUPTI_SUCCESS) {
        const char* errStr = nullptr;
        cuptiGetResultString(resCb, &errStr);
        GFL_LOG_ERROR("FATAL: Failed to register activity callbacks.");
        GFL_LOG_ERROR("Error: ", (errStr ? errStr : "unknown"), " (Code ", resCb, ")");
        return 0;
    } else {
        std::cout << "PASS: cupti success" << std::endl;
    }
    const int n = 1 << 24;
    const size_t bytes = n * sizeof(int);

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    dim3 grid(4);
    dim3 block(256);

    for (int i = 0; i < 2000; ++i) {
        vectorScale<<<grid, block>>>(d_a, 3, n);
    }
    cudaDeviceSynchronize();

    cuptiActivityFlushAll(0);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MODULE);
    return 0;
}
#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>
#include <cupti_pmsampling.h>
#include <cupti_target.h>
#include <stdio.h>

#include <string>
#include <vector>
// Global variables for Range Profiler
void CUPTIAPI CallbackHandler(void *userData, CUpti_CallbackDomain domain,
                              CUpti_CallbackId callbackId,
                              const CUpti_CallbackData *callbackData) {
    switch (domain) {
        case CUPTI_CB_DOMAIN_RUNTIME_API:
            if (callbackData->callbackSite == CUPTI_API_ENTER) {
                // access parameters passed to cudaMemcpy
                if (callbackId == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020) {
                    printf("Memcpy size = %zu\n",
                           ((cudaMemcpy_v3020_params *)(callbackData
                                                            ->functionParams))
                               ->count);
                    printf("Memcpy kind = %d\n",
                           ((cudaMemcpy_v3020_params *)(callbackData
                                                            ->functionParams))
                               ->kind);
                }
            }
            break;
        case CUPTI_CB_DOMAIN_RESOURCE: {
            // For resource domain, cbdata is CUpti_ResourceData — NOT
            // CUpti_CallbackData
            const auto *resourceData =
                reinterpret_cast<const CUpti_ResourceData *>(callbackData);

            if (callbackId == CUPTI_CBID_RESOURCE_MODULE_LOADED ||
                callbackId == CUPTI_CBID_RESOURCE_MODULE_PROFILED ||
                callbackId == CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING) {
                printf("CUPTI_CB_DOMAIN_RESOURCE cbid=%d\n", (int)callbackId);

                // resourceDescriptor holds CUpti_ModuleResourceData for module
                // callbacks
                auto *modData = static_cast<CUpti_ModuleResourceData *>(
                    resourceData->resourceDescriptor);

                if (modData) {
                    printf(
                        "modData is available — moduleId=%u  cubinSize=%zu\n",
                        modData->moduleId, modData->cubinSize);

                    // modData->pCubin contains the raw cubin bytes
                    // You can save it to a file here:
                    if (modData->pCubin && modData->cubinSize > 0) {
                        char fname[64];
                        snprintf(fname, sizeof(fname), "module_%u.cubin",
                                 modData->moduleId);
                        FILE *f = fopen(fname, "wb");
                        if (f) {
                            fwrite(modData->pCubin, 1, modData->cubinSize, f);
                            fclose(f);
                            printf("  Saved cubin → %s\n", fname);
                        }
                    }
                }
            }
            break;
        }
        default:
            break;
    }
}

// CUDA kernel for vector addition
__global__ void VectorAdd(const float *A, const float *B, float *C,
                          const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] + B[idx];
}

int main() {
    int vectorLen = 1024 * 1024;
    size_t size = vectorLen * sizeof(float);

    // Step 1: Subscribe to CUPTI callbacks
    CUpti_SubscriberHandle subscriber;
    cuptiSubscribe(&subscriber,
                   reinterpret_cast<CUpti_CallbackFunc>(CallbackHandler),
                   nullptr);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION);

    // Step 2: Enable callback for specific domains and callback IDs
    // Enable all callbacks for CUDA Runtime APIs.
    // Callback will be invoked at the entry and exit points of each of the CUDA
    // Runtime API.
    cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
    cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE,
                        CUPTI_CBID_RESOURCE_MODULE_LOADED);
    cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE,
                        CUPTI_CBID_RESOURCE_MODULE_PROFILED);
    cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE,
                        CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING);
    // Host memory allocation
    auto *h_A = static_cast<float *>(malloc(size));
    auto *h_B = static_cast<float *>(malloc(size));
    auto *h_C = static_cast<float *>(malloc(size));

    // Initialize vectors
    for (int i = 0; i < vectorLen; ++i) {
        h_A[i] = rand() / static_cast<float>(RAND_MAX);
        h_B[i] = rand() / static_cast<float>(RAND_MAX);
    }

    // Device memory allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc(reinterpret_cast<void **>(&d_A), size);
    cudaMalloc(reinterpret_cast<void **>(&d_B), size);
    cudaMalloc(reinterpret_cast<void **>(&d_C), size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 128;
    int blocksPerGrid = (vectorLen + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel (profiler will capture this call)
    VectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, vectorLen);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    cudaDeviceSynchronize();
    cudaDeviceReset();
    // Step 3: Disable callback for domains and callback IDs
    cuptiEnableDomain(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
    cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RESOURCE,
                        CUPTI_CBID_RESOURCE_MODULE_LOADED);
    cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RESOURCE,
                        CUPTI_CBID_RESOURCE_MODULE_PROFILED);
    cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RESOURCE,
                        CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING);
    cuptiUnsubscribe(subscriber);

    return 0;
}

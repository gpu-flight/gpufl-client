#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <vector>
#include <cupti_target.h>
#include <cupti_pmsampling.h>
#include <string>
#include <cupti.h>
// Global variables for Range Profiler
void CUPTIAPI
CallbackHandler(void *userData, CUpti_CallbackDomain domain, CUpti_CallbackId callbackId, const CUpti_CallbackData *callbackData) {
    switch(domain)
    {
        case CUPTI_CB_DOMAIN_RUNTIME_API:
            if (callbackData->callbackSite == CUPTI_API_ENTER)
            {
                // access parameters passed to cudaMemcpy
                if (callbackId == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020)
                {
                    printf("Memcpy size = %zu\n", ((cudaMemcpy_v3020_params *)(callbackData->functionParams))->count);
                    printf("Memcpy kind = %d\n", ((cudaMemcpy_v3020_params *)(callbackData->functionParams))->kind);
                }
            }
            break;
        case CUPTI_CB_DOMAIN_RESOURCE: {
            if (callbackId == CUPTI_CBID_RESOURCE_MODULE_LOADED | CUPTI_CBID_RESOURCE_MODULE_PROFILED | CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING) {
                printf("CUPTI_CB_DOMAIN_RESOURCE \n");

                CUpti_ModuleResourceData *pModuleResourceData = (CUpti_ModuleResourceData *)userData;
                if (pModuleResourceData) {

                    printf("modData is available");
                }
            }

        }
            break;
        default:
            break;
    }
}

// CUDA kernel for vector addition
__global__ void VectorAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

int main() {
    int vectorLen = 1024 * 1024;
    size_t size = vectorLen * sizeof(float);

    // Step 1: Subscribe to CUPTI callbacks
    CUpti_SubscriberHandle subscriber;
    cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)CallbackHandler, NULL);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION);

    // Step 2: Enable callback for specific domains and callback IDs
    // Enable all callbacks for CUDA Runtime APIs.
    // Callback will be invoked at the entry and exit points of each of the CUDA Runtime API.
    cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
    cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_MODULE_LOADED);
    cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_MODULE_PROFILED);
    cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING);
    // Host memory allocation
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize vectors
    for (int i = 0; i < vectorLen; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Device memory allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 128;
    int blocksPerGrid = (vectorLen + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel (profiler will capture this call)
    VectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, vectorLen);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    cudaDeviceSynchronize();
    cudaDeviceReset();
    // Step 3: Disable callback for domains and callback IDs
    cuptiEnableDomain(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
    cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_MODULE_LOADED);
    cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_MODULE_PROFILED);
    cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING);
    cuptiUnsubscribe(subscriber);

    return 0;
}

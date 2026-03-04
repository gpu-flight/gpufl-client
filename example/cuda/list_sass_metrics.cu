#include <cuda_runtime.h>
#include <cupti.h>
#include <iostream>

#define CHECK_CUPTI(ans) { if(ans != CUPTI_SUCCESS) { std::cerr << "CUPTI Error" << std::endl; } }

// 1. This is the only function we need to intercept for Runtime-based apps
void CUPTIAPI MyCallback(void *userdata, CUpti_CallbackDomain domain,
                         CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo) {

    if (cbInfo->callbackSite == CUPTI_API_ENTER) {
        // This ID catches almost all internal module loads in modern CUDA
        if (cbid == CUPTI_DRIVER_TRACE_CBID_cuModuleLoadDataEx ||
            cbid == CUPTI_DRIVER_TRACE_CBID_cuModuleLoadData) {

            auto params = (cuModuleLoadData_params *)(cbInfo->functionParams);
            const char* img = (const char*)params->image;

            if (img && img[0] == '.') {
                printf("\n--- INTERCEPTED PTX SOURCE ---\n%s\n", img);
            }
            }
    }
}

__global__ void my_test_kernel() { }

int main() {
    CUpti_SubscriberHandle sub;
    CHECK_CUPTI(cuptiSubscribe(&sub, (CUpti_CallbackFunc)MyCallback, NULL));

    // We enable the DataEx callback because that's what the Runtime uses for JIT
    CHECK_CUPTI(cuptiEnableCallback(1, sub, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuModuleLoadDataEx));
    CHECK_CUPTI(cuptiEnableCallback(1, sub, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuModuleLoadData));

    // IMPORTANT: You must compile your CMake project with:
    // -gencode arch=compute_86,code=compute_86
    // This forces JIT, which forces the driver to call cuModuleLoadDataEx

    std::cout << "Triggering Kernel..." << std::endl;
    my_test_kernel<<<1,1>>>();
    cudaDeviceSynchronize();

    cuptiUnsubscribe(sub);
    return 0;
}
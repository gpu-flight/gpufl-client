#include <iostream>
#include <cuda_runtime.h>
#include <cupti.h>

// Simple empty callback
void CUPTIAPI MyCallback(void* userdata, CUpti_CallbackDomain domain,
                         CUpti_CallbackId cbid, const CUpti_CallbackData* cbInfo) {}

int main() {
    std::cout << "--- STARTING CONFLICT CHECK ---" << std::endl;

    // 1. Force CUDA Initialization
    cudaFree(0);

    // 2. Try to Subscribe (The "Soft" Lock)
    CUpti_SubscriberHandle subscriber;
    CUptiResult resSub = cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)MyCallback, nullptr);

    if (resSub != CUPTI_SUCCESS) {
        const char* err; cuptiGetResultString(resSub, &err);
        std::cerr << "FAIL: cuptiSubscribe error: " << err << std::endl;
        return 1;
    }
    std::cout << "PASS: cuptiSubscribe success." << std::endl;

    // 3. Try to Enable Activity (The "Hard" Lock - This is where you fail)
    CUptiResult resAct = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);

    if (resAct == CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED) {
        std::cerr << "\n!!! CONFLICT DETECTED !!!" << std::endl;
        std::cerr << "Another tool is holding the GPU Hardware Counters." << std::endl;
        std::cerr << "It is NOT your code. It is an environment variable or background service." << std::endl;

        // Check for common hijackers
        if (std::getenv("CUDA_INJECTION64_PATH"))
            std::cerr << "Culprit: CUDA_INJECTION64_PATH is set!" << std::endl;
        if (std::getenv("COMPUTE_SANITIZER_ConfigPath"))
            std::cerr << "Culprit: COMPUTE_SANITIZER is set!" << std::endl;
        if (std::getenv("NSIGHT_CUDA_DEBUGGER"))
            std::cerr << "Culprit: Nsight Debugger is active!" << std::endl;

        return 1;
    } else if (resAct != CUPTI_SUCCESS) {
        const char* err; cuptiGetResultString(resAct, &err);
        std::cerr << "FAIL: cuptiActivityEnable error: " << err << std::endl;
        return 1;
    }

    std::cout << "PASS: cuptiActivityEnable success." << std::endl;
    std::cout << "--- YOUR SYSTEM IS CLEAN ---" << std::endl;
    return 0;
}
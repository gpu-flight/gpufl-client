// Target program for the capture-gate end-to-end tests. Linked against the
// fake libcudart and libcuda; the scenario is argv[1].

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "fake_log.h"

extern "C" {
int cudaMalloc(void** pointer, size_t bytes);
int cudaSetDevice(int device);
int cudaDeviceSynchronize(void);
int cudaDriverGetVersion(int* version);
int cuInit(unsigned int flags);
int cuMemAlloc_v2(unsigned long long* pointer, size_t bytes);
}

// A test-owned object: its "destructor" (an atexit handler registered first,
// so it runs last) marks it destroyed, and the late initializer touches it.
static int g_destroyed = 0;

extern "C" __attribute__((visibility("default"))) void TestOwnedTouch(void) {
    fake_log(g_destroyed ? "touch:after_destroy" : "touch:alive");
}

static void destroy_test_owned(void) {
    g_destroyed = 1;
    fake_log("destroyed");
}

static void app_late_atexit(void) { fake_log("app_late_atexit"); }

int main(int argc, char** argv) {
    // Registered before any CUDA call: runs after the gate's exit guard.
    atexit(destroy_test_owned);
    const char* scenario = argc > 1 ? argv[1] : "runtime";
    void* p = NULL;
    unsigned long long dp = 0;
    int version = 0;

    if (strcmp(scenario, "runtime") == 0) {
        int rc = cudaMalloc(&p, 16);
        fake_log(rc == 0 ? "app:after_first_call" : "app:first_call_failed");
        cudaDeviceSynchronize();
    } else if (strcmp(scenario, "setdevice") == 0) {
        cudaSetDevice(0);
        fake_log("app:after_first_call");
    } else if (strcmp(scenario, "unwrapped") == 0) {
        // An unwrapped first call initializes the driver; a wrapped call
        // follows.
        cudaDriverGetVersion(&version);
        cudaMalloc(&p, 16);
        fake_log("app:after_first_call");
    } else if (strcmp(scenario, "driver") == 0) {
        cuInit(0);  // unwrapped by design
        cuMemAlloc_v2(&dp, 16);
        fake_log("app:after_first_call");
    } else if (strcmp(scenario, "manual") == 0) {
        // The Rust bootstrap order: MANUAL=1, cuInit (callback fires),
        // primary context, then the handshake.
        setenv("GPUFL_INJECT_MANUAL_INIT", "1", 1);
        void* driver = dlopen("libcuda.so.1", RTLD_NOW);
        int (*cu_init)(unsigned int) = (int (*)(unsigned int))dlsym(driver, "cuInit");
        int (*retain)(void**, int) = (int (*)(void**, int))dlsym(driver, "cuDevicePrimaryCtxRetain");
        int (*set_current)(void*) = (int (*)(void*))dlsym(driver, "cuCtxSetCurrent");
        void* context = NULL;
        cu_init(0);
        retain(&context, 0);
        set_current(context);
        int (*handshake)(void) =
            (int (*)(void))dlsym(RTLD_DEFAULT, "GpuFlightInitializeInjectionAfterCuda");
        int ok = handshake ? handshake() : -1;
        fake_log(ok == 1 ? "app:handshake_1" : "app:handshake_0");
    } else if (strcmp(scenario, "exit_early") == 0) {
        // Starts initialization and leaves main at once.
        cudaDriverGetVersion(&version);
        atexit(app_late_atexit);  // registered after the gate's guard
    } else if (strcmp(scenario, "none") == 0) {
        fake_log("app:no_cuda");
    }
    fake_log("app:return");
    return 0;
}

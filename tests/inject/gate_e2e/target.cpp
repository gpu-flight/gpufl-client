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
int cudaMallocManaged(void** pointer, size_t bytes, unsigned int flags);
int cudaMallocHost(void** pointer, size_t bytes);
int cudaHostAlloc(void** pointer, size_t bytes, unsigned int flags);
int cudaFree(void* pointer);
int cudaGetDeviceCount(int* count);
int cudaGetDeviceProperties_v2(void* properties, int device);
int cudaMemGetInfo(size_t* free_bytes, size_t* total_bytes);
int cudaEventCreate(void** event);
int cudaEventCreateWithFlags(void** event, unsigned int flags);
int cudaStreamCreate(void** stream);
int cudaStreamCreateWithFlags(void** stream, unsigned int flags);
int cudaStreamSynchronize(void* stream);
int cudaMemcpy(void* dst, const void* src, size_t bytes, int kind);
int cudaMemcpyAsync(void* dst, const void* src, size_t bytes, int kind, void* stream);
int cuModuleLoadData(void** module, const void* image);
int cuModuleLoadDataEx(void** module, const void* image, unsigned int count,
                       void* options, void** values);
int cuCtxSynchronize(void);
}

// "first:<name>": the named wrapped runtime call is the application's first
// CUDA call. Returns false for an unknown name.
static bool call_first(const char* name) {
    void* p = NULL;
    void* handle = NULL;
    size_t a = 0, b = 0;
    int count = 0;
    char properties[4096];
    char host[16] = {0};
    if (!strcmp(name, "cudaMalloc")) cudaMalloc(&p, 16);
    else if (!strcmp(name, "cudaMallocManaged")) cudaMallocManaged(&p, 16, 1);
    else if (!strcmp(name, "cudaMallocHost")) cudaMallocHost(&p, 16);
    else if (!strcmp(name, "cudaHostAlloc")) cudaHostAlloc(&p, 16, 0);
    else if (!strcmp(name, "cudaFree")) cudaFree(NULL);
    else if (!strcmp(name, "cudaSetDevice")) cudaSetDevice(0);
    else if (!strcmp(name, "cudaGetDeviceCount")) cudaGetDeviceCount(&count);
    else if (!strcmp(name, "cudaGetDeviceProperties_v2")) cudaGetDeviceProperties_v2(properties, 0);
    else if (!strcmp(name, "cudaMemGetInfo")) cudaMemGetInfo(&a, &b);
    else if (!strcmp(name, "cudaEventCreate")) cudaEventCreate(&handle);
    else if (!strcmp(name, "cudaEventCreateWithFlags")) cudaEventCreateWithFlags(&handle, 0);
    else if (!strcmp(name, "cudaStreamCreate")) cudaStreamCreate(&handle);
    else if (!strcmp(name, "cudaStreamCreateWithFlags")) cudaStreamCreateWithFlags(&handle, 0);
    else if (!strcmp(name, "cudaDeviceSynchronize")) cudaDeviceSynchronize();
    else if (!strcmp(name, "cudaStreamSynchronize")) cudaStreamSynchronize(NULL);
    else if (!strcmp(name, "cudaMemcpy")) cudaMemcpy(host, host + 8, 8, 0);
    else if (!strcmp(name, "cudaMemcpyAsync")) cudaMemcpyAsync(host, host + 8, 8, 0, NULL);
    else return false;
    return true;
}

// "driver:<name>": the host initializes the driver itself (unwrapped cuInit),
// then the named wrapped driver call follows and must wait for capture.
static bool call_driver(const char* name) {
    unsigned long long dp = 0;
    void* module = NULL;
    const char image[8] = {0};
    if (!strcmp(name, "cuMemAlloc_v2")) cuMemAlloc_v2(&dp, 16);
    else if (!strcmp(name, "cuModuleLoadData")) cuModuleLoadData(&module, image);
    else if (!strcmp(name, "cuModuleLoadDataEx")) cuModuleLoadDataEx(&module, image, 0, NULL, NULL);
    else if (!strcmp(name, "cuCtxSynchronize")) cuCtxSynchronize();
    else return false;
    return true;
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

    if (strncmp(scenario, "first:", 6) == 0) {
        if (!call_first(scenario + 6)) return 2;
        fake_log("app:after_first_call");
    } else if (strncmp(scenario, "driver:", 7) == 0) {
        cuInit(0);
        if (!call_driver(scenario + 7)) return 2;
        fake_log("app:after_first_call");
    } else if (strncmp(scenario, "sleep:", 6) == 0) {
        // A long-running host: driver init through an unwrapped call, then
        // it works for the given milliseconds before its next CUDA call.
        cudaDriverGetVersion(&version);
        usleep(static_cast<useconds_t>(atoi(scenario + 6)) * 1000u);
        cudaMalloc(&p, 16);
        fake_log("app:after_sleep");
    } else if (strcmp(scenario, "runtime") == 0) {
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
        // Starts initialization and leaves main before it finishes: shortly
        // after the worker's delay (so it is inside init), or during a long
        // warmup (so it is still in its delay).
        cudaDriverGetVersion(&version);
        atexit(app_late_atexit);  // registered after the gate's guard
        usleep(200 * 1000);
    } else if (strcmp(scenario, "none") == 0) {
        fake_log("app:no_cuda");
    }
    fake_log("app:return");
    return 0;
}

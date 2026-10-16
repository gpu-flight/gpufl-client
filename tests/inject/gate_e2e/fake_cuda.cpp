// A fake libcuda.so.1 for the CPU capture-gate end-to-end tests. Like the real
// driver, its first cuInit loads CUDA_INJECTION64_PATH and calls
// InitializeInjection from inside cuInit. FAKE_CUINIT selects a failure:
//   ok (default) | fail | nocallback | block (after the callback, forever)
//   | block_first (before anything, forever)

#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "fake_log.h"

extern "C" {

static pthread_mutex_t g_init_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_initialized = 0;

static const char* mode(void) {
    const char* value = getenv("FAKE_CUINIT");
    return value && *value ? value : "ok";
}

int cuInit(unsigned int flags) {
    (void)flags;
    if (strcmp(mode(), "block_first") == 0) {
        fake_log("cuInit:block_first");
        for (;;) pause();
    }
    pthread_mutex_lock(&g_init_lock);
    int first = !g_initialized;
    g_initialized = 1;
    if (first) {
        fake_log("cuInit:first");
        const char* inject = getenv("CUDA_INJECTION64_PATH");
        if (inject && *inject && strcmp(mode(), "nocallback") != 0) {
            void* handle = dlopen(inject, RTLD_NOW);
            int (*initialize)(void*) =
                handle ? (int (*)(void*))dlsym(handle, "InitializeInjection") : NULL;
            if (initialize) {
                fake_log("cuInit:callback");
                initialize(NULL);
            }
        }
    }
    pthread_mutex_unlock(&g_init_lock);
    if (strcmp(mode(), "fail") == 0) return 100;
    if (first && strcmp(mode(), "block") == 0) {
        fake_log("cuInit:block");
        for (;;) pause();
    }
    return 0;
}

int cuMemAlloc_v2(unsigned long long* pointer, size_t bytes) {
    (void)bytes;
    fake_log("cuMemAlloc_v2");
    *pointer = 0x1000;
    return 0;
}

int cuModuleLoadData(void** module, const void* image) {
    (void)image;
    fake_log("cuModuleLoadData");
    *module = (void*)0x2000;
    return 0;
}

int cuModuleLoadDataEx(void** module, const void* image, unsigned int count,
                       void* options, void** values) {
    (void)image; (void)count; (void)options; (void)values;
    fake_log("cuModuleLoadDataEx");
    *module = (void*)0x2000;
    return 0;
}

int cuDevicePrimaryCtxRetain(void** context, int device) {
    (void)device;
    fake_log("cuDevicePrimaryCtxRetain");
    *context = (void*)0x3000;
    return 0;
}

int cuCtxSetCurrent(void* context) {
    (void)context;
    fake_log("cuCtxSetCurrent");
    return 0;
}

int cuCtxSynchronize(void) {
    fake_log("cuCtxSynchronize");
    return 0;
}

int cuLaunchKernel(void* f, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
                   unsigned by, unsigned bz, unsigned shared, void* stream,
                   void** params, void** extra) {
    (void)f; (void)gx; (void)gy; (void)gz; (void)bx; (void)by; (void)bz;
    (void)shared; (void)stream; (void)params; (void)extra;
    fake_log("cuLaunchKernel");
    return 0;
}

}  // extern "C"

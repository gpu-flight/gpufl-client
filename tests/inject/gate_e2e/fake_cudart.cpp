// A fake libcudart for the CPU capture-gate end-to-end tests. Like the real
// runtime, its first call loads libcuda.so.1 and runs cuInit. FAKE_CUDART
// selects a failure of cudaGetDeviceCount after that cuInit:
//   ok (default) | fail (35) | block (forever)

#include <dlfcn.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "fake_log.h"

extern "C" {

static int runtime_init(void) {
    void* driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    int (*cu_init)(unsigned int) =
        driver ? (int (*)(unsigned int))dlsym(driver, "cuInit") : NULL;
    return cu_init ? cu_init(0) : 3;
}

int cudaGetDeviceCount(int* count) {
    fake_log("cudaGetDeviceCount");
    const int rc = runtime_init();
    const char* mode = getenv("FAKE_CUDART");
    if (mode && strcmp(mode, "fail") == 0) return 35;
    if (mode && strcmp(mode, "block") == 0) {
        fake_log("cudaGetDeviceCount:block");
        for (;;) pause();
    }
    if (rc != 0) return 3;
    *count = 1;
    return 0;
}

#define FAKE_RUNTIME_CALL(NAME, PARAMS)   \
    int NAME PARAMS {                     \
        fake_log(#NAME);                  \
        return runtime_init() == 0 ? 0 : 3; \
    }

FAKE_RUNTIME_CALL(cudaMalloc, (void** pointer, size_t bytes))
FAKE_RUNTIME_CALL(cudaFree, (void* pointer))
FAKE_RUNTIME_CALL(cudaSetDevice, (int device))
FAKE_RUNTIME_CALL(cudaDeviceSynchronize, (void))
FAKE_RUNTIME_CALL(cudaMemcpy, (void* dst, const void* src, size_t bytes, int kind))
// A call the gate does not wrap: it initializes the driver unowned.
FAKE_RUNTIME_CALL(cudaDriverGetVersion, (int* version))

}  // extern "C"

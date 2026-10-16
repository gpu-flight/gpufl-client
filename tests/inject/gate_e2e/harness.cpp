// The capture-gate end-to-end harness: built as a stand-in libgpufl_inject.so
// from the real capture_gate.cpp, capture_gate_linux.cpp and the typed CUDA
// wrappers (cuda_interpose_linux.cpp), with only gpufl::init replaced by a
// fake. FAKE_INIT selects its outcome:
//   ok (default) | fail | block (until FAKE_INIT_RELEASE_FILE exists, or forever)
// When the target exports TestOwnedTouch, the fake calls it after it returns
// from the block, so a test can prove it never runs after the target's own
// teardown.

#include <dlfcn.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>

#include "fake_log.h"
#include "gpufl/inject/capture_gate_linux.hpp"

namespace {

gpufl::inject::GateStep FakeInitialize() {
    fake_log("init:begin");
    const char* mode = std::getenv("FAKE_INIT");
    if (mode && std::strcmp(mode, "block") == 0) {
        const char* release = std::getenv("FAKE_INIT_RELEASE_FILE");
        for (;;) {
            if (release && access(release, F_OK) == 0) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        fake_log("init:released");
        if (auto touch = reinterpret_cast<void (*)()>(dlsym(RTLD_DEFAULT, "TestOwnedTouch"))) {
            touch();
        }
    }
    if (mode && std::strcmp(mode, "fail") == 0) {
        fake_log("init:fail");
        return {false, "init_failed"};
    }
    fake_log("init:ok");
    return {true, ""};
}

void FakeShutdown() { fake_log("shutdown"); }

const bool g_bound = [] {
    gpufl::inject::LinuxGateBindings bindings;
    bindings.initialize = FakeInitialize;
    bindings.after_initialization_exit = FakeShutdown;
    gpufl::inject::BindLinuxCaptureGate(bindings);
    return true;
}();

}  // namespace

extern "C" {

__attribute__((visibility("default"))) int InitializeInjection(void*) {
    gpufl::inject::LinuxInjectionCallback();
    return 0;
}

__attribute__((visibility("default"))) int GpuFlightInitializeInjectionAfterCuda() {
    return gpufl::inject::LinuxManualHandshake();
}

}  // extern "C"

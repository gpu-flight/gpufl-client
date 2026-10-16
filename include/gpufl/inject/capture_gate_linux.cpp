// NVIDIA calls InitializeInjection from inside the driver's own cuInit, so
// CUPTI cannot start there; a worker starts it after the driver is up, while
// the application thread continues into context and module setup. The L4
// sessions that lost every kernel timestamp (2026-10-16) are consistent with
// that overlap; the CUPTI-internal cause is not established. The gate makes
// the first supported CUDA call trigger driver init itself and wait, within
// one deadline, until capture is ready, and it records every route and failure
// it cannot cover.

#if !defined(_WIN32)

#include "gpufl/inject/capture_gate_linux.hpp"

#include <dlfcn.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <utility>

#include "gpufl/core/env_vars.hpp"

namespace gpufl::inject {
namespace {

std::atomic<GateStep (*)()> g_initialize{nullptr};
std::atomic<void (*)()> g_after_initialization_exit{nullptr};
std::atomic<void (*)()> g_on_gate_created{nullptr};

const char* EnvOrNull(const char* name) {
    const char* value = std::getenv(name);
    return value && *value ? value : nullptr;
}

int64_t MonotonicNowNs() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1'000'000'000LL + ts.tv_nsec;
}

std::string DlerrorText() {
    const char* error = dlerror();
    return error ? error : "unknown";
}

bool ResolvesIntoThisLibrary(void* symbol) {
    Dl_info target{};
    Dl_info self{};
    if (dladdr(symbol, &target) == 0 ||
        dladdr(reinterpret_cast<void*>(&MonotonicNowNs), &self) == 0) {
        return false;
    }
    return target.dli_fbase == self.dli_fbase;
}

using CuInitFn = int (*)(unsigned int);

// The driver's own cuInit: the already-loaded libcuda.so.1 first, then the
// global scope (libcuda may sit ahead of this library). Never loads a driver.
GateStep ResolveCuInit(CuInitFn& out) {
    dlerror();
    void* symbol = nullptr;
    if (void* handle = dlopen("libcuda.so.1", RTLD_NOLOAD | RTLD_LAZY)) {
        dlerror();
        symbol = dlsym(handle, "cuInit");
        dlclose(handle);  // drops only the reference RTLD_NOLOAD took
    }
    if (!symbol) {
        dlerror();
        symbol = dlsym(RTLD_DEFAULT, "cuInit");
        if (!symbol) return {false, "cuinit_symbol_missing:" + DlerrorText()};
    }
    if (ResolvesIntoThisLibrary(symbol)) return {false, "cuinit_resolves_to_self"};
    out = reinterpret_cast<CuInitFn>(symbol);
    return {true, ""};
}

// Driver route trigger, and the worker's post-driver barrier: cuInit is
// idempotent and blocks on the driver's init lock.
GateStep CheckedCuInit() {
    CuInitFn fn = nullptr;
    GateStep found = ResolveCuInit(fn);
    if (!found.ok) return found;
    const int rc = fn(0);
    if (rc != 0) return {false, "cuinit_failed:" + std::to_string(rc)};
    return {true, ""};
}

// Runtime route trigger: the real cudaGetDeviceCount from the same RTLD_NEXT
// chain the wrapper forwards to, so the driver is the one cudart uses. It runs
// cuInit without creating a context.
GateStep RuntimeTrigger() {
    using CudaGetDeviceCountFn = int (*)(int*);
    static auto* fn =
        reinterpret_cast<CudaGetDeviceCountFn>(dlsym(RTLD_NEXT, "cudaGetDeviceCount"));
    if (!fn) return {false, "runtime_symbol_missing"};
    int count = 0;
    const int rc = fn(&count);
    if (rc != 0) return {false, "runtime_init_failed:" + std::to_string(rc)};
    return {true, ""};
}

GateStep BoundInitialize() {
    GateStep (*initialize)() = g_initialize.load(std::memory_order_acquire);
    if (!initialize) return {false, "initializer_unbound"};
    return initialize();
}

struct StatusTarget {
    std::string dir;
    std::string nonce;
    bool debug = false;
    std::atomic<int> printed_readiness{-1};
};

StatusTarget& Status() {
    static auto* target = [] {
        auto* t = new StatusTarget();
        if (const char* dir = EnvOrNull(env::kInjectStatusDir)) t->dir = dir;
        if (const char* nonce = EnvOrNull(env::kInjectStatusNonce)) t->nonce = nonce;
        if (const char* debug = EnvOrNull(env::kDebugOutput)) {
            t->debug = std::strcmp(debug, "1") == 0;
        }
        return t;
    }();
    return *target;
}

// Logger-independent: the status file and one stderr line under GPUFL_DEBUG.
void PublishStatus(const GateSnapshot& snapshot) {
    StatusTarget& target = Status();
    const long pid = static_cast<long>(::getpid());
    if (!target.dir.empty()) {
        // Per pid: a child that inherited the environment writes its own file.
        const std::string path =
            target.dir + "/status-" + std::to_string(pid) + ".json";
        (void)WriteGateStatusFile(path, GateStatusJson(snapshot, target.nonce, pid));
    }
    const int readiness = static_cast<int>(snapshot.readiness);
    if (target.debug && snapshot.readiness != GateReadiness::Pending &&
        target.printed_readiness.exchange(readiness) != readiness) {
        std::fprintf(stderr, "[gpufl] capture readiness: %s owner=%s reason=%s\n",
                     GateReadinessName(snapshot.readiness),
                     GateOwnerName(snapshot.owner),
                     snapshot.reason.empty() ? "-" : snapshot.reason.c_str());
    }
}

// atexit: either every initialization participant has returned and ordinary
// teardown may run, or one is still live after the deadline and the process
// ends here, without the handlers and destructors that would run under it.
// Handlers the application registered after this one already ran before it.
void ExitGuard() {
    if (!LinuxCaptureGate().AwaitAtExit()) std::_Exit(kInitWedgedExitCode);
    if (void (*after)() = g_after_initialization_exit.load(std::memory_order_acquire)) {
        after();
    }
}

void Enter(const bool driver_route) {
    if (!LinuxInjectionRequested()) return;  // an accidental preload stays inert
    CaptureGate& gate = LinuxCaptureGate();
    if (gate.Settled()) return;
    gate.Enter(driver_route ? CheckedCuInit : RuntimeTrigger);
}

}  // namespace

void BindLinuxCaptureGate(const LinuxGateBindings& bindings) {
    g_initialize.store(bindings.initialize, std::memory_order_release);
    g_after_initialization_exit.store(bindings.after_initialization_exit,
                                      std::memory_order_release);
    g_on_gate_created.store(bindings.on_gate_created, std::memory_order_release);
}

bool LinuxInjectionRequested() {
    static const bool requested = [] {
        const char* sentinel = EnvOrNull(env::kInject);
        return sentinel && std::strcmp(sentinel, "1") == 0;
    }();
    return requested;
}

CaptureGate& LinuxCaptureGate() {
    // Process lifetime: a late worker, or a call at exit, never touches a
    // destroyed gate.
    static auto* gate = [] {
        GateHooks hooks;
        hooks.now_ns = MonotonicNowNs;
        hooks.spawn = [](std::function<void()> body) {
            try {
                std::thread(std::move(body)).detach();
                return true;
            } catch (...) {
                return false;
            }
        };
        hooks.driver_barrier = CheckedCuInit;
        hooks.initialize = BoundInitialize;
        hooks.publish = PublishStatus;
        auto* created = new CaptureGate(GateConfigFromEnv(nullptr), std::move(hooks));
        if (void (*notify)() = g_on_gate_created.load(std::memory_order_acquire)) {
            notify();
        }
        // Registered before init can register shutdown, so shutdown runs first.
        std::atexit(ExitGuard);
        return created;
    }();
    return *gate;
}

void LinuxInjectionCallback() {
    if (LinuxInjectionRequested()) LinuxCaptureGate().OnInjectionCallback();
}

int LinuxManualHandshake() {
    if (!LinuxInjectionRequested()) return 0;
    return LinuxCaptureGate().ManualHandshake(CheckedCuInit);
}

}  // namespace gpufl::inject

extern "C" {

// Cross-TU capture-gate hooks used by the typed CUDA interpose translation
// unit, one per route. Hidden inside the library; only CUDA ABI symbols are
// public.
__attribute__((visibility("hidden"))) void GpuFlightCaptureGateRuntime() {
    gpufl::inject::Enter(/*driver_route=*/false);
}

__attribute__((visibility("hidden"))) void GpuFlightCaptureGateDriver() {
    gpufl::inject::Enter(/*driver_route=*/true);
}

}  // extern "C"

#endif  // !defined(_WIN32)

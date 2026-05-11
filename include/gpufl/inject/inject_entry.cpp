// Entry point for libgpufl_inject.so — the shared library the launcher
// (`gpufl trace`) preloads into the target via LD_PRELOAD and
// CUDA_INJECTION64_PATH.
//
// Two entry paths, both routing through one idempotent init:
//   1. __attribute__((constructor)) — runs at ld.so dlopen time, before
//      main(). First-chance hook; captures CUDA work that happens
//      before any framework profiler initializes.
//   2. extern "C" int InitializeInjection(void*) — NVIDIA's official
//      CUDA injection ABI; libcuda calls this after cuInit. Second
//      chance, in case the constructor was bypassed (e.g. lib wasn't
//      LD_PRELOAD'd but only set as CUDA_INJECTION64_PATH).
//
// Both go through a std::once_flag so init runs at most once.
//
// Linux-only: __attribute__((constructor)) is a GCC/Clang extension,
// and the launcher uses LD_PRELOAD which is a Linux/glibc concept.
// CMake gates this TU on UNIX AND NOT APPLE.

#include "gpufl/inject/inject_entry.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unistd.h>

#include "gpufl/gpufl.hpp"
#include "gpufl/core/monitor.hpp"  // ProfilingEngine enum

namespace {

std::once_flag g_init_once;
std::atomic<bool> g_init_ok{false};

// Non-empty env value, or std::nullopt if unset / empty.
const char* envOrNull(const char* name) {
    const char* v = std::getenv(name);
    return (v && *v) ? v : nullptr;
}

gpufl::ProfilingEngine parseEngine(const std::string& s,
                                   gpufl::ProfilingEngine fallback) {
    if (s == "pc-sampling")           return gpufl::ProfilingEngine::PcSampling;
    if (s == "sass-metrics")          return gpufl::ProfilingEngine::SassMetrics;
    if (s == "pc-sampling-with-sass") return gpufl::ProfilingEngine::PcSamplingWithSass;
    if (s == "none")                  return gpufl::ProfilingEngine::None;
    return fallback;
}

void writeCompletionByteIfRequested() {
    const char* fd_str = envOrNull(gpufl::inject::kEnvCompletionFd);
    if (!fd_str) return;
    int fd = std::atoi(fd_str);
    if (fd <= 0) return;
    char ok = 1;
    // Best-effort; if the pipe is closed (launcher gave up), ignore.
    ssize_t n = ::write(fd, &ok, 1);
    (void)n;
    ::close(fd);
}

void shutdownAndSignal() {
    if (g_init_ok.load(std::memory_order_acquire)) {
        gpufl::shutdown();
    }
    writeCompletionByteIfRequested();
}

void doInjectInit() {
    // Sentinel guard — set by the launcher. Without it, treat the
    // preload as accidental (e.g. `LD_PRELOAD=...:libgpufl_inject.so`
    // leaking into a shell) and return silently.
    const char* sentinel = envOrNull(gpufl::inject::kEnvSentinel);
    if (!sentinel || std::strcmp(sentinel, "1") != 0) return;

    // Pick the profile preset.
    gpufl::InitOptions opts;
    if (const char* p = envOrNull(gpufl::inject::kEnvProfile)) {
        if (std::strcmp(p, gpufl::inject::kProfileLight) == 0) {
            opts = gpufl::light_mode_default_options();
        } else {
            // "comprehensive" (default) and "monitoring-only" both
            // start from the comprehensive baseline; finer-grained
            // monitoring-only tuning lands in Phase 4.
            opts = gpufl::injection_mode_default_options();
        }
    } else {
        opts = gpufl::injection_mode_default_options();
    }

    // Layered overrides — env vars beat preset defaults but lose to
    // gpufl::init()'s own remote-config / programmatic tuning that
    // happens downstream of this struct.
    if (const char* v = envOrNull(gpufl::inject::kEnvAppName)) {
        opts.app_name = v;
    }
    if (const char* v = envOrNull(gpufl::inject::kEnvLogDir)) {
        // Init expects a file path; the launcher provides the dir,
        // we tack on app.log so log_rotator's three NDJSON channels
        // (app.device.log / app.scope.log / app.system.log) land in
        // the launcher's chosen dir.
        opts.log_path = std::string(v) + "/app.log";
    }
    if (const char* v = envOrNull(gpufl::inject::kEnvProfilingEngine)) {
        opts.profiling_engine = parseEngine(v, opts.profiling_engine);
    }

    // gpufl::init() reads the rest of the GPUFL_* env vars itself
    // (backend_url, api_key, api_path, remote_upload, config_name).
    // No need to plumb them through here.

    if (gpufl::init(opts)) {
        g_init_ok.store(true, std::memory_order_release);
        std::atexit(shutdownAndSignal);
    } else {
        // init() already logs the failure via its own debug logger.
        // Keep injection silent on the user-facing stderr so we don't
        // pollute the target program's output.
    }
}

}  // namespace

extern "C" {

// First-chance entry: ld.so runs us before main(). Risk: CUPTI
// subscription before cuInit may not be supported on all driver/CUPTI
// combinations. The Day-1 spike (Phase 0.1) confirms whether this
// path can stand alone or whether we must wait for InitializeInjection.
[[gnu::constructor]] static void gpuflInjectCtor() {
    std::call_once(g_init_once, doInjectInit);
}

// Second-chance entry: NVIDIA's CUDA_INJECTION64_PATH ABI. libcuda
// loads this lib when cuInit runs, then calls InitializeInjection
// with a function table (which we ignore — gpufl::init() drives CUPTI
// directly via its own subscriber). Returning 0 means "ok, proceed".
//
// Idempotent with the constructor via the once_flag, so whichever
// fires first wins; the other is a no-op.
//
// Default visibility so the symbol survives the SO's CXX_VISIBILITY_PRESET=hidden.
__attribute__((visibility("default")))
int InitializeInjection(void* /*funcTable*/) {
    std::call_once(g_init_once, doInjectInit);
    return 0;
}

}  // extern "C"

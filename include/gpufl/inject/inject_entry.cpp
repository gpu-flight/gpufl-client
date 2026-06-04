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
#include "gpufl/upload/upload_logs.hpp"  // gpufl::uploadLogs for --upload

namespace {

std::once_flag g_init_once;
std::atomic<bool> g_init_ok{false};

// Captured during init for the atexit upload path (--upload). g_log_path
// mirrors InitOptions.log_path; both are read only in shutdownAndSignal,
// which runs after doInjectInit completed under g_init_once.
std::string g_log_path;
std::atomic<bool> g_do_upload{false};

// Non-empty env value, or std::nullopt if unset / empty.
const char* envOrNull(const char* name) {
    const char* v = std::getenv(name);
    return (v && *v) ? v : nullptr;
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

        // --upload: ship the just-written NDJSON to the backend. This is
        // the forward path (gpufl::uploadLogs), not the deprecated
        // opts.remote_upload shim. All network I/O happens here, after
        // shutdown() flushed the logs and the GPU workload is done, so it
        // can never affect the target's exit code or perf. Best-effort:
        // any failure is reported via gpufl's own debug log, never raised.
        if (g_do_upload.load(std::memory_order_relaxed) && !g_log_path.empty()) {
            gpufl::UploadOptions uopts;
            uopts.log_path = g_log_path;
            if (const char* v = envOrNull("GPUFL_BACKEND_URL")) uopts.backend_url = v;
            if (const char* v = envOrNull("GPUFL_API_KEY"))     uopts.api_key = v;
            if (const char* v = envOrNull("GPUFL_API_PATH"))    uopts.api_path = v;
            uopts.report_progress = false;  // don't pollute the target's stderr
            (void)gpufl::uploadLogs(uopts);
        }
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
        } else if (std::strcmp(p, gpufl::inject::kProfileMonitoringOnly) == 0) {
            opts = gpufl::monitoring_mode_default_options();
        } else {
            // "comprehensive" (default) — full injection capture (Deep
            // engine + most observability flags on).
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
    // gpufl::init() reads the rest of the GPUFL_* env vars itself,
    // including GPUFL_PROFILING_ENGINE — it is the single string->enum
    // engine parser (see gpufl.cpp). It also reads backend_url, api_key,
    // api_path, config_name. No need to plumb any of them through here.

    // Capture what the atexit upload path needs (see shutdownAndSignal).
    g_log_path = opts.log_path;
    if (const char* u = envOrNull(gpufl::inject::kEnvUpload)) {
        g_do_upload.store(std::strcmp(u, "1") == 0, std::memory_order_relaxed);
    }

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

// First-chance entry: ld.so runs us before main().
//
// **Disabled by default.** Phase 0.1 spike (2026-05-11) confirmed
// what the source plan suspected: CUPTI subscription from a
// pre-`cuInit` constructor segfaults on no-CUDA targets (the lib is
// preloaded into `echo`, libcuda is dragged in as a DT_NEEDED, CUPTI
// subscribe tries to touch driver state that doesn't exist, SIGSEGV).
// Falling back to `InitializeInjection` only — libcuda calls that
// after `cuInit`, so a no-CUDA target like `echo` never triggers init
// (target runs clean, empty trace dir, no crash, matching
// verification 1.12.4).
//
// Set `GPUFL_INJECT_USE_CONSTRUCTOR=1` to opt back in — useful for
// workloads where the first CUDA call happens deep in third-party
// code we want to catch the lead-up to, and where the toolchain has
// been verified to tolerate pre-cuInit subscribe.
[[gnu::constructor]] static void gpuflInjectCtor() {
    const char* opt_in = std::getenv("GPUFL_INJECT_USE_CONSTRUCTOR");
    if (!opt_in || std::strcmp(opt_in, "1") != 0) return;
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

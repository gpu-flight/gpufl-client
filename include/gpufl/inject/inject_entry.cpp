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

#include "gpufl/core/env_vars.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  define GPUFL_INJECT_EXPORT __declspec(dllexport)
#else
#  include <dlfcn.h>
#  include <unistd.h>
#  define GPUFL_INJECT_EXPORT __attribute__((visibility("default")))
#endif

#ifndef NVTX_NO_IMPL
#define NVTX_NO_IMPL
#endif
#include <nvtx3/nvToolsExt.h>

#include "gpufl/gpufl.hpp"
#include "gpufl/core/activity_record.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/debug_logger.hpp"  // GFL_LOG_DEBUG (teardown tracing)
#include "gpufl/core/monitor.hpp"
#include "gpufl/core/teardown_flag.hpp"  // setProcessExitTeardown (Windows)
#include "gpufl/upload/upload_logs.hpp"  // gpufl::uploadLogs for --upload

namespace {

std::once_flag g_init_once;
std::once_flag g_deferred_wait_atexit_once;
std::once_flag g_shutdown_atexit_once;
std::atomic<bool> g_init_ok{false};
std::atomic<bool> g_deferred_init_started{false};
std::atomic<bool> g_deferred_init_finished{false};
std::atomic<bool> g_shutdown_started{false};
std::mutex g_deferred_init_mutex;
std::condition_variable g_deferred_init_cv;

// Captured during init for the atexit upload path (--upload). g_log_path
// mirrors InitOptions.log_path; both are read only in shutdownAndSignal,
// which runs after doInjectInit completed under g_init_once.
std::string g_log_path;
std::atomic<bool> g_do_upload{false};

struct ProcessScopeState {
    bool active = false;
    bool perf_scope = false;
    uint64_t instance_id = 0;
    uint32_t name_id = 0;
    std::string name;
};

ProcessScopeState& processScope() {
    static auto* state = new ProcessScopeState();
    return *state;
}

uint64_t nextProcessScopeId() {
    static std::atomic<uint64_t> next{1};
    return next.fetch_add(1, std::memory_order_relaxed);
}

std::string processScopeName() {
    if (const char* app = std::getenv(gpufl::env::kAppName)) {
        if (app[0] != '\0') return std::string("process:") + app;
    }
    return "process:gpufl_injected_target";
}

bool engineNeedsPerfScope() {
    const char* engine = std::getenv(gpufl::env::kProfilingEngine);
    if (!engine || engine[0] == '\0') return true;
    return std::strcmp(engine, "Monitor") != 0 && std::strcmp(engine, "Trace") != 0;
}

void beginProcessScope() {
    auto& state = processScope();
    if (state.active) return;

    state.active = true;
    state.perf_scope = engineNeedsPerfScope();
    state.instance_id = nextProcessScopeId();
    state.name = processScopeName();
    state.name_id = gpufl::Monitor::InternScopeName(state.name);

    gpufl::ScopeBatchRow row;
    row.ts_ns = gpufl::detail::GetTimestampNs();
    row.scope_instance_id = state.instance_id;
    row.name_id = state.name_id;
    row.event_type = 0;
    row.depth = 0;
    gpufl::Monitor::PushScopeRow(row);
    gpufl::Monitor::BeginProfilerScope(state.name.c_str());
    if (state.perf_scope) gpufl::Monitor::BeginPerfScope(state.name.c_str());
}

void endProcessScope() {
    auto& state = processScope();
    if (!state.active) return;

    gpufl::ScopeBatchRow row;
    row.ts_ns = gpufl::detail::GetTimestampNs();
    row.scope_instance_id = state.instance_id;
    row.name_id = state.name_id;
    row.event_type = 1;
    row.depth = 0;
    gpufl::Monitor::PushScopeRow(row);
    gpufl::Monitor::EndProfilerScope(state.name.c_str());
    if (state.perf_scope) gpufl::Monitor::EndPerfScope(state.name.c_str());
    state.active = false;
}

void waitForDeferredInit();

// Non-empty env value, or std::nullopt if unset / empty.
const char* envOrNull(const char* name) {
    const char* v = std::getenv(name);
    return (v && *v) ? v : nullptr;
}

int envIntOrDefault(const char* name, int fallback) {
    const char* v = envOrNull(name);
    if (!v) return fallback;
    char* end = nullptr;
    const long parsed = std::strtol(v, &end, 10);
    if (!end || end[0] != 0 || parsed < 0) return fallback;
    return static_cast<int>(parsed);
}

void sleepMs(int milliseconds) {
    if (milliseconds <= 0) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

void writeCompletionByteIfRequested() {
#ifndef _WIN32
    const char* fd_str = envOrNull(gpufl::env::kInjectCompletionFd);
    if (!fd_str) return;
    int fd = std::atoi(fd_str);
    if (fd <= 0) return;
    char ok = 1;
    // Best-effort; if the pipe is closed (launcher gave up), ignore.
    ssize_t n = ::write(fd, &ok, 1);
    (void)n;
    ::close(fd);
#endif
    // Windows: the launcher does not wire a completion fd (Phase 2 stub).
}

void shutdownAndSignal() {
    bool expected = false;
    if (!g_shutdown_started.compare_exchange_strong(
            expected, true, std::memory_order_acq_rel)) {
        return;
    }

    GFL_LOG_DEBUG("inject: shutdownAndSignal begin");
#ifdef _WIN32
    // At Windows process exit the CUDA context is being torn down by cudart;
    // tell the backend to skip the driver-teardown calls (cudaDeviceSynchronize
    // / cuptiActivityFlushAll / nvmlShutdown) that would deadlock. See
    // gpufl/core/teardown_flag.hpp. NOT set on Linux (no race) nor by the
    // embedded SDK's mid-process shutdown (context alive, flush needed).
    gpufl::detail::setProcessExitTeardown(true);
#endif
    if (g_init_ok.load(std::memory_order_acquire)) {
        sleepMs(envIntOrDefault(gpufl::env::kInjectShutdownDelayMs, 0));
        GFL_LOG_DEBUG("inject: endProcessScope()");
        endProcessScope();
        GFL_LOG_DEBUG("inject: gpufl::shutdown()");
        gpufl::shutdown();
        GFL_LOG_DEBUG("inject: gpufl::shutdown() returned");

        // --upload: ship the just-written NDJSON to the backend. This is
        // the forward path (gpufl::uploadLogs), not the deprecated
        // opts.remote_upload shim. All network I/O happens here, after
        // shutdown() flushed the logs and the GPU workload is done, so it
        // can never affect the target's exit code or perf. Best-effort:
        // any failure is reported via gpufl's own debug log, never raised.
        if (g_do_upload.load(std::memory_order_relaxed) && !g_log_path.empty()) {
            gpufl::UploadOptions uopts;
            uopts.log_path = g_log_path;
            if (const char* v = envOrNull(gpufl::env::kBackendUrl)) uopts.backend_url = v;
            if (const char* v = envOrNull(gpufl::env::kApiKey))     uopts.api_key = v;
            if (const char* v = envOrNull(gpufl::env::kApiPath))    uopts.api_path = v;
            uopts.report_progress = false;  // don't pollute the target's stderr
            (void)gpufl::uploadLogs(uopts);
        }
    }
    writeCompletionByteIfRequested();
}

void waitForDeferredInit() {
    if (g_deferred_init_started.load(std::memory_order_acquire) &&
        !g_deferred_init_finished.load(std::memory_order_acquire)) {
        std::unique_lock lock(g_deferred_init_mutex);
        g_deferred_init_cv.wait(lock, [] {
            return g_deferred_init_finished.load(std::memory_order_acquire);
        });
    }
}

#ifndef _WIN32
// Boundary waits exist only to back the launch/sync interpose wrappers,
// which are LD_PRELOAD-only (Linux). On Windows there is no interpose, so
// these are compiled out to avoid unused-function diagnostics.
void waitForDeferredInitForMs(const int wait_ms) {
    if (wait_ms <= 0) return;
    if (g_deferred_init_started.load(std::memory_order_acquire) &&
        !g_deferred_init_finished.load(std::memory_order_acquire)) {
        std::unique_lock lock(g_deferred_init_mutex);
        g_deferred_init_cv.wait_for(
            lock, std::chrono::milliseconds(wait_ms), [] {
                return g_deferred_init_finished.load(std::memory_order_acquire);
            });
    }
}

void waitAtCudaSyncBoundary() {
    waitForDeferredInitForMs(envIntOrDefault(gpufl::env::kInjectSyncWaitMs, 15000));
}

void waitAtCudaLaunchBoundary() {
    waitForDeferredInitForMs(envIntOrDefault(gpufl::env::kInjectLaunchWaitMs, 15000));
}
#endif  // !_WIN32


struct NvtxDomainStorage {
    std::string name;
};

struct NvtxStringStorage {
    std::string value;
};

NvtxDomainStorage* nvtxDomainStorage(nvtxDomainHandle_t domain) {
    return reinterpret_cast<NvtxDomainStorage*>(domain);
}

NvtxStringStorage* nvtxStringStorage(nvtxStringHandle_t string) {
    return reinterpret_cast<NvtxStringStorage*>(string);
}

struct NvtxOpenRange {
    std::string name;
    std::string domain;
    int64_t start_ns = 0;
    uint32_t marker_id = 0;
};

thread_local std::vector<NvtxOpenRange> g_nvtx_stack;
std::atomic<uint32_t> g_next_nvtx_marker_id{1};

std::string nvtxMessageFromAttributes(const nvtxEventAttributes_t* attr) {
    if (!attr) return {};
    if (attr->messageType == NVTX_MESSAGE_TYPE_ASCII && attr->message.ascii) {
        return attr->message.ascii;
    }
    if (attr->messageType == NVTX_MESSAGE_TYPE_UNICODE && attr->message.unicode) {
        return "<unicode nvtx range>";
    }
    if (attr->messageType == NVTX_MESSAGE_TYPE_REGISTERED && attr->message.registered) {
        if (auto* s = nvtxStringStorage(attr->message.registered)) return s->value;
    }
    return {};
}

std::string nvtxDomainName(nvtxDomainHandle_t domain) {
    if (auto* d = nvtxDomainStorage(domain)) return d->name;
    return {};
}

int nvtxPushRange(std::string name, std::string domain = {}) {
    if (name.empty()) name = "<unnamed nvtx range>";
    const int depth = static_cast<int>(g_nvtx_stack.size());
    g_nvtx_stack.push_back(NvtxOpenRange{
        std::move(name),
        std::move(domain),
        gpufl::detail::GetTimestampNs(),
        g_next_nvtx_marker_id.fetch_add(1, std::memory_order_relaxed),
    });
    return depth;
}

int nvtxPopRange() {
    if (g_nvtx_stack.empty()) return NVTX_NO_PUSH_POP_TRACKING;
    NvtxOpenRange open = std::move(g_nvtx_stack.back());
    g_nvtx_stack.pop_back();

    const int64_t end_ns = gpufl::detail::GetTimestampNs();
    gpufl::ActivityRecord rec{};
    rec.type = gpufl::TraceType::NVTX_MARKER;
    std::snprintf(rec.name, sizeof(rec.name), "%s", open.name.c_str());
    rec.cpu_start_ns = open.start_ns;
    rec.duration_ns = std::max<int64_t>(0, end_ns - open.start_ns);
    rec.corr_id = open.marker_id;
    std::snprintf(rec.user_scope, sizeof(rec.user_scope), "%s", open.domain.c_str());
    gpufl::Monitor::PushActivityRecord(rec);
    return static_cast<int>(g_nvtx_stack.size());
}

NvtxFunctionTable nvtxFunctionTable(NvtxGetExportTableFunc_t getExportTable,
                                    NvtxCallbackModule module) {
    if (!getExportTable) return nullptr;
    const auto callbacks = static_cast<const NvtxExportTableCallbacks*>(
        getExportTable(NVTX_ETID_CALLBACKS));
    if (!callbacks) return nullptr;
    NvtxFunctionTable table = nullptr;
    unsigned int table_size = 0;
    if (!callbacks->GetModuleFunctionTable(module, &table, &table_size)) return nullptr;
    return table;
}

namespace nvtx_injection_impl {
int RangePushA(const char* message) {
    return nvtxPushRange(message ? message : "");
}

int RangePushW(const wchar_t*) {
    return nvtxPushRange("<unicode nvtx range>");
}

int RangePushEx(const nvtxEventAttributes_t* attr) {
    return nvtxPushRange(nvtxMessageFromAttributes(attr));
}

int RangePop() {
    return nvtxPopRange();
}

nvtxDomainHandle_t DomainCreateA(const char* name) {
    return reinterpret_cast<nvtxDomainHandle_t>(new NvtxDomainStorage{name ? name : ""});
}

nvtxDomainHandle_t DomainCreateW(const wchar_t*) {
    return reinterpret_cast<nvtxDomainHandle_t>(new NvtxDomainStorage{"<unicode nvtx domain>"});
}

void DomainDestroy(nvtxDomainHandle_t domain) {
    delete nvtxDomainStorage(domain);
}

nvtxStringHandle_t DomainRegisterStringA(nvtxDomainHandle_t, const char* string) {
    return reinterpret_cast<nvtxStringHandle_t>(new NvtxStringStorage{string ? string : ""});
}

nvtxStringHandle_t DomainRegisterStringW(nvtxDomainHandle_t, const wchar_t*) {
    return reinterpret_cast<nvtxStringHandle_t>(new NvtxStringStorage{"<unicode nvtx string>"});
}

int DomainRangePushEx(nvtxDomainHandle_t domain, const nvtxEventAttributes_t* attr) {
    return nvtxPushRange(nvtxMessageFromAttributes(attr), nvtxDomainName(domain));
}

int DomainRangePop(nvtxDomainHandle_t) {
    return nvtxPopRange();
}
}  // namespace nvtx_injection_impl

void registerDeferredWaitAtexit() {
    std::call_once(g_deferred_wait_atexit_once, [] {
        // atexit handlers run in reverse registration order. Register this
        // wait guard early, then register shutdown only after init succeeds,
        // so shutdown runs first and this guard only covers failed/slow init.
        std::atexit(waitForDeferredInit);
    });
}

void registerShutdownAtexit() {
    std::call_once(g_shutdown_atexit_once, [] {
        std::atexit(shutdownAndSignal);
    });
}

void doInjectInit() {
    // Sentinel guard — set by the launcher. Without it, treat the
    // preload as accidental (e.g. `LD_PRELOAD=...:libgpufl_inject.so`
    // leaking into a shell) and return silently.
    const char* sentinel = envOrNull(gpufl::env::kInject);
    if (!sentinel || std::strcmp(sentinel, "1") != 0) return;

    // Pick the profile preset.
    gpufl::InitOptions opts;
    if (const char* p = envOrNull(gpufl::env::kInjectProfile)) {
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
    if (const char* v = envOrNull(gpufl::env::kAppName)) {
        opts.app_name = v;
    }
    if (const char* v = envOrNull(gpufl::env::kLogDir)) {
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
    if (const char* u = envOrNull(gpufl::env::kInjectUpload)) {
        g_do_upload.store(std::strcmp(u, "1") == 0, std::memory_order_relaxed);
    }

    if (gpufl::init(opts)) {
        if (envIntOrDefault(gpufl::env::kInjectProcessScope, 1) != 0) {
            beginProcessScope();
        }
        g_init_ok.store(true, std::memory_order_release);
        registerShutdownAtexit();
    } else {
        // init() already logs the failure via its own debug logger.
        // Keep injection silent on the user-facing stderr so we don't
        // pollute the target program's output.
    }
}


void markDeferredInitFinished() {
    {
        std::lock_guard lock(g_deferred_init_mutex);
        g_deferred_init_finished.store(true, std::memory_order_release);
    }
    g_deferred_init_cv.notify_all();
}

void startDeferredInjectInit() {
    registerDeferredWaitAtexit();

    bool expected = false;
    if (!g_deferred_init_started.compare_exchange_strong(
            expected, true, std::memory_order_acq_rel)) {
        return;
    }

    std::thread([] {
        // NVIDIA calls InitializeInjection from inside the CUDA driver
        // injection path. CUPTI subscription/activity setup can report
        // success there but later deliver no callbacks. Step out of that
        // callback frame before touching CUPTI.
        sleepMs(envIntOrDefault(gpufl::env::kInjectInitDelayMs, 1));
        try {
            std::call_once(g_init_once, doInjectInit);
        } catch (...) {
            // Injection must never throw through libcuda's callback path.
        }
        markDeferredInitFinished();
    }).detach();
}

}  // namespace

extern "C" {


GPUFL_INJECT_EXPORT int InitializeInjectionNvtx2(
    const NvtxGetExportTableFunc_t getExportTable) {
    const NvtxFunctionTable core = nvtxFunctionTable(getExportTable, NVTX_CB_MODULE_CORE);
    const NvtxFunctionTable core2 = nvtxFunctionTable(getExportTable, NVTX_CB_MODULE_CORE2);
    if (!core || !core2) return 0;

    *core[NVTX_CBID_CORE_RangePushA] =
        reinterpret_cast<NvtxFunctionPointer>(nvtx_injection_impl::RangePushA);
    *core[NVTX_CBID_CORE_RangePushW] =
        reinterpret_cast<NvtxFunctionPointer>(nvtx_injection_impl::RangePushW);
    *core[NVTX_CBID_CORE_RangePushEx] =
        reinterpret_cast<NvtxFunctionPointer>(nvtx_injection_impl::RangePushEx);
    *core[NVTX_CBID_CORE_RangePop] =
        reinterpret_cast<NvtxFunctionPointer>(nvtx_injection_impl::RangePop);

    *core2[NVTX_CBID_CORE2_DomainCreateA] =
        reinterpret_cast<NvtxFunctionPointer>(nvtx_injection_impl::DomainCreateA);
    *core2[NVTX_CBID_CORE2_DomainCreateW] =
        reinterpret_cast<NvtxFunctionPointer>(nvtx_injection_impl::DomainCreateW);
    *core2[NVTX_CBID_CORE2_DomainDestroy] =
        reinterpret_cast<NvtxFunctionPointer>(nvtx_injection_impl::DomainDestroy);
    *core2[NVTX_CBID_CORE2_DomainRegisterStringA] =
        reinterpret_cast<NvtxFunctionPointer>(nvtx_injection_impl::DomainRegisterStringA);
    *core2[NVTX_CBID_CORE2_DomainRegisterStringW] =
        reinterpret_cast<NvtxFunctionPointer>(nvtx_injection_impl::DomainRegisterStringW);
    *core2[NVTX_CBID_CORE2_DomainRangePushEx] =
        reinterpret_cast<NvtxFunctionPointer>(nvtx_injection_impl::DomainRangePushEx);
    *core2[NVTX_CBID_CORE2_DomainRangePop] =
        reinterpret_cast<NvtxFunctionPointer>(nvtx_injection_impl::DomainRangePop);
    return 1;
}

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
#ifndef _WIN32
[[gnu::constructor]] static void gpuflInjectCtor() {
    const char* opt_in = std::getenv(gpufl::env::kInjectUseConstructor);
    if (!opt_in || std::strcmp(opt_in, "1") != 0) return;
    std::call_once(g_init_once, doInjectInit);
}
#endif  // !_WIN32 — pre-cuInit constructor path is Linux-only; on Windows
        // it would run under the loader lock, so we rely on the
        // InitializeInjection ABI path (driver-invoked, lock-safe) instead.


// Second-chance entry: NVIDIA's CUDA_INJECTION64_PATH ABI. libcuda
// loads this lib when cuInit runs, then calls InitializeInjection
// with a function table (which we ignore — gpufl::init() drives CUPTI
// directly via its own subscriber). Returning 0 means "ok, proceed".
//
// Idempotent with the constructor via the once_flag, so whichever
// fires first wins; the other is a no-op.
//
// Exported so the driver resolves it: visibility("default") on Linux to
// survive the SO's hidden preset; __declspec(dllexport) on Windows.
GPUFL_INJECT_EXPORT
int InitializeInjection(void* /*funcTable*/) {
#ifdef _WIN32
    // Windows has no LD_PRELOAD interpose to serialize the target's first
    // kernel launch behind init (that's the Linux path's safety net). A
    // deferred (background-thread) init therefore runs CONCURRENTLY with the
    // app's CUDA work and its context teardown, and CUPTI/NVML setup deadlocks
    // against the driver lock (observed: init thread wedges, process becomes
    // unkillable). The driver calls InitializeInjection before the app's first
    // kernel, so initialize SYNCHRONOUSLY here — the standard CUPTI/Nsight
    // injection pattern — so init fully completes before any kernel runs.
    std::call_once(g_init_once, doInjectInit);
#else
    startDeferredInjectInit();
#endif
    return 0;
}

#ifndef _WIN32
// Launch/sync symbol interposition (wait-for-init + forward) is Linux/glibc
// only: it relies on LD_PRELOAD shadowing libcudart's symbols. Windows has
// no preload interposition, so these wrappers don't exist there — Windows
// injection relies solely on the CUDA injection ABI (InitializeInjection).
struct GpuflDim3 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
};

using CudaSync0Fn = int (*)();
using CudaStreamSyncFn = int (*)(void*);
using CudaLaunchKernelFn = int (*)(const void*, GpuflDim3, GpuflDim3, void**, std::size_t, void*);
using CudaLaunchKernelExCFn = int (*)(const void*, const void*, void**);
using CuLaunchKernelFn = int (*)(void*, unsigned int, unsigned int, unsigned int,
                                 unsigned int, unsigned int, unsigned int,
                                 unsigned int, void*, void**, void**);

// The launch/sync symbols below are interposed via LD_PRELOAD: our
// definition shadows libcudart's, and we forward to the real one via
// dlsym(RTLD_NEXT). The resolved pointer is cached in a function-local
// `static` (thread-safe magic-static init) so the link-map walk runs
// ONCE per symbol, not on every kernel launch — a PyTorch hot loop does
// tens of thousands of launches/sec, where a per-call dlsym is real
// overhead. RTLD_NEXT resolution order is fixed for the process lifetime,
// so the cache is always valid.
__attribute__((visibility("default")))
int __cudaLaunchKernel(const void* func, const GpuflDim3 gridDim, const GpuflDim3 blockDim,
                       void** args, const std::size_t sharedMem, void* stream) {
    waitAtCudaLaunchBoundary();
    static auto* fn = reinterpret_cast<CudaLaunchKernelFn>(dlsym(RTLD_NEXT, "__cudaLaunchKernel"));
    return fn ? fn(func, gridDim, blockDim, args, sharedMem, stream) : 0;
}

__attribute__((visibility("default")))
int cudaLaunchKernel(const void* func, const GpuflDim3 gridDim, GpuflDim3 blockDim,
                     void** args,
    const std::size_t sharedMem, void* stream) {
    waitAtCudaLaunchBoundary();
    static auto* fn = reinterpret_cast<CudaLaunchKernelFn>(dlsym(RTLD_NEXT, "cudaLaunchKernel"));
    return fn ? fn(func, gridDim, blockDim, args, sharedMem, stream) : 0;
}

__attribute__((visibility("default")))
int cudaLaunchKernelExC(const void* config, const void* func, void** args) {
    waitAtCudaLaunchBoundary();
    static auto* fn = reinterpret_cast<CudaLaunchKernelExCFn>(dlsym(RTLD_NEXT, "cudaLaunchKernelExC"));
    return fn ? fn(config, func, args) : 0;
}

__attribute__((visibility("default")))
int cuLaunchKernel(void* f, const unsigned int gridDimX, const unsigned int gridDimY,
    const unsigned int gridDimZ, const unsigned int blockDimX,
    const unsigned int blockDimY, const unsigned int blockDimZ,
    const unsigned int sharedMemBytes, void* hStream,
                   void** kernelParams, void** extra) {
    waitAtCudaLaunchBoundary();
    static auto* fn = reinterpret_cast<CuLaunchKernelFn>(dlsym(RTLD_NEXT, "cuLaunchKernel"));
    return fn ? fn(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                   blockDimZ, sharedMemBytes, hStream, kernelParams, extra)
              : 0;
}

__attribute__((visibility("default")))
int cudaDeviceSynchronize() {
    waitAtCudaSyncBoundary();
    static auto* fn = reinterpret_cast<CudaSync0Fn>(dlsym(RTLD_NEXT, "cudaDeviceSynchronize"));
    return fn ? fn() : 0;
}

__attribute__((visibility("default")))
int cuCtxSynchronize() {
    waitAtCudaSyncBoundary();
    static auto* fn = reinterpret_cast<CudaSync0Fn>(dlsym(RTLD_NEXT, "cuCtxSynchronize"));
    return fn ? fn() : 0;
}

__attribute__((visibility("default")))
int cudaStreamSynchronize(void* stream) {
    waitAtCudaSyncBoundary();
    static auto* fn = reinterpret_cast<CudaStreamSyncFn>(dlsym(RTLD_NEXT, "cudaStreamSynchronize"));
    return fn ? fn(stream) : 0;
}
#endif  // !_WIN32

}  // extern "C"

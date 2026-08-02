#include "gpufl.hpp"

#include "gpufl/core/env_vars.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include "gpufl/backends/host_collector.hpp"
#include "gpufl/core/client_startup.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/teardown_flag.hpp"  // detail::isProcessExitTeardown
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/deep_window.hpp"
#include "gpufl/core/deep_window_rules.hpp"
#include "gpufl/core/events.hpp"
#include "gpufl/core/logger/logger.hpp"
#include "gpufl/core/version.hpp"
#include "gpufl/upload/upload_logs.hpp"
// NOTE: we intentionally do NOT include <httplib.h> in this TU.
// httplib pulls in <winsock2.h>, which collides with the legacy
// <winsock.h> included transitively by <windows.h> (used below for
// VEH + admin detection). The version-discovery probe implementation
// lives in remote_config.cpp, which includes httplib first and avoids
// windows.h entirely.
#include "gpufl/core/model/lifecycle_model.hpp"
#include "gpufl/core/model/system_event_model.hpp"
#include "gpufl/core/monitor.hpp"
#include "gpufl/core/monitor_backend.hpp"
#include "gpufl/core/runtime.hpp"
#include "gpufl/core/segment_runtime.hpp"
#include "gpufl/core/session_bootstrap.hpp"
#include "gpufl/core/scope_registry.hpp"
#include "gpufl/report/text_report.hpp"

// NVTX (NVIDIA Tools Extension) - zero-overhead annotation library.
// When GPUFL_HAS_NVTX is defined (see CMakeLists NVTX block), GFL_SCOPE
// emits a paired nvtxRangePushA/Pop around its body. The range is:
//   - visible to Nsight Systems (cross-tool validation),
//   - captured by GPUFlight's own CUPTI marker path (unified pipeline
//     with framework-emitted NVTX from PyTorch / cuDNN / etc.),
//   - zero-overhead when no profiler is attached.
// This is additive to the native scope_event - see plan's
// "GFL_SCOPE and NVTX are complementary layers" section.
#if GPUFL_HAS_NVTX
// Modern CUDA toolkits ship NVTX under the nvtx3/ prefix (CUDA 11.0+);
// older toolkits have it at the top level. Try v3 first, fall back.
#  if __has_include(<nvtx3/nvToolsExt.h>)
#    include <nvtx3/nvToolsExt.h>
#  elif __has_include(<nvToolsExt.h>)
#    include <nvToolsExt.h>
#  else
#    undef GPUFL_HAS_NVTX
#  endif
#endif

#if GPUFL_HAS_NVTX
// Guard flag: NVTX function-pointer table may contain null entries
// before init() has wired up CUPTI's injection, and after shutdown()
// has torn it down (or if CUPTI crashed mid-session). Calling
// nvtxRangePushA/Pop through a null entry produces an access
// violation (0xC0000005 reading 0x00000000).
//
// We flip this flag to true at the end of `init()` on success, and
// back to false at the start of `shutdown()`. ScopedMonitor destructors
// firing outside that window (e.g. during static teardown at process
// exit, or before init was ever called) skip NVTX entirely.
namespace gpufl {
std::atomic<bool> g_nvtx_available{false};
}  // namespace gpufl

// SEH-protected NVTX wrappers (Windows / MSVC).
//
// Access violations from NVTX are Windows STRUCTURED exceptions, not
// C++ exceptions - a normal try/catch cannot intercept them. We wrap
// the call in __try/__except and map any caught exception to rc = -1.
//
// IMPORTANT: these helpers MUST NOT contain C++ objects with
// destructors. MSVC forbids __try/__except in functions that also
// need C++ unwinding. Keep them minimal - just the raw NVTX call.
#if defined(_MSC_VER)
namespace gpufl {
namespace detail {

// Separate TU-local symbols so link-time code-gen can't inline our SEH
// around the caller's cleanup. `noinline` makes the intent explicit.
__declspec(noinline) inline int SafeNvtxRangePushA(const char* name) {
    __try {
        return nvtxRangePushA(name);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        // An AV here means NVTX's injection table has a null entry.
        // Mark NVTX unavailable so the rest of the session skips it.
        g_nvtx_available.store(false, std::memory_order_release);
        return -1;
    }
}

__declspec(noinline) inline int SafeNvtxRangePop() {
    __try {
        return ::nvtxRangePop();
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        g_nvtx_available.store(false, std::memory_order_release);
        return -1;
    }
}

}  // namespace detail
}  // namespace gpufl
#define GPUFL_SAFE_NVTX_PUSH(name) ::gpufl::detail::SafeNvtxRangePushA((name))
#define GPUFL_SAFE_NVTX_POP()      ::gpufl::detail::SafeNvtxRangePop()
#else
// Non-MSVC: rely on the guard flag alone. Clang/GCC on Linux can also
// trap SIGSEGV via signal handlers, but NVTX injection issues on
// Linux are rare enough that we don't add complexity here.
#define GPUFL_SAFE_NVTX_PUSH(name) ::nvtxRangePushA((name))
#define GPUFL_SAFE_NVTX_POP()      ::nvtxRangePop()
#endif  // _MSC_VER
#endif  // GPUFL_HAS_NVTX

#if GPUFL_HAS_NVTX
// nvtxRangePushA returns the 0-based nesting level on success, or a
// negative value on error (injection not initialized, internal NVTX
// error, etc). We route failures through GFL_LOG_ERROR - the project's
// standard logger - rather than fprintf. A static std::atomic<bool>
// guard caps the message at one per process so a persistent failure
// doesn't spam every GFL_SCOPE enter/exit.
#define GPUFL_NVTX_PUSH(name)                                                   \
    do {                                                                        \
        if (!::gpufl::g_nvtx_available.load(std::memory_order_acquire)) break;  \
        int _gpufl_nvtx_rc = GPUFL_SAFE_NVTX_PUSH((name));                      \
        if (_gpufl_nvtx_rc < 0) {                                               \
            static std::atomic<bool> _gpufl_nvtx_push_logged{false};            \
            if (!_gpufl_nvtx_push_logged.exchange(true)) {                      \
                GFL_LOG_ERROR(                                                  \
                    "nvtxRangePushA failed (rc=", _gpufl_nvtx_rc,               \
                    ") for '", (name),                                          \
                    "' - NVTX markers will not be captured for this session. " \
                    "Verify the CUPTI library exports "                         \
                    "InitializeInjectionNvtx2 and that "                        \
                    "NVTX_INJECTION64_PATH points to it.");                     \
            }                                                                   \
        }                                                                       \
    } while (0)

#define GPUFL_NVTX_POP()                                                        \
    do {                                                                        \
        if (!::gpufl::g_nvtx_available.load(std::memory_order_acquire)) break;  \
        int _gpufl_nvtx_rc = GPUFL_SAFE_NVTX_POP();                             \
        if (_gpufl_nvtx_rc < 0) {                                               \
            static std::atomic<bool> _gpufl_nvtx_pop_logged{false};             \
            if (!_gpufl_nvtx_pop_logged.exchange(true)) {                       \
                GFL_LOG_ERROR(                                                  \
                    "nvtxRangePop failed (rc=", _gpufl_nvtx_rc,                 \
                    ") - unbalanced push/pop, NVTX injection not "             \
                    "initialized, or caught structured exception from "         \
                    "NVTX injection table.");                                   \
            }                                                                   \
        }                                                                       \
    } while (0)
#else
#define GPUFL_NVTX_PUSH(name) ((void)0)
#define GPUFL_NVTX_POP()      ((void)0)
#endif

namespace gpufl {
std::atomic<int> g_systemSampleRateMs{0};
InitOptions g_opts;

static std::atomic<uint64_t> g_nextScopeId{1};

static uint64_t nextScopeId_() {
    return g_nextScopeId.fetch_add(1, std::memory_order_relaxed);
}

namespace {

// True if GPUFL_DISABLED env var is set to a truthy value. Mirrors the
// Python wrapper's vocabulary (`1`/`true`/`yes`/`on`, case-insensitive)
// so the two layers stay interchangeable. Empty / unset / anything else
// → false.
bool envDisabled_() {
    const char* v = std::getenv(gpufl::env::kDisabled);
    if (!v) return false;
    std::string s(v);
    // Trim ASCII whitespace.
    auto notWs = [](unsigned char c){ return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), notWs));
    s.erase(std::find_if(s.rbegin(), s.rend(), notWs).base(), s.end());
    // Lower-case.
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s == "1" || s == "true" || s == "yes" || s == "on";
}

}  // namespace

bool init(const InitOptions& opts) {
    // ── Disable kill switch ─────────────────────────────────────────────
    // Env var wins over the InitOptions field - it's the "force off
    // without editing code" knob. When disabled, we return immediately
    // BEFORE allocating anything: no Runtime, no Monitor, no logger, no
    // version-probe thread. Every other public entry point already
    // short-circuits when `runtime() == nullptr` (gpufl::shutdown,
    // systemStart/Stop, ScopedMonitor::init_/~ScopedMonitor), so the
    // disabled state cascades for free - no per-call-site checks needed.
    if (envDisabled_() || !opts.enabled) {
        // Keep g_opts at defaults so any caller reading it post-init
        // (rare - most paths gate on `runtime()` first) sees a clean
        // disabled-state shape.
        g_opts = InitOptions{};
        g_opts.enabled = false;
        return false;
    }

    g_opts = opts;
    detail::ClientStartup startup(g_opts);
    if (!startup.start()) return false;

#if GPUFL_HAS_NVTX
    // Enable NVTX push/pop now that CUPTI has wired up its injection.
    // Before this point, nvtxRangePop could dereference a null entry in
    // NVTX's function-pointer table and crash with an access violation.
    g_nvtx_available.store(true, std::memory_order_release);
#endif

    // Arm the launcher's time-based deep window, if one was asked for. Last,
    // so the delay is measured from a fully-initialized runtime, and safe
    // from here: it only records the request; the arm itself happens on the
    // app thread at the first launch past the delay.
    scheduleEnvDeepWindow();
    // Same reason as above: the rule needs a live backend to ask about deep
    // engine capability, and a metric window that starts from a running
    // runtime rather than from a half-built one.
    detail::DeepWindowRules::InstallFromEnv();

    GFL_LOG_DEBUG("Initialization complete!");
    return true;
}

void systemStart(std::string name) {
    Runtime* rt = runtime();
    const auto segment = rt ? rt->acquireSegmentContext() : nullptr;
    if (!segment || !segment->logger) return;
    {
        SystemStartEvent e;
        e.pid = detail::GetPid();
        e.app = rt->app_name;
        e.name = std::move(name);
        e.session_id = segment->session_id;
        e.ts_ns = detail::GetTimestampNs();
        if (rt->collector) e.devices = rt->collector->sampleAll();
        if (rt->host_collector) e.host = rt->host_collector->sample();
        segment->logger->write(model::SystemStartModel(e));
    }
    // Activate the sampler under the ref-counted model. If continuous
    // mode already took a baseline activation at init(), this stacks on
    // top of it (sampler keeps running). If continuous mode is off,
    // this is what actually starts the worker.
    if (g_opts.system_sample_rate_ms > 0 && rt->collector) {
        rt->sampler.activate();
    }
}

void systemStop(std::string name) {
    Runtime* rt = runtime();
    const auto segment = rt ? rt->acquireSegmentContext() : nullptr;
    if (!segment || !segment->logger) return;

    // Symmetric with systemStart: drop one activation. The sampler
    // worker only stops when the activation count hits zero, so
    // overlapping scopes / nested start/stop cycles compose correctly.
    if (g_opts.system_sample_rate_ms > 0 && rt->collector) {
        rt->sampler.deactivate();
    }

    SystemStopEvent e;
    e.pid = detail::GetPid();
    e.app = rt->app_name;
    e.session_id = segment->session_id;
    e.name = std::move(name);
    e.ts_ns = detail::GetTimestampNs();
    if (rt->collector) e.devices = rt->collector->sampleAll();
    if (rt->host_collector) e.host = rt->host_collector->sample();
    segment->logger->write(model::SystemStopModel(e));
}

void shutdown() {
#if GPUFL_HAS_NVTX
    // Flip the NVTX guard BEFORE tearing down CUPTI so any late scope
    // destructors (e.g. scopes still unwinding, or scopes running in
    // other threads during shutdown) skip the NVTX calls and cannot
    // crash when CUPTI's injection table is torn down.
    g_nvtx_available.store(false, std::memory_order_release);
#endif

    Runtime* rt = runtime();
    if (!rt) return;

    // Close a still-open deep window before anything is torn down, so its
    // engines disarm through the normal scope-stop path and the window
    // lands in the log rather than vanishing with the session.
    DeepWindow::Close(DeepWindowClose::SessionStop);

    GFL_LOG_DEBUG("Shutdown: begin -> sampler.shutdown()");
    // Stop the system sampler before CUPTI/backend teardown. The sampler can
    // be inside NVML while shutdown begins, especially in injection mode where
    // process exit races with late CUDA initialization. Joining it first keeps
    // backend shutdown from overlapping with telemetry collection.
    rt->sampler.shutdown();

    // Windows-injection process exit: the CUPTI release (cuptiPCSamplingStop/
    // Disable) can hang or crash against the context the driver is tearing down.
    // So drain + flush every batch, emit capabilities + the shutdown marker, and
    // CLOSE the log BEFORE that release - a teardown failure then costs no data.
    // Embedded/normal exits keep the clean order (Monitor::Shutdown releases
    // CUPTI first so its activity flush can deliver the final kernels).
    const bool processExit = detail::isProcessExitTeardown();

    if (processExit) {
        GFL_LOG_DEBUG("Shutdown: process-exit -> DrainAndFinalizeForExit()");
        Monitor::DrainAndFinalizeForExit();
    } else {
        GFL_LOG_DEBUG("Shutdown: sampler stopped -> Monitor::Shutdown()");
        Monitor::Shutdown();
    }
    GFL_LOG_DEBUG("Shutdown: monitor drained -> finalize logs");
    auto final_segment = rt->acquireSegmentContext("shutdown");
    if (!final_segment || !final_segment->logger) {
        GFL_LOG_ERROR("Shutdown: active segment context disappeared");
        set_runtime(nullptr);
        return;
    }

    // The optional "sampling_end" sample is skipped on Windows-injection exit:
    // collector->sampleAll() does slow NVML/NVAPI work against the context cudart
    // has already destroyed, and the process can be terminated mid-call. That
    // dropped the shutdown marker written just below (logs falsely "synthetic"
    // even though every kernel already flushed in DrainAndFinalizeForExit). The
    // marker is written first instead; the final metric sample is non-essential.
    if (g_opts.continuous_system_sampling && rt->collector && !processExit) {
        SystemStopEvent e;
        e.pid = detail::GetPid();
        e.app = rt->app_name;
        e.session_id = final_segment->session_id;
        e.name = "sampling_end";
        e.ts_ns = detail::GetTimestampNs();
        if (rt->collector) e.devices = rt->collector->sampleAll();
        if (rt->host_collector) e.host = rt->host_collector->sample();
        final_segment->logger->write(model::SystemStopModel(e));
    }

    const int64_t ended_ns = detail::GetTimestampNs();
    if (rt->segment_runtime) {
        // finish() seals the active context and waits for every writer lease.
        // This shutdown path is itself the final writer, so release its lease
        // before entering that barrier.
        final_segment.reset();
        rt->segment_runtime->finish(ended_ns);
    } else {
        ShutdownEvent se;
        se.pid = detail::GetPid();
        se.app = rt->app_name;
        se.session_id = final_segment->session_id;
        se.ts_ns = ended_ns;
        final_segment->logger->write(model::ShutdownEventModel(se));

        GFL_LOG_DEBUG("Shutdown: writing events done -> logger->close()");
        final_segment->logger->close();
        GFL_LOG_DEBUG("Shutdown: logger->close() returned");
    }

    // Logs are durable now. Release the CUPTI backend LAST so that if
    // cuptiPCSamplingStop/Disable hangs or crashes against the dying context,
    // the run's data is already saved.
    if (processExit) {
        GFL_LOG_DEBUG("Shutdown: process-exit -> ReleaseBackendForExit()");
        Monitor::ReleaseBackendForExit();
        GFL_LOG_DEBUG("Shutdown: ReleaseBackendForExit() returned");
    }

    set_runtime(nullptr);

    GFL_LOG_DEBUG("Shutdown complete!");
}

// ---- ScopedMonitor ----
ScopedMonitor::ScopedMonitor(std::string name)
    : ScopedMonitor(std::move(name), "", false) {}

ScopedMonitor::ScopedMonitor(std::string name, std::string tag)
    : ScopedMonitor(std::move(name), std::move(tag), false) {}

ScopedMonitor::ScopedMonitor(std::string name, const bool deep_profiling)
    : ScopedMonitor(std::move(name), "", deep_profiling) {}

ScopedMonitor::ScopedMonitor(std::string name, std::string tag,
                             bool /*deep_profiling*/)
    : name_(std::move(name)),
      tag_(std::move(tag)),
      pid_(detail::GetPid()),
      start_ns_(detail::GetTimestampNs()),
      scope_id_(nextScopeId_()) {
    init_(ScopeMeta{});  // no benchmark metadata
}

// Canonical 1.0.3+ ctor - single options object. Tag now lives
// inside ScopeMeta (was a separate parameter in the earlier draft)
// so the call site has a single source of truth and the variadic
// GFL_SCOPE macro can wrap any combination of fields in one
// ScopeMeta{...} literal.
ScopedMonitor::ScopedMonitor(std::string name, ScopeMeta meta)
    : name_(std::move(name)),
      tag_(std::move(meta.tag)),
      pid_(detail::GetPid()),
      start_ns_(detail::GetTimestampNs()),
      scope_id_(nextScopeId_()) {
    init_(meta);  // meta.tag is moved-from; init_ only reads repeat/warmup
}

void ScopedMonitor::init_(const ScopeMeta& meta) {
    Runtime* rt = runtime();
    if (!rt || !rt->hasSegmentContext()) return;

    auto& stack = getThreadScopeStack();
    const int depth = static_cast<int>(stack.size());
    stack.push_back(name_);

    // Scope-driven system-metric sampling. If continuous mode is off,
    // every scope takes one activation on the way in and balances it on
    // the way out. The Sampler's ref count handles nesting and overlap
    // correctly (overlapping scopes / explicit systemStart all stack).
    // We snapshot the decision at scope entry so the destructor can't
    // double-activate or miss a deactivation if continuous mode is
    // toggled mid-scope (which shouldn't happen, but defends against it).
    if (!g_opts.continuous_system_sampling &&
        g_opts.system_sample_rate_ms > 0 &&
        rt->collector) {
        rt->sampler.activate();
        sampler_activated_ = true;
    }

    const uint32_t name_id = Monitor::InternScopeName(name_);
    ScopeBatchRow row;
    row.ts_ns = start_ns_;
    row.scope_instance_id = scope_id_;
    row.name_id = name_id;
    row.event_type = 0;  // begin
    row.depth = depth;
    row.original_start_ns = start_ns_;
    // Benchmark metadata - 0/0 for the legacy ctors, populated for the
    // ScopeMeta overload. End row (in dtor) keeps these at 0; backend
    // joins by scope_instance_id to read the begin-row values.
    row.repeat = meta.repeat;
    row.warmup = meta.warmup;
    Monitor::PushScopeRow(row);

    // Scope callbacks are useful for both tracing and profiling backends.
    Monitor::BeginProfilerScope(name_.c_str());
    // Perf scope (Range Profiler / Perfworks). Harmless no-op for engines
    // that don't use perf scopes (PC / PM). Shared with DeepWindow so both
    // paths agree on which engines get one.
    detail::BeginPerfScopeIfEnabled(name_.c_str(), /*is_deep_window=*/false);
}

ScopedMonitor::~ScopedMonitor() {
    Runtime* rt = runtime();
    if (!rt || !rt->hasSegmentContext()) {
        // Best-effort: if the runtime is already gone but we'd taken a
        // sampler activation, we can't deactivate (no Sampler instance
        // to talk to). Sampler::shutdown() in gpufl::shutdown() will
        // have zeroed activations anyway, so we just drop the flag.
        sampler_activated_ = false;
        return;
    }

    // Balance the activation taken in init_() before any other dtor
    // work, so that the sampler can wind down promptly when the
    // outermost scope exits. The Sampler's ref count guarantees that
    // overlapping scopes / explicit systemStart keep it running until
    // all activators have released.
    if (sampler_activated_) {
        rt->sampler.deactivate();
        sampler_activated_ = false;
    }

    auto& stack = getThreadScopeStack();
    if (!stack.empty()) stack.pop_back();
    const int depth = static_cast<int>(stack.size());

    ScopeBatchRow row;
    row.scope_instance_id = scope_id_;
    row.name_id = Monitor::InternScopeName(name_);
    row.event_type = 1;  // end
    row.depth = depth;
    const int64_t end_ns = Monitor::CaptureScopeCloseTimestamp(scope_id_);
    row.ts_ns = end_ns;
    row.original_start_ns = start_ns_;
    Monitor::PushScopeRow(row);

    // Scopes are recorded via scope_event only - we no longer echo each
    // scope as an NVTX marker. That echo duplicated scope_event (the SPA
    // had to de-dupe it) and only the framework NVTX path remains useful.
    Monitor::EndProfilerScope(name_.c_str());
    detail::EndPerfScopeIfEnabled(name_.c_str(), pid_, start_ns_, end_ns,
                                  /*is_deep_window=*/false);
}
void generateReport(const std::string& output_path) {
    namespace fs = std::filesystem;

    const auto report_source = detail::lastSessionReportSource();
    fs::path p(report_source.log_path);
    if (p.extension() == ".log") {
        p.replace_extension();
    }

    report::TextReport::Options opts;
    const fs::path sessionDir = p / report_source.session_id;
    if (!report_source.session_id.empty() && fs::exists(sessionDir)) {
        opts.log_dir = sessionDir.string();
        opts.log_prefix.clear();
    } else {
        std::string dir = p.parent_path().string();
        if (dir.empty()) dir = ".";
        opts.log_dir = dir;
        opts.log_prefix = p.filename().string();
    }
    std::string text = report::TextReport(opts).generate();

    if (output_path.empty()) {
        std::cout << text;
    } else {
        std::ofstream file(output_path);
        if (file.is_open()) file << text;
    }
}

}  // namespace gpufl

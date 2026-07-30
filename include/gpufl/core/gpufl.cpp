#include "gpufl.hpp"

#include "gpufl/core/env_vars.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include "gpufl/backends/host_collector.hpp"
#include "gpufl/core/backend_factory.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/teardown_flag.hpp"  // detail::isProcessExitTeardown
#include "gpufl/core/config_file_loader.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/deep_window.hpp"
#include "gpufl/core/deep_window_rules.hpp"
#include "gpufl/core/dictionary_manager.hpp"
#include "gpufl/core/events.hpp"
#include "gpufl/core/logger/logger.hpp"
#include "gpufl/core/remote_config.hpp"
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
#include "gpufl/core/segmentation_config.hpp"
#include "gpufl/core/scope_registry.hpp"
#include "gpufl/report/text_report.hpp"
#if GPUFL_HAS_CUDA || defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

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

namespace {

MonitorBackendKind ToMonitorBackendKind(const BackendKind backend) {
    switch (backend) {
        case BackendKind::Nvidia:
            return MonitorBackendKind::Nvidia;
        case BackendKind::Amd:
            return MonitorBackendKind::Amd;
        case BackendKind::None:
            return MonitorBackendKind::None;
        case BackendKind::Auto:
        default:
            return MonitorBackendKind::Auto;
    }
}

}  // namespace

static std::string defaultLogPath_(const std::string& app) {
    // v1.2: log_path is a directory (sessions nest inside it as
    // `<log_path>/<session_id>/<channel>.log`). The legacy convention
    // returned "<app>.log" which the rotator stripped down to "<app>"
    // anyway; explicitly return just "<app>" so debug output and
    // `clean_logs(log_path=...)` show the same value as what's on
    // disk.
    return app;
}

// Remembered after init() for use by generateReport() after shutdown()
static std::string g_lastLogPath;
static std::string g_lastSessionId;
static std::string g_lastAppName;

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

bool windowsInjectedProcess_() {
#if defined(_WIN32)
    const char* injected = std::getenv(gpufl::env::kInject);
    return injected && std::string(injected) == "1";
#else
    return false;
#endif
}

bool parseNonNegativeEnv_(const char* key, uint64_t& value,
                          std::string& error) {
    value = 0;
    const char* raw = std::getenv(key);
    if (!raw || !*raw) return true;
    if (*raw == '-') {
        error = std::string(key) + " must be a non-negative integer";
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(raw, &end, 10);
    if (end == raw || *end != '\0' || errno == ERANGE) {
        error = std::string(key) + "='" + raw +
                "' is invalid (expected a non-negative integer)";
        return false;
    }
    value = static_cast<uint64_t>(parsed);
    return true;
}

bool isUuidV4_(const char* value) {
    if (!value || std::strlen(value) != 36) return false;
    for (size_t i = 0; i < 36; ++i) {
        if (i == 8 || i == 13 || i == 18 || i == 23) {
            if (value[i] != '-') return false;
        } else if (!std::isxdigit(static_cast<unsigned char>(value[i]))) {
            return false;
        }
    }
    if (value[14] != '4') return false;
    const char variant =
        static_cast<char>(std::tolower(static_cast<unsigned char>(value[19])));
    return variant == '8' || variant == '9' || variant == 'a' ||
           variant == 'b';
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

    // Read config file early - before anything uses the options
    {
        std::string configPath = g_opts.config_file;
        if (configPath.empty()) {
            if (const char* env = std::getenv(gpufl::env::kConfigFile)) configPath = env;
        }
        if (!configPath.empty()) {
            ConfigFileLoader::apply(g_opts, configPath);
        }
    }

    {
        // Resolve api_path (InitOptions value or GPUFL_API_PATH) and normalize
        // once - the version-discovery probe below appends to it. Backend
        // creds live on UploadOptions now, not InitOptions; the probe reads
        // GPUFL_BACKEND_URL straight from the environment.
        std::string apiPath = g_opts.api_path;
        if (apiPath.empty()) {
            if (const char* e = std::getenv(gpufl::env::kApiPath)) apiPath = e;
        }
        g_opts.api_path = normalizeApiPath(apiPath);
    }

    DebugLogger::setEnabled(g_opts.enable_debug_output);
    GFL_LOG_DEBUG("Initializing...");

    uint64_t segment_every_ms = 0;
    uint64_t segment_max_rows = 0;
    std::string segmentation_error;
    if (!parseNonNegativeEnv_(env::kSegmentEveryMs, segment_every_ms,
                              segmentation_error) ||
        !parseNonNegativeEnv_(env::kSegmentMaxRows, segment_max_rows,
                              segmentation_error)) {
        GFL_LOG_ERROR(segmentation_error);
        return false;
    }
    const bool segmented = segment_every_ms > 0 || segment_max_rows > 0;
    if (segment_every_ms >
        static_cast<uint64_t>((std::numeric_limits<int64_t>::max)())) {
        GFL_LOG_ERROR(env::kSegmentEveryMs,
                      " exceeds the supported signed 64-bit millisecond range");
        return false;
    }
    const char* env_run_id = std::getenv(env::kRunId);
    if (segmented && (!env_run_id || !*env_run_id)) {
        GFL_LOG_ERROR("Session segmentation requires GPUFL_RUN_ID. The "
                      "launcher must generate one run ID before starting the "
                      "target.");
        return false;
    }
    if (segmented && !isUuidV4_(env_run_id)) {
        GFL_LOG_ERROR(env::kRunId, "='", env_run_id,
                      "' is invalid (expected a UUIDv4)");
        return false;
    }
    if (segmented && !segmentation::kRuntimeReady) {
        GFL_LOG_ERROR(
            "This build does not include executable session segmentation.");
        return false;
    }

    if (runtime()) {
        GFL_LOG_DEBUG("Runtime already exists, shutting down first...");
        shutdown();
    }

    auto rt = std::make_unique<Runtime>();
    rt->app_name = g_opts.app_name.empty() ? "gpufl" : g_opts.app_name;
    rt->session_id = detail::GenerateSessionId();
    if (segmented) {
        rt->run_id = env_run_id;
        rt->segment_index = 0;
        rt->segment_every_ms = static_cast<int64_t>(segment_every_ms);
        rt->segment_max_rows = segment_max_rows;
    }
    rt->logger = std::make_shared<Logger>();
    rt->host_collector = std::make_unique<HostCollector>();

    const std::string logPath =
        g_opts.log_path.empty() ? defaultLogPath_(rt->app_name) : g_opts.log_path;

    Logger::Options logOpts;
    logOpts.base_path = logPath;
    // Threaded through so the rotator can write under
    // `<base_path>/<session_id>/<channel>.log` - v1.2 disk layout. The
    // uploader uses the directory name to discover sessions instead of
    // parsing job_start events out of flat log files.
    logOpts.session_id = rt->session_id;
    logOpts.system_sample_rate_ms = g_opts.system_sample_rate_ms;
    logOpts.flush_always = g_opts.flush_logs_always;
    if (const char* v = std::getenv(env::kFlushLogsAlways)) {
        std::string flag(v);
        std::transform(flag.begin(), flag.end(), flag.begin(),
                       [](unsigned char c) {
                           return static_cast<char>(std::tolower(c));
                       });
        if (flag == "1" || flag == "true" || flag == "yes" ||
            flag == "on") {
            logOpts.flush_always = true;
        }
    }
    if (const char* v = std::getenv(env::kLogRotateBytes)) {
        if (const auto bytes = std::strtoull(v, nullptr, 10); bytes > 0) {
            logOpts.rotate_bytes = static_cast<std::size_t>(bytes);
        }
    }
    if (const char* v = std::getenv(env::kLogRotateAfterMs)) {
        if (const auto ms = std::strtoll(v, nullptr, 10); ms > 0) {
            logOpts.rotate_after_ms = static_cast<std::int64_t>(ms);
        }
    }
    if (const char* v = std::getenv(env::kLogMaxSpoolBytes)) {
        logOpts.max_spool_bytes =
            static_cast<std::uint64_t>(std::strtoull(v, nullptr, 10));
    }
    if (const char* v = std::getenv(env::kLogMinFreeBytes)) {
        logOpts.min_free_bytes =
            static_cast<std::uint64_t>(std::strtoull(v, nullptr, 10));
    }

    g_lastLogPath = logPath;
    g_lastSessionId = rt->session_id;
    g_lastAppName = rt->app_name;

    GFL_LOG_DEBUG("Opening log file: ", logPath);
    if (!rt->logger->open(logOpts)) {
        GFL_LOG_ERROR("Failed to open logger at: ", logPath);
        return false;
    }
    auto initial_dictionary =
        segmented ? std::make_shared<SegmentDictionaryEmitter>() : nullptr;
    if (!rt->publishSegmentContext(std::make_shared<SegmentContext>(
            rt->run_id, rt->session_id, rt->segment_index,
            detail::GetTimestampNs(), rt->logger, initial_dictionary))) {
        GFL_LOG_ERROR("Failed to publish the initial segment context");
        rt->logger->close();
        return false;
    }

    // Fire-and-forget version-discovery probe. Hits
    // <backend_url><api_path>/info/version with 2s timeouts to detect
    // client/backend version drift early and emit a clear warning.
    // Must NEVER block init - detached, bounded by httplib timeouts.
    // Reads GPUFL_BACKEND_URL from the environment (creds live on
    // UploadOptions now); skipped when unset (offline / file-only mode).
    std::string probeUrl;
    if (const char* e = std::getenv(gpufl::env::kBackendUrl)) probeUrl = e;
    else if (const char* e2 = std::getenv(gpufl::env::kRemoteConfig)) probeUrl = e2;
    if (!probeUrl.empty()) {
        std::thread([url = probeUrl, ap = g_opts.api_path] {
            probeBackendVersion(url, ap);
        }).detach();
    }

    set_runtime(std::move(rt));
    rt = nullptr;  // rt is now moved

    GFL_LOG_DEBUG("Initializing Monitor (CUPTI)...");
    MonitorOptions mOpts;
    mOpts.enable_debug_output = g_opts.enable_debug_output;
    mOpts.profiling_engine = g_opts.profiling_engine;

    // Allow environment variable override: GPUFL_PROFILING_ENGINE.
    // Accepts exactly the six canonical engine names. Unrecognized
    // values are logged and ignored (the engine stays at whatever
    // g_opts set above) rather than silently doing nothing.
    if (const char* envEngine = std::getenv(gpufl::env::kProfilingEngine)) {
        const std::string val(envEngine);
        bool matched = true;
        if (val == "Monitor")                  mOpts.profiling_engine = ProfilingEngine::Monitor;
        else if (val == "Trace")               mOpts.profiling_engine = ProfilingEngine::Trace;
        else if (val == "PcSampling")          mOpts.profiling_engine = ProfilingEngine::PcSampling;
        else if (val == "SassMetrics")         mOpts.profiling_engine = ProfilingEngine::SassMetrics;
        else if (val == "PmSampling")          mOpts.profiling_engine = ProfilingEngine::PmSampling;
        else if (val == "RangeProfiler")       mOpts.profiling_engine = ProfilingEngine::RangeProfiler;
        else if (val == "RangeProfilerKernelReplay")
            mOpts.profiling_engine = ProfilingEngine::RangeProfilerKernelReplay;
        else if (val == "Deep")                mOpts.profiling_engine = ProfilingEngine::Deep;
        else matched = false;
        if (matched) {
            GFL_LOG_DEBUG("GPUFL_PROFILING_ENGINE override: ", val);
        } else {
            GFL_LOG_ERROR(
                "GPUFL_PROFILING_ENGINE='", val, "' is not a recognized "
                "engine name. Valid values: Monitor, Trace, PcSampling, "
                "SassMetrics, PmSampling, RangeProfiler, Deep. Keeping current engine "
                "selection.");
        }
    }

    // Allow environment override of the PC sampling period (log2 cycles/sample),
    // e.g. `gpufl trace --pc-sample-period`. The injection path has no other way
    // to reach pc_sampling_period. CUPTI accepts 5..31; out-of-range/garbage is
    // logged and ignored so a typo can't silently disable sampling.
    if (const char* v = std::getenv(gpufl::env::kPcSamplingPeriod)) {
        char* end = nullptr;
        const unsigned long n = std::strtoul(v, &end, 10);
        if (end != v && *end == '\0' && n >= 5 && n <= 31) {
            mOpts.pc_sampling_period = static_cast<uint32_t>(n);
            GFL_LOG_DEBUG("GPUFL_PC_SAMPLING_PERIOD override: 2^", n, " = ",
                          (1ul << n), " cycles/sample");
        } else {
            GFL_LOG_ERROR("GPUFL_PC_SAMPLING_PERIOD='", v, "' is invalid "
                          "(expected an integer 5..31). Keeping ",
                          mOpts.pc_sampling_period, ".");
        }
    }

    mOpts.kernel_sample_rate_ms = g_opts.kernel_sample_rate_ms;
    mOpts.enable_stack_trace = g_opts.enable_stack_trace;
    mOpts.enable_source_collection = g_opts.enable_source_collection;
    // Propagate the framework-correlation flag to the backend so
    // CuptiBackend::start can decide whether to enable
    // CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION.
    mOpts.enable_external_correlation = g_opts.enable_external_correlation;
    mOpts.enable_synchronization      = g_opts.enable_synchronization;
    mOpts.enable_memory_tracking      = g_opts.enable_memory_tracking;
    mOpts.enable_cuda_graphs_tracking = g_opts.enable_cuda_graphs_tracking;
    mOpts.pm_sampling_interval_us = g_opts.pm_sampling_interval_us;
    mOpts.pm_sampling_max_samples = g_opts.pm_sampling_max_samples;
    mOpts.pm_sampling_preset = g_opts.pm_sampling_preset;
    mOpts.pm_sampling_metrics = g_opts.pm_sampling_metrics;
    mOpts.pm_sampling_scope_only = g_opts.pm_sampling_scope_only;
    mOpts.deep_arm_mode = g_opts.deep_window_only ? DeepArmMode::WindowOnly
                                                  : DeepArmMode::Always;
    // GPUFL_DEEP_ARM reaches the injection path, which can't set InitOptions.
    if (const char* v = std::getenv(gpufl::env::kDeepArm)) {
        const std::string val(v);
        if (val == "window") {
            mOpts.deep_arm_mode = DeepArmMode::WindowOnly;
        } else if (val == "always") {
            mOpts.deep_arm_mode = DeepArmMode::Always;
        } else {
            GFL_LOG_ERROR("GPUFL_DEEP_ARM='", val,
                          "' is not recognized. Valid values: always, window. "
                          "Keeping current deep arm mode.");
        }
    }
    // WindowOnly subsumes the PM-specific gate: PM sampling already arms on
    // scope start, which is exactly what a window is.
    if (mOpts.deep_arm_mode == DeepArmMode::WindowOnly) {
        mOpts.pm_sampling_scope_only = true;
    }
    mOpts.backend_kind = ToMonitorBackendKind(g_opts.backend);

    // EAGER module loading is OPT-IN. By default we leave CUDA on its normal
    // LAZY loading; the per-architecture SASS exclusion gate
    // (GPUFL_SASS_EXCLUDE_ARCHS, in SassMetricsEngine) is the default guard
    // for the CUPTI lazy-patching deadlock - it disables SASS only on
    // architectures confirmed to hang, rather than paying EAGER's
    // whole-process startup/memory cost everywhere. EAGER remains available
    // as a per-run alternative: GPUFL_EAGER_MODULE_LOADING=1 forces it (it
    // finalizes every module up front, while the process is quiescent, so the
    // concurrent-launch finalize that triggers the deadlock never happens).
    //
    // This MUST run before the first CUDA call below (cudaGetDevice creates
    // the context, which reads CUDA_MODULE_LOADING). Honor a value the user
    // already set. Python callers apply the same opt-in earlier in
    // gpufl.init(); this covers the pure-C++ path.
    if (mOpts.profiling_engine == ProfilingEngine::SassMetrics ||
        mOpts.profiling_engine == ProfilingEngine::Deep) {
        const char* knobEnv = std::getenv(gpufl::env::kEagerModuleLoading);
        const std::string knob = knobEnv ? knobEnv : "";
        const bool optedIn = (knob == "1" || knob == "true" ||
                              knob == "yes" || knob == "on");
        if (optedIn && std::getenv(gpufl::env::kCudaModuleLoading) == nullptr) {
#if defined(_WIN32)
            _putenv_s(gpufl::env::kCudaModuleLoading, "EAGER");
#else
            setenv(gpufl::env::kCudaModuleLoading, "EAGER", /*overwrite=*/0);
#endif
            GFL_LOG_DEBUG("[gpufl] CUDA_MODULE_LOADING=EAGER set "
                          "(GPUFL_EAGER_MODULE_LOADING opt-in) for SASS/Deep.");
        }
    }

    // Auto-tune kernel_sample_rate_ms on older NVIDIA GPUs where SASS metric
    // overhead per kernel launch is much higher. A default 50ms on sm_86 can
    // lead to hundreds of captured kernels per second each carrying
    // instrumentation replay cost; bump to 200ms so users get a workable
    // profile without wild slowdowns. Users can still explicitly set a lower
    // value in InitOptions or via config file.
#if GPUFL_HAS_CUDA || defined(__CUDACC__)
    if (mOpts.kernel_sample_rate_ms > 0 && mOpts.kernel_sample_rate_ms < 200 &&
        (mOpts.profiling_engine == ProfilingEngine::SassMetrics ||
         mOpts.profiling_engine == ProfilingEngine::Deep)) {
        cudaDeviceProp prop{};
        int devId = 0;
        if (cudaGetDevice(&devId) == cudaSuccess &&
            cudaGetDeviceProperties(&prop, devId) == cudaSuccess) {
            const bool preSm120 = prop.major < 12;
            if (preSm120 && mOpts.kernel_sample_rate_ms == g_opts.kernel_sample_rate_ms) {
                GFL_LOG_DEBUG("[gpufl] Auto-tuning kernel_sample_rate_ms 50 -> 200 "
                              "on sm_", prop.major, prop.minor,
                              " (SASS metrics have significant per-launch overhead "
                              "on pre-sm_120 GPUs). Set the value explicitly to override.");
                mOpts.kernel_sample_rate_ms = 200;
            }
        }
    }
#endif

    Monitor::Initialize(mOpts);

    GFL_LOG_DEBUG("Starting Monitor...");
    Monitor::Start();
    GFL_LOG_DEBUG("Monitor started");

    Runtime* rt_ptr = runtime();
    const auto segment = rt_ptr ? rt_ptr->acquireSegmentContext() : nullptr;
    if (!segment) {
        GFL_LOG_ERROR("Missing active segment context before job_start");
        return false;
    }

    // Runtime backend selection
    std::string backendReason;
    auto backendCollectors =
        CreateBackendCollectors(g_opts.backend, &backendReason);
    rt_ptr->unified_gpu_collector = std::move(backendCollectors.unified_collector);
    rt_ptr->collector = std::move(backendCollectors.telemetry_collector);
    rt_ptr->static_info_collector =
        std::move(backendCollectors.static_info_collector);

    if (!rt_ptr->collector) {
        GFL_LOG_ERROR("Failed to initialize GPU backend: ", backendReason);
    }

    // init event with inventory (optional)
    InitEvent ie;
    ie.pid = detail::GetPid();
    ie.session_id = segment->session_id;
    ie.app = rt_ptr->app_name;
    ie.log_path = logPath;
    ie.ts_ns = detail::GetTimestampNs();
    // Collector may be unavailable on systems without NVML/ROCm. Guard usage.
    if (rt_ptr->collector) {
        ie.devices = rt_ptr->collector->sampleAll();
    }
    const bool skipStaticInfoDuringInject = windowsInjectedProcess_();
    if (rt_ptr->static_info_collector && !skipStaticInfoDuringInject) {
        ie.gpu_static_device_infos =
            rt_ptr->static_info_collector->sampleStaticInfo();
    } else if (skipStaticInfoDuringInject) {
        GFL_LOG_DEBUG("Skipping CUDA static GPU inventory during Windows injection init.");
    }
    ie.host = rt_ptr->host_collector->sample();

    ie.session_kind = ProfilingEngineSessionKind(mOpts.profiling_engine);
    ie.profiling_engine = ProfilingEngineWireName(mOpts.profiling_engine);
    ie.run_id = segment->run_id;
    ie.segment_index = segment->segment_index;

    // Multi-pass grouping (P1): the launcher's multi-pass driver tags each
    // child with GPUFL_ANALYSIS_ID + its 0-based GPUFL_PASS_INDEX and the
    // GPUFL_PASS_COUNT total so the backend can stitch the isolated passes
    // into one analysis. Read straight from the env here (same pattern as the
    // GPUFL_PROFILING_ENGINE override above). Absent → an ordinary single-pass
    // run: analysis_id stays empty and the three fields are omitted from
    // job_start (see InitEventModel), keeping single runs wire-identical.
    if (const char* envAnalysis = std::getenv(gpufl::env::kAnalysisId);
        envAnalysis && *envAnalysis) {
        ie.analysis_id = envAnalysis;
        if (const char* envIdx = std::getenv(gpufl::env::kPassIndex))
            ie.pass_index = std::atoi(envIdx);
        if (const char* envCnt = std::getenv(gpufl::env::kPassCount))
            ie.pass_count = std::atoi(envCnt);
        GFL_LOG_DEBUG("Multi-pass: analysis_id=", ie.analysis_id,
                      " pass ", ie.pass_index, "/", ie.pass_count);
    }

    segment->logger->write(model::InitEventModel(ie));

    if (segmented) {
        SegmentRuntime::Options segment_options;
        segment_options.runtime = rt_ptr;
        segment_options.logger_options = logOpts;
        segment_options.init_template = ie;
        segment_options.segment_every_ms = rt_ptr->segment_every_ms;
        segment_options.segment_max_rows = rt_ptr->segment_max_rows;
        rt_ptr->segment_runtime =
            std::make_shared<SegmentRuntime>(std::move(segment_options));
        if (!rt_ptr->segment_runtime->start()) {
            GFL_LOG_ERROR("Failed to start SegmentRuntime");
            shutdown();
            return false;
        }
    }

    // Configure the sampler with collectors / interval. This does NOT
    // start the worker - that happens via activate(), driven either by
    // the continuous-mode baseline activation below or by GFL_SCOPE
    // entry / systemStart() at runtime.
    if (g_opts.system_sample_rate_ms > 0 && rt_ptr->collector) {
        rt_ptr->sampler.configure(
            rt_ptr->app_name,
            [rt_ptr] { return rt_ptr->acquireSegmentContext(); },
            rt_ptr->collector, g_opts.system_sample_rate_ms,
            rt_ptr->host_collector.get(),
            [rt_ptr](const uint32_t index, const uint64_t rows,
                     const int64_t steady_ns, const int64_t event_ns) {
                if (rt_ptr->segment_runtime) {
                    rt_ptr->segment_runtime->noteRows(
                        index, rows, steady_ns, event_ns);
                }
            });
    }

    // Continuous mode: emit the SystemStart event and take the baseline
    // activation that keeps the sampler running until shutdown().
    if (g_opts.continuous_system_sampling && segment->logger) {
        SystemStartEvent e;
        e.pid = gpufl::detail::GetPid();
        e.app = rt_ptr->app_name;
        e.name = "sampling_start";
        e.session_id = segment->session_id;
        e.ts_ns = gpufl::detail::GetTimestampNs();
        if (rt_ptr->collector) e.devices = rt_ptr->collector->sampleAll();
        if (rt_ptr->host_collector) e.host = rt_ptr->host_collector->sample();
        segment->logger->write(model::SystemStartModel(e));
    }
    if (g_opts.continuous_system_sampling && g_opts.system_sample_rate_ms > 0 &&
        rt_ptr->collector) {
        rt_ptr->sampler.activate();
    }

    // Intentionally disabled - shutdown order must be explicit to avoid CUPTI
    // teardown races std::atexit(shutdown);

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
    auto final_segment = rt->acquireSegmentContext();
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

    fs::path p(g_lastLogPath);
    if (p.extension() == ".log") {
        p.replace_extension();
    }

    report::TextReport::Options opts;
    const fs::path sessionDir = p / g_lastSessionId;
    if (!g_lastSessionId.empty() && fs::exists(sessionDir)) {
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

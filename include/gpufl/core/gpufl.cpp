#include "gpufl.hpp"

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include "gpufl/backends/host_collector.hpp"
#include "gpufl/core/backend_factory.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/config_file_loader.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/events.hpp"
#include "gpufl/core/logger/http_log_sink.hpp"
#include "gpufl/core/logger/logger.hpp"
#include "gpufl/core/remote_config.hpp"
#include "gpufl/core/version.hpp"
// NOTE: we intentionally do NOT include <httplib.h> in this TU.
// httplib pulls in <winsock2.h>, which collides with the legacy
// <winsock.h> included transitively by <windows.h> (used below for
// VEH + admin detection). The fetchRemoteConfig() implementation lives
// in remote_config.cpp, which includes httplib first and avoids
// windows.h entirely. See plan file for details.
#include "gpufl/core/model/lifecycle_model.hpp"
#include "gpufl/core/model/perf_metric_model.hpp"
#include "gpufl/core/model/system_event_model.hpp"
#include "gpufl/core/monitor.hpp"
#include "gpufl/core/monitor_backend.hpp"
#include "gpufl/core/runtime.hpp"
#include "gpufl/core/scope_registry.hpp"
#include "gpufl/report/text_report.hpp"
#if GPUFL_HAS_CUDA || defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

// NVTX (NVIDIA Tools Extension) — zero-overhead annotation library.
// When GPUFL_HAS_NVTX is defined (see CMakeLists NVTX block), GFL_SCOPE
// emits a paired nvtxRangePushA/Pop around its body. The range is:
//   - visible to Nsight Systems (cross-tool validation),
//   - captured by GPUFlight's own CUPTI marker path (unified pipeline
//     with framework-emitted NVTX from PyTorch / cuDNN / etc.),
//   - zero-overhead when no profiler is attached.
// This is additive to the native scope_event — see plan's
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
// C++ exceptions — a normal try/catch cannot intercept them. We wrap
// the call in __try/__except and map any caught exception to rc = -1.
//
// IMPORTANT: these helpers MUST NOT contain C++ objects with
// destructors. MSVC forbids __try/__except in functions that also
// need C++ unwinding. Keep them minimal — just the raw NVTX call.
#if defined(_MSC_VER)
namespace gpufl {
namespace detail {

// Separate TU-local symbols so link-time code-gen can't inline our SEH
// around the caller's cleanup. `noinline` makes the intent explicit.
__declspec(noinline) inline int SafeNvtxRangePushA(const char* name) {
    __try {
        return ::nvtxRangePushA(name);
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
// error, etc). We route failures through GFL_LOG_ERROR — the project's
// standard logger — rather than fprintf. A static std::atomic<bool>
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
                    "' — NVTX markers will not be captured for this session. " \
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
                    ") — unbalanced push/pop, NVTX injection not "             \
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

// fetchRemoteConfig + its helpers (urlEncode, parseBackendBaseUrl,
// applyRemoteConfigToOpts) live in remote_config.cpp now — kept out of
// this TU so the httplib include chain (which pulls in winsock2.h on
// Windows) never collides with this file's <windows.h> usage for VEH
// and admin elevation checks. gpufl.cpp just declares its intent by
// including the header above and calling through.

}  // namespace

static std::string defaultLogPath_(const std::string& app) {
    return app + ".log";
}

// Remembered after init() for use by generateReport() after shutdown()
static std::string g_lastLogPath;
static std::string g_lastAppName;

static std::atomic<uint64_t> g_nextScopeId{1};

static uint64_t nextScopeId_() {
    return g_nextScopeId.fetch_add(1, std::memory_order_relaxed);
}

bool init(const InitOptions& opts) {
    g_opts = opts;

    // Read config file early — before anything uses the options
    {
        std::string configPath = g_opts.config_file;
        if (configPath.empty()) {
            if (const char* env = std::getenv("GPUFL_CONFIG_FILE")) configPath = env;
        }
        if (!configPath.empty()) {
            ConfigFileLoader::apply(g_opts, configPath);
        }
    }

    // Fetch remote named config — a dict of InitOptions overrides hosted
    // at `<backend_url><api_path>/config?config=<config_name>`. Mirrors what
    // the Python _fetch_remote_config() used to do; moved into C++ so
    // pure C++ consumers (e.g. compiled sass_divergence_demo) get the
    // same capability without a Python wrapper.
    //
    // Opt-in: we ONLY fetch when config_name is non-empty. Setting
    // backend_url + api_key alone is not enough — it could mean the
    // user only wants log upload, or nothing remote at all. Requiring
    // a name makes the fetch predictable and keeps "set the URL for
    // upload" from silently triggering a config pull.
    {
        std::string url  = g_opts.backend_url;
        std::string key  = g_opts.api_key;
        std::string name = g_opts.config_name;
        std::string apiPath = g_opts.api_path;
        // Env-var fallbacks so scripts / containers can set these
        // without touching code.
        if (url.empty()) {
            if (const char* e = std::getenv("GPUFL_BACKEND_URL")) url = e;
            // Legacy name — accept for one release to ease migration.
            else if (const char* e2 = std::getenv("GPUFL_REMOTE_CONFIG")) url = e2;
        }
        if (key.empty()) {
            if (const char* e = std::getenv("GPUFL_API_KEY")) key = e;
        }
        if (name.empty()) {
            if (const char* e = std::getenv("GPUFL_CONFIG_NAME")) name = e;
        }
        if (apiPath.empty()) {
            if (const char* e = std::getenv("GPUFL_API_PATH")) apiPath = e;
        }
        // Normalize once — every downstream consumer (fetchRemoteConfig,
        // HttpLogSink, the discovery probe below) just appends after this.
        apiPath = normalizeApiPath(apiPath);
        // Reflect env-var-resolved values back onto g_opts so downstream
        // consumers (HttpLogSink wiring below) see consistent values.
        g_opts.backend_url  = url;
        g_opts.api_key      = key;
        g_opts.config_name  = name;
        g_opts.api_path     = apiPath;

        if (!name.empty() && !url.empty() && !key.empty()) {
            fetchRemoteConfig(url, apiPath, key, name, g_opts);
        }
    }

    DebugLogger::setEnabled(g_opts.enable_debug_output);
    GFL_LOG_DEBUG("Initializing...");
    if (runtime()) {
        GFL_LOG_DEBUG("Runtime already exists, shutting down first...");
        shutdown();
    }

    auto rt = std::make_unique<Runtime>();
    rt->app_name = g_opts.app_name.empty() ? "gpufl" : g_opts.app_name;
    rt->session_id = detail::GenerateSessionId();
    rt->logger = std::make_shared<Logger>();
    rt->host_collector = std::make_unique<HostCollector>();

    const std::string logPath =
        g_opts.log_path.empty() ? defaultLogPath_(rt->app_name) : g_opts.log_path;

    Logger::Options logOpts;
    logOpts.base_path = logPath;
    logOpts.system_sample_rate_ms = g_opts.system_sample_rate_ms;
    logOpts.flush_always = g_opts.flush_logs_always;

    g_lastLogPath = logPath;
    g_lastAppName = rt->app_name;

    GFL_LOG_DEBUG("Opening log file: ", logPath);
    if (!rt->logger->open(logOpts)) {
        GFL_LOG_ERROR("Failed to open logger at: ", logPath);
        return false;
    }

    // Optional HttpLogSink — posts every NDJSON line directly to the
    // backend alongside the file sink. Used for interactive contexts
    // (local dev, SSH, Jupyter) where deploying the monitor daemon is
    // heavy. File sink still writes normally so nothing is lost.
    bool remote_upload_enabled = g_opts.remote_upload;
    if (const char* env = std::getenv("GPUFL_REMOTE_UPLOAD")) {
        // Any non-empty, non-"0"/"false" value turns it on.
        const std::string v(env);
        remote_upload_enabled =
            !v.empty() && v != "0" && v != "false" && v != "False";
    }
    if (remote_upload_enabled) {
        if (g_opts.backend_url.empty() || g_opts.api_key.empty()) {
            GFL_LOG_ERROR(
                "remote_upload=true but backend_url or api_key is empty "
                "— skipping HttpLogSink. Set GPUFL_BACKEND_URL + "
                "GPUFL_API_KEY (or pass backend_url / api_key kwargs).");
        } else {
            HttpLogSink::Options httpOpts;
            httpOpts.base_url = g_opts.backend_url;
            httpOpts.api_path = g_opts.api_path;
            httpOpts.api_key  = g_opts.api_key;
            rt->logger->addSink(
                std::make_unique<HttpLogSink>(std::move(httpOpts)));
            GFL_LOG_DEBUG(
                "HttpLogSink attached — uploading to ",
                g_opts.backend_url, g_opts.api_path);
        }
    }

    // Fire-and-forget version-discovery probe. Hits
    // <backend_url><api_path>/info/version with 2s timeouts to detect
    // client/backend version drift early and emit a clear warning.
    // Must NEVER block init — detached, bounded by httplib timeouts.
    // Skipped when backend_url is empty (offline / file-only mode).
    if (!g_opts.backend_url.empty()) {
        std::thread([url = g_opts.backend_url,
                     ap  = g_opts.api_path]() {
            probeBackendVersion(url, ap);
        }).detach();
    }

    set_runtime(std::move(rt));
    rt = nullptr;  // rt is now moved

    GFL_LOG_DEBUG("Initializing Monitor (CUPTI)...");
    MonitorOptions mOpts;
    mOpts.collect_kernel_details = g_opts.enable_kernel_details;
    mOpts.enable_debug_output = g_opts.enable_debug_output;
    mOpts.profiling_engine = g_opts.profiling_engine;

    // Allow environment variable override: GPUFL_PROFILING_ENGINE
    if (const char* envEngine = std::getenv("GPUFL_PROFILING_ENGINE")) {
        const std::string val(envEngine);
        if (val == "None")               mOpts.profiling_engine = ProfilingEngine::None;
        else if (val == "PcSampling")    mOpts.profiling_engine = ProfilingEngine::PcSampling;
        else if (val == "SassMetrics")   mOpts.profiling_engine = ProfilingEngine::SassMetrics;
        else if (val == "RangeProfiler") mOpts.profiling_engine = ProfilingEngine::RangeProfiler;
        else if (val == "PcSamplingWithSass") mOpts.profiling_engine = ProfilingEngine::PcSamplingWithSass;
        GFL_LOG_DEBUG("GPUFL_PROFILING_ENGINE override: ", val);
    }
    mOpts.kernel_sample_rate_ms = g_opts.kernel_sample_rate_ms;
    if (mOpts.profiling_engine != ProfilingEngine::None) {
        mOpts.collect_kernel_details = true;
    }
    mOpts.enable_stack_trace = g_opts.enable_stack_trace;
    mOpts.enable_source_collection = g_opts.enable_source_collection;
    // Propagate the framework-correlation flag to the backend so
    // CuptiBackend::start can decide whether to enable
    // CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION.
    mOpts.enable_external_correlation = g_opts.enable_external_correlation;
    mOpts.enable_synchronization      = g_opts.enable_synchronization;
    mOpts.enable_memory_tracking      = g_opts.enable_memory_tracking;
    mOpts.enable_cuda_graphs_tracking = g_opts.enable_cuda_graphs_tracking;
    mOpts.backend_kind = ToMonitorBackendKind(g_opts.backend);

    // Auto-tune kernel_sample_rate_ms on older NVIDIA GPUs where SASS metric
    // overhead per kernel launch is much higher. A default 50ms on sm_86 can
    // lead to hundreds of captured kernels per second each carrying
    // instrumentation replay cost; bump to 200ms so users get a workable
    // profile without wild slowdowns. Users can still explicitly set a lower
    // value in InitOptions or via config file.
#if GPUFL_HAS_CUDA || defined(__CUDACC__)
    if (mOpts.kernel_sample_rate_ms > 0 && mOpts.kernel_sample_rate_ms < 200 &&
        (mOpts.profiling_engine == ProfilingEngine::SassMetrics ||
         mOpts.profiling_engine == ProfilingEngine::PcSamplingWithSass)) {
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
    ie.session_id = rt_ptr->session_id;
    ie.app = rt_ptr->app_name;
    ie.log_path = logPath;
    ie.ts_ns = detail::GetTimestampNs();
    // Collector may be unavailable on systems without NVML/ROCm. Guard usage.
    if (rt_ptr->collector) {
        ie.devices = rt_ptr->collector->sampleAll();
    }
    if (rt_ptr->static_info_collector) {
        ie.gpu_static_device_infos =
            rt_ptr->static_info_collector->sampleStaticInfo();
    }
    ie.host = rt_ptr->host_collector->sample();

    rt_ptr->logger->write(model::InitEventModel(ie));

    // Start sampler if enabled and collector exists
    if (g_opts.sampling_auto_start && rt_ptr->logger) {
        SystemStartEvent e;
        e.pid = gpufl::detail::GetPid();
        e.app = rt_ptr->app_name;
        e.name = "sampling_start";
        e.session_id = rt_ptr->session_id;
        e.ts_ns = gpufl::detail::GetTimestampNs();
        if (rt_ptr->collector) e.devices = rt_ptr->collector->sampleAll();
        if (rt_ptr->host_collector) e.host = rt_ptr->host_collector->sample();
        rt_ptr->logger->write(model::SystemStartModel(e));
    }
    if (g_opts.sampling_auto_start && g_opts.system_sample_rate_ms > 0 &&
        rt_ptr->collector) {
        rt_ptr->sampler.start(rt_ptr->app_name, rt_ptr->session_id,
                              rt_ptr->logger, rt_ptr->collector,
                              g_opts.system_sample_rate_ms, rt_ptr->app_name,
                              rt_ptr->host_collector.get());
    }

    // Intentionally disabled — shutdown order must be explicit to avoid CUPTI
    // teardown races std::atexit(shutdown);

#if GPUFL_HAS_NVTX
    // Enable NVTX push/pop now that CUPTI has wired up its injection.
    // Before this point, nvtxRangePop could dereference a null entry in
    // NVTX's function-pointer table and crash with an access violation.
    g_nvtx_available.store(true, std::memory_order_release);
#endif

    GFL_LOG_DEBUG("Initialization complete!");
    return true;
}

void systemStart(std::string name) {
    Runtime* rt = runtime();
    if (!rt || !rt->logger) return;
    {
        SystemStartEvent e;
        e.pid = detail::GetPid();
        e.app = rt->app_name;
        e.name = std::move(name);
        e.session_id = rt->session_id;
        e.ts_ns = detail::GetTimestampNs();
        if (rt->collector) e.devices = rt->collector->sampleAll();
        if (rt->host_collector) e.host = rt->host_collector->sample();
        rt->logger->write(model::SystemStartModel(e));
    }
    if (g_opts.system_sample_rate_ms > 0 && rt->collector) {
        rt->sampler.start(rt->app_name, rt->session_id, rt->logger,
                          rt->collector, g_opts.system_sample_rate_ms, name,
                          rt->host_collector.get());
    }
}

void systemStop(std::string name) {
    Runtime* rt = runtime();
    if (!rt || !rt->logger) return;

    rt->sampler.stop();

    SystemStopEvent e;
    e.pid = detail::GetPid();
    e.app = rt->app_name;
    e.session_id = rt->session_id;
    e.name = std::move(name);
    e.ts_ns = detail::GetTimestampNs();
    if (rt->collector) e.devices = rt->collector->sampleAll();
    if (rt->host_collector) e.host = rt->host_collector->sample();
    rt->logger->write(model::SystemStopModel(e));
}

void shutdown() {
#if GPUFL_HAS_NVTX
    // Flip the NVTX guard BEFORE tearing down CUPTI so any late scope
    // destructors (e.g. scopes still unwinding, or scopes running in
    // other threads during shutdown) skip the NVTX calls and cannot
    // crash when CUPTI's injection table is torn down.
    g_nvtx_available.store(false, std::memory_order_release);
#endif

    Monitor::Stop();
    Monitor::Shutdown();
    Runtime* rt = runtime();
    if (!rt) return;

    rt->sampler.stop();

    if (g_opts.sampling_auto_start && rt->collector) {
        SystemStopEvent e;
        e.pid = detail::GetPid();
        e.app = rt->app_name;
        e.session_id = rt->session_id;
        e.name = "sampling_end";
        e.ts_ns = detail::GetTimestampNs();
        if (rt->collector) e.devices = rt->collector->sampleAll();
        if (rt->host_collector) e.host = rt->host_collector->sample();
        rt->logger->write(model::SystemStopModel(e));
    }

    ShutdownEvent se;
    se.pid = detail::GetPid();
    se.app = rt->app_name;
    se.session_id = rt->session_id;
    se.ts_ns = detail::GetTimestampNs();
    rt->logger->write(model::ShutdownEventModel(se));

    rt->logger->close();
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
                             bool deep_profiling)
    : name_(std::move(name)),
      tag_(std::move(tag)),
      pid_(detail::GetPid()),
      start_ns_(detail::GetTimestampNs()),
      scope_id_(nextScopeId_()) {
    if (const Runtime* rt = runtime(); !rt || !rt->logger) return;

    auto& stack = getThreadScopeStack();
    const int depth = static_cast<int>(stack.size());
    stack.push_back(name_);

    const uint32_t name_id = Monitor::InternScopeName(name_);
    ScopeBatchRow row;
    row.ts_ns = start_ns_;
    row.scope_instance_id = scope_id_;
    row.name_id = name_id;
    row.event_type = 0;  // begin
    row.depth = depth;
    Monitor::PushScopeRow(row);

    // Scope callbacks are useful for both tracing and profiling backends.
    Monitor::BeginProfilerScope(name_.c_str());
    if (g_opts.profiling_engine != ProfilingEngine::None) {
        Monitor::BeginPerfScope(name_.c_str());
    }

    // Emit an NVTX range alongside the native scope_event. This makes the
    // scope visible to Nsight Systems and captured uniformly via CUPTI's
    // marker activity path (same pipeline that picks up PyTorch /
    // cuDNN / NCCL framework ranges). No-op if NVTX isn't available.
    GPUFL_NVTX_PUSH(name_.c_str());
}

ScopedMonitor::~ScopedMonitor() {
    // Pop the NVTX range first — symmetric with the push in the
    // constructor. Safe to call even if runtime() is gone; NVTX keeps
    // its own internal range stack.
    GPUFL_NVTX_POP();

    const Runtime* rt = runtime();
    if (!rt || !rt->logger) return;

    auto& stack = getThreadScopeStack();
    if (!stack.empty()) stack.pop_back();
    const int depth = static_cast<int>(stack.size());

    ScopeBatchRow row;
    row.ts_ns = detail::GetTimestampNs();
    row.scope_instance_id = scope_id_;
    row.name_id = Monitor::InternScopeName(name_);
    row.event_type = 1;  // end
    row.depth = depth;
    Monitor::PushScopeRow(row);

    Monitor::EndProfilerScope(name_.c_str());
    if (g_opts.profiling_engine != ProfilingEngine::None) {
        Monitor::EndPerfScope(
            name_.c_str());  // triggers EndPerfPassAndDecode first
        if (IMonitorBackend* b = Monitor::GetBackend()) {
            if (auto event_opt = b->TakeLastPerfEvent()) {
                PerfMetricEvent& pe = *event_opt;
                pe.pid = pid_;
                pe.app = rt->app_name;
                pe.session_id = rt->session_id;
                pe.name = name_;
                pe.start_ns = start_ns_;
                pe.end_ns = detail::GetTimestampNs();
                rt->logger->write(model::PerfMetricModel(pe));

                GFL_LOG_DEBUG("Log Perf Metric Event");
            }
        }
    }
}
void generateReport(const std::string& output_path) {
    namespace fs = std::filesystem;

    fs::path p(g_lastLogPath);
    std::string dir = p.parent_path().string();
    if (dir.empty()) dir = ".";

    std::string prefix = p.filename().string();
    if (prefix.size() > 4 && prefix.substr(prefix.size() - 4) == ".log")
        prefix = prefix.substr(0, prefix.size() - 4);

    report::TextReport::Options opts;
    opts.log_dir = dir;
    opts.log_prefix = prefix;
    std::string text = report::TextReport(opts).generate();

    if (output_path.empty()) {
        std::cout << text;
    } else {
        std::ofstream file(output_path);
        if (file.is_open()) file << text;
    }
}

}  // namespace gpufl

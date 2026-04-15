#include "gpufl.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "gpufl/backends/host_collector.hpp"
#include "gpufl/core/backend_factory.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/config_file_loader.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/events.hpp"
#include "gpufl/core/logger/logger.hpp"
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
    mOpts.backend_kind = ToMonitorBackendKind(g_opts.backend);
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

ScopedMonitor::ScopedMonitor(std::string name, bool deep_profiling)
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
}

ScopedMonitor::~ScopedMonitor() {
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

#include "gpufl.hpp"

#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "../backends/nvidia/cuda_collector.hpp"
#include "gpufl/backends/host_collector.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/events.hpp"
#include "gpufl/core/logger.hpp"
#include "gpufl/core/monitor.hpp"
#include "gpufl/core/runtime.hpp"
#include "gpufl/core/scope_registry.hpp"
#if GPUFL_HAS_CUDA || defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

#if GPUFL_ENABLE_NVIDIA && GPUFL_HAS_NVML
#include "gpufl/backends/nvidia/nvml_collector.hpp"
#endif

#if GPUFL_ENABLE_AMD && GPUFL_HAS_ROCM
#include "gpufl/backends/amd/rocm_collector.hpp"
#endif

namespace gpufl {
std::atomic<int> g_systemSampleRateMs{0};
InitOptions g_opts;

static std::string defaultLogPath_(const std::string& app) {
    return app + ".log";
}

static std::atomic<uint64_t> g_nextScopeId{1};

static uint64_t nextScopeId_() {
    return g_nextScopeId.fetch_add(1, std::memory_order_relaxed);
}

static std::shared_ptr<ISystemCollector<DeviceSample>> createCollector_(
    const BackendKind backend, std::string* reasonOut) {
    if (reasonOut) reasonOut->clear();

    auto setReason = [&](const std::string& r) {
        if (reasonOut && reasonOut->empty()) *reasonOut = r;
    };

    auto tryNvml = [&]() -> std::shared_ptr<ISystemCollector<DeviceSample>> {
#if GPUFL_ENABLE_NVIDIA && GPUFL_HAS_NVML
        return std::make_shared<gpufl::nvidia::NvmlCollector>();
#else
        setReason(
            "NVIDIA telemetry not available (GPUFL_ENABLE_NVIDIA=OFF or NVML "
            "not found).");
        return nullptr;
#endif
    };

    auto tryRocm = [&]() -> std::shared_ptr<ISystemCollector<DeviceSample>> {
#if GPUFL_ENABLE_AMD && GPUFL_HAS_ROCM
        return std::make_shared<gpufl::amd::RocmCollector>();
#else
        setReason(
            "AMD telemetry not available (GPUFL_ENABLE_AMD=OFF or ROCm not "
            "found).");
        return nullptr;
#endif
    };

    switch (backend) {
        case BackendKind::None:
            return nullptr;

        case BackendKind::Nvidia: {
            auto c = tryNvml();
            if (!c)
                setReason("Requested backend=nvidia but NVML is unavailable.");
            return c;
        }

        case BackendKind::Amd: {
            auto c = tryRocm();
            if (!c) setReason("Requested backend=amd but ROCm is unavailable.");
            return c;
        }

        case BackendKind::Auto:
        default: {
            // Prefer NVML first, then ROCm
            if (auto c = tryNvml()) return c;
            if (auto c = tryRocm()) return c;
            setReason(
                "No GPU backend available (NVML/ROCm not compiled in or not "
                "available).");
            return nullptr;
        }
    }
}

bool init(const InitOptions& opts) {
    g_opts = opts;
    DebugLogger::setEnabled(opts.enable_debug_output);
    GFL_LOG_DEBUG("Initializing...");
    if (runtime()) {
        GFL_LOG_DEBUG("Runtime already exists, shutting down first...");
        shutdown();
    }

    auto rt = std::make_unique<Runtime>();
    rt->app_name = opts.app_name.empty() ? "gpufl" : opts.app_name;
    rt->session_id = detail::GenerateSessionId();
    rt->logger = std::make_shared<Logger>();
    rt->host_collector = std::make_unique<HostCollector>();
    rt->cuda_collector = std::make_unique<nvidia::CudaCollector>();

    const std::string logPath =
        opts.log_path.empty() ? defaultLogPath_(rt->app_name) : opts.log_path;

    Logger::Options logOpts;
    logOpts.base_path = logPath;
    logOpts.system_sample_rate_ms = opts.system_sample_rate_ms;

    GFL_LOG_DEBUG("Opening log file: ", logPath);
    if (!rt->logger->open(logOpts)) {
        GFL_LOG_ERROR("Failed to open logger at: ", logPath);
        return false;
    }

    set_runtime(std::move(rt));
    rt = nullptr;  // rt is now moved

    GFL_LOG_DEBUG("Initializing Monitor (CUPTI)...");
    MonitorOptions mOpts;
    mOpts.collect_kernel_details = opts.enable_kernel_details;
    mOpts.enable_debug_output = opts.enable_debug_output;
    mOpts.is_profiling = opts.enable_profiling;
    mOpts.kernel_sample_rate_ms = opts.kernel_sample_rate_ms;
    if (mOpts.is_profiling) {
        mOpts.collect_kernel_details =
            true;  // if profiling is on, then it should be true.
    }
    mOpts.enable_stack_trace = opts.enable_stack_trace;
    Monitor::Initialize(mOpts);

    GFL_LOG_DEBUG("Starting Monitor...");
    Monitor::Start();
    GFL_LOG_DEBUG("Monitor started");

    Runtime* rt_ptr = runtime();

    // Runtime backend selection
    std::string backendReason;
    rt_ptr->collector = createCollector_(opts.backend, &backendReason);
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
    if (opts.backend == BackendKind::Auto ||
        opts.backend == BackendKind::Nvidia) {
#if GPUFL_HAS_CUDA
        ie.cuda_static_device_infos = rt_ptr->cuda_collector->sampleAll();
#endif
    }
    ie.host = rt_ptr->host_collector->sample();

    rt_ptr->logger->logInit(ie);

    // Start sampler if enabled and collector exists
    if (opts.sampling_auto_start && rt_ptr->logger) {
        SystemStartEvent e;
        e.pid = gpufl::detail::GetPid();
        e.app = rt_ptr->app_name;
        e.name = "sampling_start";
        e.session_id = rt_ptr->session_id;
        e.ts_ns = gpufl::detail::GetTimestampNs();
        if (rt_ptr->collector) e.devices = rt_ptr->collector->sampleAll();
        if (rt_ptr->host_collector)
            e.host = rt_ptr->host_collector->sample();
        rt_ptr->logger->logSystemStart(e);
    }
    if (opts.sampling_auto_start && opts.system_sample_rate_ms > 0 &&
        rt_ptr->collector) {
        rt_ptr->sampler.start(rt_ptr->app_name, rt_ptr->session_id,
                              rt_ptr->logger, rt_ptr->collector,
                              opts.system_sample_rate_ms, rt_ptr->app_name);
    }

    // std::atexit(shutdown);

    GFL_LOG_DEBUG("Initialization complete!");
    return true;
}

void systemStart(std::string name) {
    Runtime* rt = runtime();
    if (!rt || !rt->logger) return;
    {
        SystemStartEvent e;
        e.pid = gpufl::detail::GetPid();
        e.app = rt->app_name;
        e.name = std::move(name);
        e.session_id = rt->session_id;
        e.ts_ns = gpufl::detail::GetTimestampNs();
        if (rt->collector) e.devices = rt->collector->sampleAll();
        if (rt->host_collector) e.host = rt->host_collector->sample();
        rt->logger->logSystemStart(e);
    }
    if (g_opts.system_sample_rate_ms > 0 && rt->collector) {
        rt->sampler.start(rt->app_name, rt->session_id, rt->logger,
                          rt->collector, g_opts.system_sample_rate_ms, name);
    }
}

void systemStop(std::string name) {
    Runtime* rt = runtime();
    if (!rt || !rt->logger) return;

    rt->sampler.stop();

    SystemStopEvent e;
    e.pid = gpufl::detail::GetPid();
    e.app = rt->app_name;
    e.session_id = rt->session_id;
    e.name = std::move(name);
    e.ts_ns = gpufl::detail::GetTimestampNs();
    if (rt->collector) e.devices = rt->collector->sampleAll();
    if (rt->host_collector) e.host = rt->host_collector->sample();
    rt->logger->logSystemStop(e);
}

void shutdown() {
    Monitor::Stop();
    Monitor::Shutdown();
    Runtime* rt = runtime();
    if (!rt) return;

    rt->sampler.stop();

    if (g_opts.sampling_auto_start && rt->collector) {
        SystemStopEvent e;
        e.pid = gpufl::detail::GetPid();
        e.app = rt->app_name;
        e.session_id = rt->session_id;
        e.name = "sampling_end";
        e.ts_ns = gpufl::detail::GetTimestampNs();
        if (rt->collector) e.devices = rt->collector->sampleAll();
        if (rt->host_collector) e.host = rt->host_collector->sample();
        rt->logger->logSystemStop(e);
    }

    ShutdownEvent se;
    se.pid = detail::GetPid();
    se.app = rt->app_name;
    se.session_id = rt->session_id;
    se.ts_ns = detail::GetTimestampNs();
    rt->logger->logShutdown(se);

    rt->logger->close();
    set_runtime(nullptr);

    GFL_LOG_DEBUG("Shutdown complete!");
}

// ---- ScopedMonitor ----

ScopedMonitor::ScopedMonitor(std::string name, std::string tag)
    : name_(std::move(name)),
      tag_(std::move(tag)),
      pid_(detail::GetPid()),
      start_ts_(detail::GetTimestampNs()),
      scope_id_(nextScopeId_()) {
    Runtime* rt = runtime();
    if (!rt || !rt->logger) return;
    ScopeBeginEvent e;
    e.pid = pid_;
    e.app = rt->app_name;
    e.session_id = rt->session_id;
    e.name = name_;
    e.tag = tag_;
    e.ts_ns = start_ts_;
    e.scope_id = scope_id_;

    auto& stack = getThreadScopeStack();

    e.scope_depth = stack.size();
    if (!stack.empty()) {
        std::string fullPath;
        for (size_t i = 0; i < stack.size(); ++i) {
            if (i > 0) fullPath += "|";
            fullPath += stack[i];
        }
        e.user_scope = fullPath + "|" + name_;
    } else {
        e.user_scope = name_;
    }
    stack.push_back(name_);

    if (rt->host_collector) {
        e.host = rt->host_collector->sample();
    }
    if (rt->collector) {
        e.devices = rt->collector->sampleAll();
    }
    rt->logger->logScopeBegin(e);

    // profiling
    Monitor::BeginProfilerScope(name_.c_str());
}

ScopedMonitor::~ScopedMonitor() {
    const Runtime* rt = runtime();
    if (!rt || !rt->logger) return;

    ScopeEndEvent e;
    e.pid = pid_;
    e.app = rt->app_name;
    e.session_id = rt->session_id;
    e.name = name_;
    e.tag = tag_;
    e.ts_ns = detail::GetTimestampNs();
    e.scope_id = scope_id_;

    auto& stack = getThreadScopeStack();

    if (!stack.empty()) {
        stack.pop_back();
    }
    e.scope_depth = stack.size();
    if (!stack.empty()) {
        std::string fullPath;
        for (size_t i = 0; i < stack.size(); ++i) {
            if (i > 0) fullPath += "|";
            fullPath += stack[i];
        }
        e.user_scope = fullPath + "|" + name_;
    } else {
        e.user_scope = name_;
    }

    if (rt->host_collector) {
        e.host = rt->host_collector->sample();
    }
    if (rt->collector) {
        e.devices = rt->collector->sampleAll();
    }

    rt->logger->logScopeEnd(e);

    Monitor::EndProfilerScope(name_.c_str());
}
}  // namespace gpufl

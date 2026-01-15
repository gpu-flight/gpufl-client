#include "gpufl.hpp"

#include <memory>
#include <string>
#include <utility>
#include <iostream>

#include "gpufl/core/events.hpp"
#include "gpufl/core/runtime.hpp"
#include "gpufl/core/logger.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/monitor.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/backends/host_collector.hpp"
#include "../backends/nvidia/cuda_collector.hpp"
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

    static std::shared_ptr<ISystemCollector<DeviceSample>> createCollector_(const BackendKind backend, std::string* reasonOut) {
        if (reasonOut) reasonOut->clear();

        auto setReason = [&](const std::string& r) {
            if (reasonOut && reasonOut->empty()) *reasonOut = r;
        };

        auto tryNvml = [&]() -> std::shared_ptr<ISystemCollector<DeviceSample>> {
#if GPUFL_ENABLE_NVIDIA && GPUFL_HAS_NVML
            return std::make_shared<gpufl::nvidia::NvmlCollector>();
#else
            setReason("NVIDIA telemetry not available (GPUFL_ENABLE_NVIDIA=OFF or NVML not found).");
            return nullptr;
#endif
        };

        auto tryRocm = [&]() -> std::shared_ptr<ISystemCollector<DeviceSample>> {
#if GPUFL_ENABLE_AMD && GPUFL_HAS_ROCM
            return std::make_shared<gpufl::amd::RocmCollector>();
#else
            setReason("AMD telemetry not available (GPUFL_ENABLE_AMD=OFF or ROCm not found).");
            return nullptr;
#endif
        };

        switch (backend) {
            case BackendKind::None:
                return nullptr;

            case BackendKind::Nvidia: {
                auto c = tryNvml();
                if (!c) setReason("Requested backend=nvidia but NVML is unavailable.");
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
                setReason("No GPU backend available (NVML/ROCm not compiled in or not available).");
                return nullptr;
            }
        }
    }

    bool init(const InitOptions& opts) {
        g_opts = opts;
        DebugLogger::setEnabled(opts.enableDebugOutput);
        GFL_LOG_DEBUG("Initializing...");
        if (runtime()) {
            GFL_LOG_DEBUG("Runtime already exists, shutting down first...");
            shutdown();
        }

        auto rt = std::make_unique<Runtime>();
        rt->appName = opts.appName.empty() ? "gpufl" : opts.appName;
        rt->sessionId = detail::generateSessionId();
        rt->logger = std::make_shared<Logger>();
        rt->hostCollector = std::make_unique<HostCollector>();
        rt->cudaCollector = std::make_unique<nvidia::CudaCollector>();

        const std::string logPath = opts.logPath.empty()
            ? defaultLogPath_(rt->appName)
            : opts.logPath;

        Logger::Options logOpts;
        logOpts.basePath = logPath;
        logOpts.systemSampleRateMs = opts.systemSampleRateMs;

        GFL_LOG_DEBUG("Opening log file: ", logPath);
        if (!rt->logger->open(logOpts)) {
            GFL_LOG_ERROR("Failed to open logger at: ", logPath);
            return false;
        }

        set_runtime(std::move(rt));
        rt = nullptr; // rt is now moved

        GFL_LOG_DEBUG("Initializing Monitor (CUPTI)...");
        MonitorOptions mOpts;
        mOpts.collectKernelDetails = opts.enableKernelDetails;
        mOpts.enableDebugOutput = opts.enableDebugOutput;
        mOpts.isProfiling = opts.enableProfiling;
        if (mOpts.isProfiling) {
            mOpts.collectKernelDetails = true; // if profiling is on, then it should be true.
        }
        mOpts.enableStackTrace = opts.enableStackTrace;
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
        ie.pid = detail::getPid();
        ie.sessionId = rt_ptr->sessionId;
        ie.app = rt_ptr->appName;
        ie.logPath = logPath;
        ie.tsNs = detail::getTimestampNs();
        // Collector may be unavailable on systems without NVML/ROCm. Guard usage.
        if (rt_ptr->collector) {
            ie.devices = rt_ptr->collector->sampleAll();
        }
        if (opts.backend == BackendKind::Auto || opts.backend == BackendKind::Nvidia) {
#if GPUFL_HAS_CUDA
            ie.cudaStaticDeviceInfos = rt_ptr->cudaCollector->sampleAll();
#endif
        }
        ie.host = rt_ptr->hostCollector->sample();

        rt_ptr->logger->logInit(ie);

        // Start sampler if enabled and collector exists
        if (opts.samplingAutoStart && rt_ptr->logger) {
            SystemStartEvent e;
            e.pid = gpufl::detail::getPid();
            e.app = rt_ptr->appName;
            e.name = "sampling_start";
            e.sessionId = rt_ptr->sessionId;
            e.tsNs = gpufl::detail::getTimestampNs();
            if (rt_ptr->collector) e.devices = rt_ptr->collector->sampleAll();
            if (rt_ptr->hostCollector) e.host = rt_ptr->hostCollector->sample();
            rt_ptr->logger->logSystemStart(e);

        }
        if (opts.samplingAutoStart && opts.systemSampleRateMs > 0 && rt_ptr->collector) {
            rt_ptr->sampler.start(rt_ptr->appName, rt_ptr->sessionId, rt_ptr->logger, rt_ptr->collector, opts.systemSampleRateMs, rt_ptr->appName);
        }

        //std::atexit(shutdown);

        GFL_LOG_DEBUG("Initialization complete!");
        return true;
    }

    void systemStart(std::string name) {
        Runtime* rt = runtime();
        if (!rt || !rt->logger) return;
        {
            SystemStartEvent e;
            e.pid = gpufl::detail::getPid();
            e.app = rt->appName;
            e.name = std::move(name);
            e.sessionId = rt->sessionId;
            e.tsNs = gpufl::detail::getTimestampNs();
            if (rt->collector) e.devices = rt->collector->sampleAll();
            if (rt->hostCollector) e.host = rt->hostCollector->sample();
            rt->logger->logSystemStart(e);
        }
        if (g_opts.systemSampleRateMs > 0 && rt->collector) {
            rt->sampler.start(rt->appName, rt->sessionId, rt->logger, rt->collector, g_opts.systemSampleRateMs, name);
        }
    }

    void systemStop(std::string name) {
        Runtime* rt = runtime();
        if (!rt || !rt->logger) return;

        rt->sampler.stop();

        SystemStopEvent e;
        e.pid = gpufl::detail::getPid();
        e.app = rt->appName;
        e.sessionId = rt->sessionId;
        e.name = std::move(name);
        e.tsNs = gpufl::detail::getTimestampNs();
        if (rt->collector) e.devices = rt->collector->sampleAll();
        if (rt->hostCollector) e.host = rt->hostCollector->sample();
        rt->logger->logSystemStop(e);
    }

    void shutdown() {
        Monitor::Stop();
        Monitor::Shutdown();
        Runtime* rt = runtime();
        if (!rt) return;

        rt->sampler.stop();

        if (g_opts.samplingAutoStart && rt->collector) {
            SystemStopEvent e;
            e.pid = gpufl::detail::getPid();
            e.app = rt->appName;
            e.sessionId = rt->sessionId;
            e.name = "sampling_end";
            e.tsNs = gpufl::detail::getTimestampNs();
            if (rt->collector) e.devices = rt->collector->sampleAll();
            if (rt->hostCollector) e.host = rt->hostCollector->sample();
            rt->logger->logSystemStop(e);
        }

        ShutdownEvent se;
        se.pid = detail::getPid();
        se.app = rt->appName;
        se.sessionId = rt->sessionId;
        se.tsNs = detail::getTimestampNs();
        rt->logger->logShutdown(se);

        rt->logger->close();
        set_runtime(nullptr);

        GFL_LOG_DEBUG("Shutdown complete!");
    }

    // ---- ScopedMonitor ----

    ScopedMonitor::ScopedMonitor(std::string name, std::string tag)
        : name_(std::move(name)),
          tag_(std::move(tag)),
          pid_(detail::getPid()),
          startTs_(detail::getTimestampNs()),
          scopeId_(nextScopeId_()) {

        Runtime* rt = runtime();
        if (!rt || !rt->logger) return;
        ScopeBeginEvent e;
        e.pid = pid_;
        e.app = rt->appName;
        e.sessionId = rt->sessionId;
        e.name = name_;
        e.tag = tag_;
        e.tsNs = startTs_;
        e.scopeId = scopeId_;

        auto& stack = getThreadScopeStack();

        e.scopeDepth = stack.size();
        if (!stack.empty()) {
            std::string fullPath;
            for (size_t i = 0; i < stack.size(); ++i) {
                if (i > 0) fullPath += "|";
                fullPath += stack[i];
            }
            e.userScope = fullPath + "|" + name_;
        } else {
            e.userScope = name_;
        }
        stack.push_back(name_);

        if (rt->hostCollector) {
            e.host = rt->hostCollector->sample();
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
        e.app = rt->appName;
        e.sessionId = rt->sessionId;
        e.name = name_;
        e.tag = tag_;
        e.tsNs = detail::getTimestampNs();
        e.scopeId = scopeId_;

        auto& stack = getThreadScopeStack();

        if (!stack.empty()) {
            stack.pop_back();
        }
        e.scopeDepth = stack.size();
        if (!stack.empty()) {
            std::string fullPath;
            for (size_t i = 0; i < stack.size(); ++i) {
                if (i > 0) fullPath += "|";
                fullPath += stack[i];
            }
            e.userScope = fullPath + "|" + name_;
        } else {
            e.userScope = name_;
        }

        if (rt->hostCollector) {
            e.host = rt->hostCollector->sample();
        }
        if (rt->collector) {
            e.devices = rt->collector->sampleAll();
        }

        rt->logger->logScopeEnd(e);

        Monitor::EndProfilerScope(name_.c_str());

    }
} // namespace gpufl

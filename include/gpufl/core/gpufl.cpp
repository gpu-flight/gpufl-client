#include "gpufl.hpp"

#include <memory>
#include <string>
#include <utility>

#include "gpufl/core/events.hpp"
#include "gpufl/core/runtime.hpp"
#include "gpufl/core/logger.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/backends/host_collector.hpp"

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
    struct InitOptions;

    static std::string defaultLogPath_(const std::string& app) {
        return app + ".log";
    }

    static std::atomic<uint64_t> g_nextScopeId{1};

    static uint64_t nextScopeId_() {
        return g_nextScopeId.fetch_add(1, std::memory_order_relaxed);
    }

    static std::shared_ptr<ISystemCollector> createCollector_(const BackendKind backend, std::string* reasonOut) {
        if (reasonOut) reasonOut->clear();

        auto setReason = [&](const std::string& r) {
            if (reasonOut && reasonOut->empty()) *reasonOut = r;
        };

        auto tryNvml = [&]() -> std::shared_ptr<ISystemCollector> {
#if GPUFL_ENABLE_NVIDIA && GPUFL_HAS_NVML
            return std::make_shared<gpufl::nvidia::NvmlCollector>();
#else
            setReason("NVIDIA telemetry not available (GPUFL_ENABLE_NVIDIA=OFF or NVML not found).");
            return nullptr;
#endif
        };

        auto tryRocm = [&]() -> std::shared_ptr<ISystemCollector> {
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
        if (runtime()) {
            shutdown();
        }

        auto rt = std::make_unique<Runtime>();
        rt->appName = opts.appName.empty() ? "gpufl" : opts.appName;

        rt->logger = std::make_shared<Logger>();
        rt->hostCollector = std::make_unique<HostCollector>();

        const std::string logPath = opts.logPath.empty()
            ? defaultLogPath_(rt->appName)
            : opts.logPath;

        Logger::Options logOpts;
        logOpts.basePath = logPath;
        logOpts.systemSampleRateMs = opts.systemSampleRateMs;

        g_systemSampleRateMs.store(opts.systemSampleRateMs, std::memory_order_relaxed);

        if (!rt->logger->open(logOpts)) {
            return false;
        }

        // Runtime backend selection
        std::string backendReason;
        rt->collector = createCollector_(opts.backend, &backendReason);

        // init event with inventory (optional)
        InitEvent ie;
        ie.pid = detail::getPid();
        ie.app = rt->appName;
        ie.logPath = logPath;
        ie.tsNs = detail::getTimestampNs();
        ie.devices = rt->collector->sampleAll();
        ie.host = rt->hostCollector->sample();

        rt->logger->logInit(ie);

        // Start sampler if enabled and collector exists
        if (opts.systemSampleRateMs > 0 && rt->collector) {
            rt->sampler.start(rt->appName, rt->logger, rt->collector, opts.systemSampleRateMs, rt->appName);
        }

        set_runtime(std::move(rt));
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
            e.tsNs = gpufl::detail::getTimestampNs();
            if (rt->collector) e.devices = rt->collector->sampleAll();
            if (rt->hostCollector) e.host = rt->hostCollector->sample();
            rt->logger->logSystemStart(e);
        }
        if (const int intervalMs = g_systemSampleRateMs.load(std::memory_order_relaxed); intervalMs > 0 && rt->collector) {
            rt->sampler.start(rt->appName, rt->logger, rt->collector, intervalMs, name);
        }
    }

    void systemStop(std::string name) {
        Runtime* rt = runtime();
        if (!rt || !rt->logger) return;

        rt->sampler.stop();

        SystemStopEvent e;
        e.pid = gpufl::detail::getPid();
        e.app = rt->appName;
        e.name = std::move(name);
        e.tsNs = gpufl::detail::getTimestampNs();
        if (rt->collector) e.devices = rt->collector->sampleAll();
        if (rt->hostCollector) e.host = rt->hostCollector->sample();
        rt->logger->logSystemStop(e);
    }

    void shutdown() {
        Runtime* rt = runtime();
        if (!rt) return;

        rt->sampler.stop();

        ShutdownEvent se;
        se.pid = detail::getPid();
        se.app = rt->appName;
        se.tsNs = detail::getTimestampNs();
        rt->logger->logShutdown(se);

        rt->logger->close();
        set_runtime(nullptr);
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
        e.name = name_;
        e.tag = tag_;
        e.tsNs = startTs_;
        e.scopeId = scopeId_;

        if (rt->collector) {
            e.devices = rt->collector->sampleAll();
        }
        if (rt->hostCollector) e.host = rt->hostCollector->sample();
        rt->logger->logScopeBegin(e);
    }

    ScopedMonitor::~ScopedMonitor() {
        const Runtime* rt = runtime();
        if (!rt || !rt->logger) return;

        ScopeEndEvent e;
        e.pid = pid_;
        e.app = rt->appName;
        e.name = name_;
        e.tag = tag_;
        e.tsNs = detail::getTimestampNs();
        e.scopeId = scopeId_;

        if (rt->collector) {
            e.devices = rt->collector->sampleAll();
        }
        if (rt->hostCollector) e.host = rt->hostCollector->sample();

        rt->logger->logScopeEnd(e);
    }
} // namespace gpufl

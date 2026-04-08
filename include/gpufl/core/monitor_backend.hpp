#pragma once

#include <optional>

#include "gpufl/core/events.hpp"
#include "gpufl/core/monitor.hpp"

namespace gpufl {

/**
 * @brief Interface for backend-specific monitoring implementations.
 *
 * Backends implement this interface to provide platform-specific
 * kernel and event monitoring (e.g., CUPTI for NVIDIA, ROCTracer for AMD).
 */
class IMonitorBackend {
   public:
    virtual ~IMonitorBackend() = default;

    /**
     * @brief initialize the monitoring backend with given options.
     * @param opts Configuration options for monitoring
     */
    virtual void initialize(const MonitorOptions& opts) = 0;

    /**
     * @brief shutdown the monitoring backend and release resources.
     */
    virtual void shutdown() = 0;

    /**
     * @brief start active monitoring/tracing.
     */
    virtual void start() = 0;

    /**
     * @brief stop active monitoring/tracing.
     */
    virtual void stop() = 0;

    virtual bool IsMonitoringMode() = 0;

    virtual bool IsProfilingMode() = 0;

    virtual void OnScopeStart(const char* name) {}
    virtual void OnScopeStop(const char* name) {}

    /** @brief Periodically drain buffered profiling data. Thread-safe. */
    virtual void DrainProfilingData() {}

    virtual void OnPerfScopeStart(const char* name) {}
    virtual void OnPerfScopeStop(const char* name) {}
    virtual std::optional<PerfMetricEvent> TakeLastPerfEvent() { return std::nullopt; }
};

}  // namespace gpufl

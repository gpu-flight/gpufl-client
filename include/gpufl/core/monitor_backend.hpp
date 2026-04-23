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

    /**
     * @brief True if the selected profiling engine attempted to start but
     *        was blocked by CUPTI_ERROR_INSUFFICIENT_PRIVILEGES (or the
     *        virtualized equivalent). Checked by gpufl::init() after
     *        start() to surface a clear user-facing error rather than
     *        letting kernel launches crash with a half-initialized CUPTI.
     *
     *        Default: false (backends that don't track this return false
     *        and fall back to the previous best-effort behavior).
     */
    virtual bool HasInsufficientPrivileges() const { return false; }

    /**
     * @brief True when the profiling engine (if any) is producing data.
     *        None / monitoring-only backends return true (they don't do
     *        profiling, so they aren't blocked). Used by the frontend
     *        via session metadata to explain why PC samples are missing.
     */
    virtual bool IsProfilingOperational() const { return true; }

    virtual void OnScopeStart(const char* name) {}
    virtual void OnScopeStop(const char* name) {}

    /** @brief Periodically drain buffered profiling data. Thread-safe. */
    virtual void DrainProfilingData() {}

    virtual void OnPerfScopeStart(const char* name) {}
    virtual void OnPerfScopeStop(const char* name) {}
    virtual std::optional<PerfMetricEvent> TakeLastPerfEvent() { return std::nullopt; }
};

}  // namespace gpufl

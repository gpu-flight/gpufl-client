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
     * @brief Emit the capture-capabilities report (what was actually
     *        collected) to the logger, without releasing any backend
     *        resources. Used by the Windows-injection process-exit path to
     *        write capabilities BEFORE the fragile CUPTI release. Idempotent;
     *        a no-op for backends that report no capabilities.
     */
    virtual void emitCapabilities() {}

    /**
     * @brief Decode and write end-of-session profiling data that only becomes
     *        available once the engine is stopped (e.g. Range Profiler
     *        kernel-replay per-kernel metrics — achieved occupancy, SM
     *        throughput, cache hit rates) to the logger, WITHOUT the fragile
     *        CUPTI thread-join/release teardown. Like emitCapabilities(), the
     *        Windows-injection process-exit path calls this BEFORE the logger
     *        closes — otherwise the engine stop (and thus this data) is
     *        deferred to ReleaseBackendForExit, which runs after the log is
     *        closed, silently dropping every decoded metric event. Idempotent;
     *        a no-op for backends/engines that produce no such deferred data.
     */
    virtual void emitPendingPerfEvents() {}

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

    /**
     * @brief Scope hooks for a bounded deep window, as opposed to an
     * ordinary user scope.
     *
     * They arm and disarm the same engines, but a backend running in
     * DeepArmMode::WindowOnly has to tell the two apart: under that mode
     * ordinary scopes must NOT arm anything, or a per-step GFL_SCOPE in a
     * training loop would leave the engines armed for the whole run and
     * the window would mean nothing. Default: identical to a user scope,
     * which is right for every backend that arms unconditionally.
     */
    virtual void OnDeepWindowStart(const char* name) { OnScopeStart(name); }
    virtual void OnDeepWindowStop(const char* name) { OnScopeStop(name); }

    /** @brief Periodically drain buffered profiling data. Thread-safe. */
    virtual void DrainProfilingData() {}

    virtual void OnPerfScopeStart(const char* name) {}
    virtual void OnPerfScopeStop(const char* name) {}
    // Perf-scope counterparts of OnDeepWindowStart/Stop; see those.
    virtual void OnDeepWindowPerfStart(const char* name) { OnPerfScopeStart(name); }
    virtual void OnDeepWindowPerfStop(const char* name) { OnPerfScopeStop(name); }
    virtual std::optional<PerfMetricEvent> TakeLastPerfEvent() { return std::nullopt; }
};

}  // namespace gpufl

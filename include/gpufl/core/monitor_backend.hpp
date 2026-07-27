#pragma once

#include <optional>
#include <string>
#include <vector>

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

    /**
     * @brief Disarm, and report the wire names of the engines that WERE armed.
     *
     * An empty list means this window armed nothing, which is a real outcome
     * and not an error signal. Trace is never listed: it collects for the
     * whole session rather than arming with the window, so naming it here
     * would credit the window with data it did not gate.
     *
     * The names are read here rather than through a separate query for two
     * reasons. They have to be sampled BEFORE the disarm - afterwards nothing
     * is armed and the answer is always empty - and folding it into the call
     * that does the disarming makes that ordering impossible to get wrong.
     *
     * Deliberately a point-in-time reading, NOT the session's verdict.
     * capture_capabilities.selected_engine answers "what did this session end
     * up being" and can only be computed at session end, since it depends on
     * what each engine finally produced. A window needs the other question:
     * "what was armed while THIS window was open". The two legitimately
     * differ - a Deep run whose SASS declines its first arm falls back to PC
     * sampling, and a window that closed before an engine got blocked saw a
     * different world than the session summary reports. Keeping both is what
     * makes the window row an audit record rather than a duplicate.
     *
     * This is also what explains a window's launch coverage: a set containing
     * SASS or a replaying Range profiler covers ~25x fewer launches per second
     * than one holding only PC or PM sampling.
     */
    virtual std::vector<std::string> OnDeepWindowStop(const char* name) {
        OnScopeStop(name);
        return {};
    }

    /** @brief Periodically drain buffered profiling data. Thread-safe. */
    virtual void DrainProfilingData() {}

    /**
     * @brief Close a deep window whose bound has been reached.
     *
     * Called from the collector on every iteration, not on the slower flush
     * beat, because it decides how closely a window tracks its deadline.
     * Must stay cheap when there is nothing to close.
     *
     * It lives here rather than on the launch callback because the engines'
     * teardown calls fail with CUPTI_ERROR_UNKNOWN when made from inside a
     * CUPTI callback; the collector is off that path and can make the CUDA
     * context current itself.
     */
    virtual void ServiceDeepWindow() {}

    /**
     * @brief Is any engine that a window could arm actually prepared?
     *
     * The capability gate for conditional windows. Checking the configured
     * engine enum is necessary but not sufficient: a Trace-only run resolves to
     * a valid engine and still arms nothing inside a window, so a rule would
     * spend its whole budget opening windows that collect no deep data.
     *
     * Answers about what was PREPARED, not what is currently armed - under
     * WindowOnly nothing is armed until a window opens, which is precisely when
     * the answer is needed.
     */
    virtual bool DeepEnginesPrepared() const { return false; }

    virtual void OnPerfScopeStart(const char* name) {}
    virtual void OnPerfScopeStop(const char* name) {}
    // Perf-scope counterparts of OnDeepWindowStart/Stop; see those.
    virtual void OnDeepWindowPerfStart(const char* name) { OnPerfScopeStart(name); }
    virtual void OnDeepWindowPerfStop(const char* name) { OnPerfScopeStop(name); }
    virtual std::optional<PerfMetricEvent> TakeLastPerfEvent() { return std::nullopt; }
};

}  // namespace gpufl

#pragma once

#include <cstdint>
#include <string>

namespace gpufl {

/**
 * @brief Why a deep window closed.
 *
 * Reported on every deep_window event. A window that exhausted its launch
 * budget covered exactly what the caller asked for; one that hit the
 * deadline may have covered far less, because kernel replay stretches wall
 * time without advancing the application. Without this field a short
 * window looks like a bug rather than the bound doing its job.
 */
enum class DeepWindowClose { Deadline, LaunchBudget, Manual, SessionStop };

/** Wire name for a close reason (goes into the deep_window event). */
const char* DeepWindowCloseName(DeepWindowClose reason);

/**
 * @brief Bounds for one deep window.
 *
 * Both bounds are optional and combine with OR - whichever is reached
 * first closes the window. Wall time alone is a poor bound for the replay
 * engines (SASS / Range), where a three-second window can cover a handful
 * of launches; a launch budget says what the caller actually meant.
 */
struct DeepWindowSpec {
    int64_t     max_duration_ms = 0;  // 0 = no time bound
    uint64_t    max_launches    = 0;  // 0 = no launch bound
    // Minimum quiet time after a window closes before another may open.
    // 0 = none. Without it a condition that stays true reopens a window the
    // instant the last one expired, and the run pays deep cost forever. Only
    // the library knows when the last window closed, so this bound belongs
    // here rather than in the caller's trigger.
    int64_t     cooldown_ms     = 0;
    std::string name = "deep_window";
};

/**
 * @brief The profiler scope that deep engines arm on, opened by a trigger
 * and closed by a bound instead of by a destructor.
 *
 * The deep engines are already scope-gated (PM sampling arms in
 * onScopeStart, SASS arms on scope start and flushes on scope stop, the
 * Range profiler's scope mode initializes Perfworks lazily), so this adds
 * no new arming mechanism. What it adds is a scope that nobody has to
 * close by hand.
 *
 * Process-wide singleton: a window is a property of the session, not of a
 * thread, and the CUPTI calls behind it are context-bound.
 */
class DeepWindow {
   public:
    /**
     * @brief Open a window if none is active.
     *
     * A second call while a window is open is IGNORED, not an extension,
     * so a trigger that fires every training step cannot hold the window
     * open indefinitely. Returns true only if this call opened it.
     *
     * Arms the engines on the calling thread. Call it from the application
     * thread that runs the workload - that thread is context-current.
     */
    static bool Open(const DeepWindowSpec& spec);

    /**
     * @brief Ask for a window from a thread that must not arm one itself.
     *
     * Arming runs CUPTI calls that are only safe on the application thread
     * at a launch boundary, so a trigger living anywhere else - the sampler
     * evaluating a rule, a listener woken by an external signal, a timer -
     * records the request here and the next launch performs the open. The
     * mirror of how a deadline reached off the app thread defers its close.
     *
     * A pending request is replaced, not queued: the newest spec wins.
     */
    static void RequestOpen(const DeepWindowSpec& spec);

    /**
     * @brief Request an open and get back a token identifying it.
     *
     * For callers that must know whether the window they asked for actually
     * happened - a rule with a window budget cannot count a window that never
     * opened, and cannot let a manual one consume its budget either.
     *
     * Returns 0 when the request is refused outright (a window is already open,
     * or the cooldown has not elapsed). A non-zero token matches
     * LastOpenedToken() once, and only once, the requested window opens.
     */
    static uint64_t RequestOpenTagged(const DeepWindowSpec& spec);

    /** @brief Token of the most recent open; 0 when it was not from a request. */
    static uint64_t LastOpenedToken();

    /**
     * @brief Token of the request still queued, or 0 when none is.
     *
     * An open is serviced on a later beat, so "has not opened yet" and "will
     * never open" look identical without this. A caller that treated the first
     * as the second would abandon a window that was about to open.
     */
    static uint64_t PendingOpenToken();

    /** @brief Monotonic count of windows that actually opened. */
    static uint64_t OpensCompleted();

    /**
     * @brief Same, but not before `delay_ms` have passed.
     *
     * Backs the launcher's time-based trigger, which is the only trigger
     * available when the target's source can't be edited. Costs no thread:
     * the due-time is checked on the launch beat that is already running.
     */
    static void ScheduleOpenAfter(int64_t delay_ms, const DeepWindowSpec& spec);

    /** @brief Close an active window. No-op when none is open. */
    static void Close(DeepWindowClose reason);

    static bool Active();

    /**
     * @brief Per-launch bound accounting, driven from the CUPTI launch
     * callback.
     *
     * Consumes one launch of budget and RECORDS that a bound was reached.
     * It deliberately does not close: this runs inside a CUPTI callback,
     * and the engines' teardown calls (cuptiPmSamplingDecodeData,
     * cuptiPCSamplingStop) return CUPTI_ERROR_UNKNOWN when invoked from
     * there. Verified on Linux/driver 610.43; Windows happened to tolerate
     * it, which is why the first version looked correct.
     */
    static void OnLaunch();

    /**
     * @brief Cheap, lock-free: is there an arm or a disarm waiting?
     *
     * Lets the collector poll every iteration and pay for making a CUDA
     * context current only when there is actually something to do.
     */
    static bool HasPendingWork();

    /**
     * @brief Perform a pending arm or disarm.
     *
     * Both halves run here, off the CUPTI callback path and with the CUDA
     * context current, which CuptiBackend::ServiceDeepWindow arranges. Arm
     * and disarm are kept on the SAME thread deliberately: PM sampling's
     * decode rejects a session whose start and stop straddle threads.
     */
    static void ServicePending();

    /** @brief Test seam: drop all state without touching a backend. */
    static void ResetForTesting();

   private:
    // Claims a pending request and opens it. Called from OnLaunch only.
    static void TakePendingOpen_();
    // Records that a bound was reached without acting on it.
    static void RequestClose_(DeepWindowClose reason);
};

namespace detail {

// Perf scopes (Range Profiler / Perfworks) only mean something for engines
// that use them. Shared by ScopedMonitor and DeepWindow so the two paths
// can't drift apart on which engines get one.
// `is_deep_window` routes to the backend's deep-window hooks, which stay
// live under DeepArmMode::WindowOnly while ordinary user scopes go quiet.
bool PerfScopeEnabled();
void BeginPerfScopeIfEnabled(const char* name, bool is_deep_window);
// Ends the perf scope and writes the decoded PerfMetricEvent, if the engine
// produced one. Mirrors the tail of ScopedMonitor's destructor.
void EndPerfScopeIfEnabled(const char* name, int pid, int64_t start_ns,
                           int64_t end_ns, bool is_deep_window);

}  // namespace detail
}  // namespace gpufl

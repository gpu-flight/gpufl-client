#include "gpufl/core/deep_window.hpp"

#include <atomic>
#include <cstdlib>
#include <mutex>

#include "gpufl.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/env_vars.hpp"
#include "gpufl/core/events.hpp"
#include "gpufl/core/logger/logger.hpp"
#include "gpufl/core/model/deep_window_model.hpp"
#include "gpufl/core/model/perf_metric_model.hpp"
#include "gpufl/core/monitor.hpp"
#include "gpufl/core/monitor_backend.hpp"
#include "gpufl/core/runtime.hpp"
#include "gpufl/core/teardown_flag.hpp"  // detail::isProcessExitTeardown

namespace gpufl {
namespace {

// Serializes open/close transitions. Held across the engine arm/disarm so a
// window can't be closed between "state says open" and "engines are armed".
std::mutex g_mu;

// Read lock-free from the launch callback on every launch while a window is
// open, so the hot path never touches g_mu. Also the re-entrancy guard:
// Close() clears it before the engine teardown, and that teardown can
// synchronize the device and re-enter the launch callback.
std::atomic g_active{false};

std::atomic<int64_t>  g_deadline_ns{0};        // 0 = no time bound
std::atomic<uint64_t> g_launches_remaining{0};  // 0 = no launch bound
std::atomic<uint64_t> g_launches_covered{0};
// Set by the launch callback when a bound is reached; consumed by the
// collector, which is the thread allowed to run the engines' teardown.
std::atomic     g_close_requested{false};
std::atomic<int> g_close_reason{static_cast<int>(DeepWindowClose::Deadline)};

// An open asked for by a thread that can't arm one itself. Checked lock-free
// on the launch beat; g_pending is only read once this is set, so the
// hot path pays for it exactly when a trigger is waiting.
std::atomic       g_open_requested{false};
std::atomic<int64_t> g_pending_open_at_ns{0};  // 0 = at the next launch

// The queued request, as ONE record. Spec and owner token were separate fields
// once, and an untagged request arriving after a tagged one then replaced only
// the spec - so a manual window opened carrying a rule's token and was charged
// to that rule's budget. Replacing the record replaces both or neither.
struct PendingOpen {
    DeepWindowSpec spec;
    uint64_t owner_token = 0;   // 0 = nobody is waiting on this one
};

// First request wins; a second is refused while one is still queued.
//
// The alternative - newest wins - silently discards whichever trigger asked
// first, and with both --deep-after and a rule configured that is decided by
// which one happens to run first rather than by anything the user chose. A
// refusal is at least reported: a rule gets token 0 and goes back to armed,
// and the scheduled window logs that it kept its place.
bool PendingIsQueued() { return g_open_requested.load(std::memory_order_acquire); }
PendingOpen      g_pending;    // guarded by g_mu

// When the last window closed, so a cooldown can be enforced. 0 = never.
std::atomic<int64_t> g_last_close_ns{0};

// Attribution for windows opened through a tagged request.
//
// A plain "a window opened" counter is not enough: a manual or scheduled
// window opening while a rule's request is outstanding would be counted
// against that rule's budget, and the budget is meant to bound what the RULE
// costs. Tokens start at 1 so 0 keeps its meaning of "not from a request".
std::atomic<uint64_t> g_next_open_token{1};
std::atomic<uint64_t> g_last_opened_token{0};
std::atomic<uint64_t> g_opens_completed{0};
// Token of the request currently being serviced. Only TakePendingOpen_ sets
// it, and only for the duration of the Open() call it drives, so a direct
// Open() can never inherit someone else's attribution.
thread_local uint64_t g_claimed_token = 0;

DeepWindowTrigger g_trigger;   // guarded by g_mu
int64_t     g_opened_ns = 0;
int64_t     g_requested_duration_ms = 0;
uint64_t    g_requested_max_launches = 0;
std::string g_name;
// Pairs the window's scope begin/end rows. 0 = no scope row was pushed, which
// is what a run with no Monitor backend looks like.
uint64_t    g_scope_instance_id = 0;

bool ComboActive() {
    const char* combo = std::getenv(env::kEngineCombo);
    return combo && combo[0] != '\0';
}

// Non-negative integer from env, or `fallback` when unset or malformed.
int64_t EnvUnsignedOr(const char* name, const int64_t fallback) {
    const char* v = std::getenv(name);
    if (!v || v[0] == '\0') return fallback;
    char* end = nullptr;
    const long long n = std::strtoll(v, &end, 10);
    if (end == v || *end != '\0' || n < 0) {
        GFL_LOG_ERROR(name, "='", v,
                      "' is not a non-negative integer. Ignoring.");
        return fallback;
    }
    return n;
}

// Fills bounds left at 0 from the environment, so an operator can size a
// window the application code already asks for - and so the injected path,
// which can't set a DeepWindowSpec, can set one at all.
void ApplyEnvDefaults(DeepWindowSpec& spec) {
    if (spec.max_duration_ms == 0) {
        spec.max_duration_ms = EnvUnsignedOr(env::kDeepWindowMs, 0);
    }
    if (spec.max_launches == 0) {
        spec.max_launches =
            static_cast<uint64_t>(EnvUnsignedOr(env::kDeepWindowMaxLaunches, 0));
    }
    if (spec.cooldown_ms == 0) {
        spec.cooldown_ms = EnvUnsignedOr(env::kDeepWindowCooldownMs, 0);
    }
}

// True while a just-closed window's cooldown is still running.
bool InCooldown(const DeepWindowSpec& spec) {
    if (spec.cooldown_ms <= 0) return false;
    const int64_t last = g_last_close_ns.load(std::memory_order_relaxed);
    if (last == 0) return false;
    return detail::GetTimestampNs() - last < spec.cooldown_ms * 1000000;
}

}  // namespace

const char* DeepWindowCloseName(const DeepWindowClose reason) {
    switch (reason) {
        case DeepWindowClose::Deadline:     return "deadline";
        case DeepWindowClose::LaunchBudget: return "launch_budget";
        case DeepWindowClose::Manual:       return "manual";
        case DeepWindowClose::SessionStop:  return "session_stop";
    }
    return "unknown";
}

namespace detail {

bool PerfScopeEnabled() {
    // Also fire for an engine combo with a Trace base - otherwise a
    // Trace+RangeProfiler combo would never trigger Range's perf scope.
    return g_opts.profiling_engine != ProfilingEngine::Monitor &&
           (g_opts.profiling_engine != ProfilingEngine::Trace || ComboActive());
}

void BeginPerfScopeIfEnabled(const char* name, const bool is_deep_window) {
    if (!PerfScopeEnabled()) return;
    if (is_deep_window) {
        Monitor::BeginDeepWindowPerfScope(name);
    } else {
        Monitor::BeginPerfScope(name);
    }
}

void EndPerfScopeIfEnabled(const char* name, const int pid,
                           const int64_t start_ns, const int64_t end_ns,
                           const bool is_deep_window) {
    if (!PerfScopeEnabled()) return;
    // Triggers EndPerfPassAndDecode first.
    if (is_deep_window) {
        Monitor::EndDeepWindowPerfScope(name);
    } else {
        Monitor::EndPerfScope(name);
    }

    const Runtime* rt = runtime();
    if (!rt || !rt->logger) return;
    IMonitorBackend* backend = Monitor::GetBackend();
    if (!backend) return;
    auto event_opt = backend->TakeLastPerfEvent();
    if (!event_opt) return;

    PerfMetricEvent& pe = *event_opt;
    pe.pid        = pid;
    pe.app        = rt->app_name;
    pe.session_id = rt->session_id;
    pe.name       = name ? name : "";
    pe.start_ns   = start_ns;
    pe.end_ns     = end_ns;
    rt->logger->write(model::PerfMetricModel(pe));
}

}  // namespace detail

bool DeepWindow::Active() {
    return g_active.load(std::memory_order_acquire);
}

bool DeepWindow::Open(const DeepWindowSpec& spec) {
    if (const Runtime* rt = runtime(); !rt || !rt->logger) return false;

    std::string name;
    {
        std::lock_guard lk(g_mu);
        if (g_active.load(std::memory_order_relaxed)) {
            // Not an extension. A trigger that fires every step would
            // otherwise hold the window open for the rest of the run.
            return false;
        }
        if (InCooldown(spec)) {
            // A condition that stays true would otherwise reopen a window the
            // moment the last one expired, and the run never stops paying.
            GFL_LOG_DEBUG("[DeepWindow] open suppressed: cooldown ",
                          spec.cooldown_ms, "ms not elapsed");
            return false;
        }

        // Publish who this open belongs to, then clear the claim so a later
        // direct Open() cannot inherit it. The claim is taken only by
        // RequestOpenTagged and consumed exactly once, here.
        g_last_opened_token.store(g_claimed_token, std::memory_order_release);
        g_opens_completed.fetch_add(1, std::memory_order_acq_rel);

        g_opened_ns = detail::GetTimestampNs();
        g_trigger = spec.trigger;
        g_name = spec.name.empty() ? "deep_window" : spec.name;
        g_requested_duration_ms = spec.max_duration_ms;
        g_requested_max_launches = spec.max_launches;
        g_deadline_ns.store(spec.max_duration_ms > 0
                                ? g_opened_ns + spec.max_duration_ms * 1000000
                                : 0,
                            std::memory_order_relaxed);
        g_launches_remaining.store(spec.max_launches, std::memory_order_relaxed);
        g_launches_covered.store(0, std::memory_order_relaxed);
        g_close_requested.store(false, std::memory_order_relaxed);
        name = g_name;

        // Publish last: once this is true the launch callback starts
        // consuming budget, and everything it reads is already set.
        g_active.store(true, std::memory_order_release);

        // Open the window as a real scope, not just an arming signal. This is
        // what puts it on the timeline as its own range AND makes the samples
        // collected inside it carry the window's name instead of the enclosing
        // process scope - the sample writers stamp whatever scope is active.
        // Pushed before the engines arm so nothing collected can land under
        // the parent name.
        g_scope_instance_id = Monitor::AllocateScopeInstanceId();
        ScopeBatchRow open_row;
        open_row.ts_ns = g_opened_ns;
        open_row.scope_instance_id = g_scope_instance_id;
        open_row.name_id = Monitor::InternScopeName(g_name);
        open_row.event_type = 0;
        open_row.depth = Monitor::OpenScopeDepth();
        Monitor::PushScopeRow(open_row);

        // Arms the deep engines. Runs under the lock so a concurrent close
        // can't disarm engines this call hasn't armed yet; safe because the
        // arm path doesn't re-enter DeepWindow.
        Monitor::BeginDeepWindowScope(name.c_str());
        detail::BeginPerfScopeIfEnabled(name.c_str(), /*is_deep_window=*/true);
    }

    GFL_LOG_DEBUG("[DeepWindow] opened name=", name,
                  " duration_ms=", spec.max_duration_ms,
                  " max_launches=", spec.max_launches);
    return true;
}

void DeepWindow::Close(const DeepWindowClose reason) {
    if (!g_active.load(std::memory_order_acquire)) return;

    DeepWindowEvent ev;
    std::string name;
    {
        int64_t start_ns = 0;
        std::lock_guard lk(g_mu);
        if (!g_active.load(std::memory_order_relaxed)) return;
        // Clear before the engine teardown below. That teardown can
        // synchronize the device and re-enter the launch callback, and
        // OnLaunch's lock-free check turns the re-entry into a no-op
        // instead of a deadlock on g_mu.
        g_active.store(false, std::memory_order_release);
        g_close_requested.store(false, std::memory_order_relaxed);

        name = g_name;
        start_ns = g_opened_ns;
        const int64_t end_ns = detail::GetTimestampNs();
        // Starts the cooldown clock. Set from the decision point, not after
        // the engine teardown, so a slow disarm doesn't shorten the quiet time.
        g_last_close_ns.store(end_ns, std::memory_order_relaxed);

        ev.trigger                = g_trigger;
        ev.pid                    = detail::GetPid();
        ev.name                   = g_name;
        ev.close_reason           = DeepWindowCloseName(reason);
        ev.start_ns               = start_ns;
        ev.end_ns                 = end_ns;
        ev.duration_ns            = end_ns - start_ns;
        ev.launches_covered       = g_launches_covered.load(std::memory_order_relaxed);
        ev.requested_duration_ms  = g_requested_duration_ms;
        ev.requested_max_launches = g_requested_max_launches;

        // Disarms the deep engines and drains whatever they collected.
        // Skipped on process-exit teardown, where cudart has already
        // destroyed the context and the scope-stop path would fault against
        // it - the engines' own exit handling flushes there instead. The
        // event below is still written so the window is on the record.
        if (!detail::isProcessExitTeardown()) {
            // The disarm hands back what WAS armed - an audit record of this
            // window, distinct from the session-end verdict in
            // capture_capabilities, which the two can legitimately disagree
            // with when an engine fell back or got blocked mid-run.
            ev.engines = Monitor::EndDeepWindowScope(name.c_str());
            detail::EndPerfScopeIfEnabled(name.c_str(), ev.pid, start_ns, end_ns,
                                          /*is_deep_window=*/true);
        } else {
            // Teardown skipped the disarm, so nothing observed the armed set.
            // Name the resolved request instead - less trustworthy than a real
            // reading, but close_reason marks the row as a teardown close.
            // Not folded into an empty-list check: an empty list from a real
            // disarm means this window armed nothing, and overwriting that
            // would erase the one record of it.
            ev.engines = {
                ProfilingEngineWireName(Monitor::ResolvedProfilingEngine())};
        }

        // Close the scope last. The disarm above drains what the engines
        // collected, and those samples belong to this window - closing first
        // would hand the name back to the process scope and mislabel them.
        // Pushed even on the teardown path, where skipping it would leave the
        // scope open and every later sample carrying the window's name.
        if (g_scope_instance_id != 0) {
            ScopeBatchRow close_row;
            close_row.ts_ns = end_ns;
            close_row.scope_instance_id = g_scope_instance_id;
            close_row.name_id = Monitor::InternScopeName(name);
            close_row.event_type = 1;
            close_row.depth = 0;   // ignored on close; the open row carries it
            // end_ns intentionally precedes engine disarm so the window event
            // measures the requested boundary. Publish it only after the final
            // drain above, immediately before the scope-state transition.
            Monitor::MarkScopeClosePending(g_scope_instance_id, end_ns);
            Monitor::PushScopeRow(close_row);
            g_scope_instance_id = 0;
        }
    }

    if (const Runtime* rt = runtime(); rt && rt->logger) {
        ev.app = rt->app_name;
        ev.session_id = rt->session_id;
        rt->logger->write(model::DeepWindowModel(ev));
    }

    GFL_LOG_DEBUG("[DeepWindow] closed name=", name,
                  " reason=", ev.close_reason,
                  " duration_ns=", ev.duration_ns,
                  " launches_covered=", ev.launches_covered);
}

void DeepWindow::RequestOpen(const DeepWindowSpec& spec) {
    ScheduleOpenAfter(0, spec);
}

uint64_t DeepWindow::RequestOpenTagged(const DeepWindowSpec& spec) {
    uint64_t token = 0;
    {
        std::lock_guard lk(g_mu);
        // Decided here so the caller learns now, rather than discovering later
        // that a window it counted on never opened.
        if (g_active.load(std::memory_order_relaxed)) return 0;
        if (InCooldown(spec)) return 0;
        if (PendingIsQueued()) {
            // Someone else - typically the launcher's --deep-after window - is
            // already waiting. Refusing here means the rule retries later
            // instead of cancelling a window the user explicitly scheduled.
            GFL_LOG_DEBUG("[DeepWindow] tagged open refused: a request is "
                          "already queued");
            return 0;
        }

        token = g_next_open_token.fetch_add(1, std::memory_order_relaxed);
        // Spec and token published together, under one lock. Splitting them is
        // how an untagged request could inherit a rule's attribution.
        g_pending.spec = spec;
        g_pending.owner_token = token;
        g_pending_open_at_ns.store(0, std::memory_order_relaxed);
        // Published INSIDE the lock. Releasing first left a window where a
        // second caller took the lock, saw no request queued, and overwrote
        // this one - which is precisely the first-wins rule this is meant to
        // enforce, defeated by two threads asking at once.
        g_open_requested.store(true, std::memory_order_release);
    }
    GFL_LOG_DEBUG("[DeepWindow] tagged open requested token=", token,
                  " duration_ms=", spec.max_duration_ms);
    return token;
}

uint64_t DeepWindow::LastOpenedToken() {
    return g_last_opened_token.load(std::memory_order_acquire);
}

uint64_t DeepWindow::PendingOpenToken() {
    if (!g_open_requested.load(std::memory_order_acquire)) return 0;
    std::lock_guard lk(g_mu);
    return g_pending.owner_token;
}

uint64_t DeepWindow::OpensCompleted() {
    return g_opens_completed.load(std::memory_order_acquire);
}

void DeepWindow::ScheduleOpenAfter(const int64_t delay_ms,
                                   const DeepWindowSpec& spec) {
    {
        std::lock_guard lk(g_mu);
        if (PendingIsQueued()) {
            // Keeps the queued request rather than replacing it. Overwriting
            // would let a scheduled window cancel a rule's window - or the
            // reverse - purely on call order.
            GFL_LOG_DEBUG("[DeepWindow] open request ignored: one is already "
                          "queued (owner_token=", g_pending.owner_token, ")");
            return;
        }
        g_pending.spec = spec;
        // Untagged: nobody is counting this window against a budget. Cleared
        // explicitly, so replacing a tagged request cannot leave its owner
        // attached to somebody else's window.
        g_pending.owner_token = 0;
        g_pending_open_at_ns.store(
            delay_ms > 0 ? detail::GetTimestampNs() + delay_ms * 1000000 : 0,
            std::memory_order_relaxed);
        // Under the lock, for the same reason as the tagged path: the launch
        // beat still reads the spec only once this is set, and a concurrent
        // caller can no longer slip in between the two.
        g_open_requested.store(true, std::memory_order_release);
    }
    GFL_LOG_DEBUG("[DeepWindow] open requested delay_ms=", delay_ms,
                  " duration_ms=", spec.max_duration_ms,
                  " max_launches=", spec.max_launches);
}

// Runs on the collector, off the CUPTI callback path. Claims the request
// before opening so nothing can act on it twice.
void DeepWindow::TakePendingOpen_() {
    if (!g_open_requested.exchange(false, std::memory_order_acq_rel)) return;

    DeepWindowSpec spec;
    {
        std::lock_guard lk(g_mu);
        spec = g_pending.spec;
        // Claim the token here, not in Open(): Open can still refuse, and a
        // token left behind would keep reading as "queued" for the rest of the
        // run, so a rule would wait forever for a window that was already
        // turned down.
        g_claimed_token = g_pending.owner_token;
        g_pending.owner_token = 0;
    }
    // Outside the lock: Open takes it too.
    Open(spec);
    g_claimed_token = 0;
}

void DeepWindow::OnLaunch() {
    // Arming is the collector's job too - see ServicePending. This callback
    // only counts.
    if (!g_active.load(std::memory_order_acquire)) return;

    g_launches_covered.fetch_add(1, std::memory_order_relaxed);

    if (g_launches_remaining.load(std::memory_order_relaxed) > 0) {
        // fetch_sub returns the PREVIOUS value, so 1 means this launch
        // consumed the last of the budget.
        if (g_launches_remaining.fetch_sub(1, std::memory_order_relaxed) <= 1) {
            RequestClose_(DeepWindowClose::LaunchBudget);
        }
    }
}

void DeepWindow::RequestClose_(const DeepWindowClose reason) {
    // First reason wins; a later bound can't relabel a close already asked
    // for. Handing this to the collector instead of closing here is the
    // whole point - see OnLaunch.
    bool expected = false;
    if (g_close_requested.compare_exchange_strong(expected, true,
                                                  std::memory_order_acq_rel)) {
        g_close_reason.store(static_cast<int>(reason), std::memory_order_release);
    }
}

namespace {

bool CloseDue_() {
    if (!g_active.load(std::memory_order_acquire)) return false;
    if (g_close_requested.load(std::memory_order_acquire)) return true;
    const int64_t deadline = g_deadline_ns.load(std::memory_order_relaxed);
    return deadline > 0 && detail::GetTimestampNs() >= deadline;
}

bool OpenDue_() {
    if (g_active.load(std::memory_order_acquire)) return false;
    if (!g_open_requested.load(std::memory_order_acquire)) return false;
    const int64_t due = g_pending_open_at_ns.load(std::memory_order_relaxed);
    return due <= 0 || detail::GetTimestampNs() >= due;
}

}  // namespace

bool DeepWindow::HasPendingWork() { return CloseDue_() || OpenDue_(); }

void DeepWindow::ServicePending() {
    if (CloseDue_()) {
        const DeepWindowClose reason =
            g_close_requested.load(std::memory_order_acquire)
                ? static_cast<DeepWindowClose>(
                      g_close_reason.load(std::memory_order_acquire))
                : DeepWindowClose::Deadline;
        Close(reason);
        return;
    }
    if (OpenDue_()) TakePendingOpen_();
}

void DeepWindow::ResetForTesting() {
    std::lock_guard<std::mutex> lk(g_mu);
    g_active.store(false, std::memory_order_release);
    g_next_open_token.store(1, std::memory_order_relaxed);
    g_last_opened_token.store(0, std::memory_order_relaxed);
    g_opens_completed.store(0, std::memory_order_relaxed);
    g_pending = PendingOpen{};
    g_deadline_ns.store(0, std::memory_order_relaxed);
    g_launches_remaining.store(0, std::memory_order_relaxed);
    g_launches_covered.store(0, std::memory_order_relaxed);
    g_close_requested.store(false, std::memory_order_relaxed);
    g_close_reason.store(static_cast<int>(DeepWindowClose::Deadline),
                         std::memory_order_relaxed);
    g_open_requested.store(false, std::memory_order_relaxed);
    g_pending_open_at_ns.store(0, std::memory_order_relaxed);
    g_pending = PendingOpen{};
    g_last_close_ns.store(0, std::memory_order_relaxed);
    g_opened_ns = 0;
    g_requested_duration_ms = 0;
    g_requested_max_launches = 0;
    g_name.clear();
    g_scope_instance_id = 0;
}

// ---- Public API ----

void deepWindow(const DeepWindowSpec& spec) {
    DeepWindowSpec resolved = spec;
    ApplyEnvDefaults(resolved);
    DeepWindow::Open(resolved);
}

void deepWindow(const int64_t max_duration_ms, const uint64_t max_launches) {
    DeepWindowSpec spec;
    spec.max_duration_ms = max_duration_ms;
    spec.max_launches = max_launches;
    deepWindow(spec);
}

// Reads the launcher's time-based trigger, the only one available when the
// target's source can't be edited. Called once from init(); a no-op unless
// GPUFL_DEEP_AFTER_MS is set.
void scheduleEnvDeepWindow() {
    const int64_t after_ms = EnvUnsignedOr(env::kDeepAfterMs, -1);
    if (after_ms < 0) return;

    DeepWindowSpec spec;
    ApplyEnvDefaults(spec);
    if (spec.max_duration_ms == 0 && spec.max_launches == 0) {
        GFL_LOG_ERROR(
            env::kDeepAfterMs,
            " is set but no window bound is - the window would never close. "
            "Set ", env::kDeepWindowMs, " or ", env::kDeepWindowMaxLaunches, ".");
        return;
    }
    DeepWindow::ScheduleOpenAfter(after_ms, spec);
}

void deepWindowClose() { DeepWindow::Close(DeepWindowClose::Manual); }

bool deepWindowActive() { return DeepWindow::Active(); }

}  // namespace gpufl

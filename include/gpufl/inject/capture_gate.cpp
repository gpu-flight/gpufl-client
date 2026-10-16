#include "gpufl/inject/capture_gate.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "gpufl/core/env_vars.hpp"
#include "gpufl/core/events/lifecycle_events.hpp"

#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace gpufl::inject {
namespace {

// Generous on purpose: a slow but healthy gpufl::init (PerfWorks setup on a
// large GPU) must not fail a manual handshake or be killed by the launcher.
constexpr int64_t kDefaultWaitMs = 60'000;
constexpr int64_t kMaxWaitMs = 600'000;
constexpr int kDefaultDelayMs = 1;
constexpr std::size_t kMaxReasonBytes = 200;

// Which gate, if any, this thread belongs to. A gate never waits on its own
// initializer, inside the driver's injection callback, or inside the driver
// init it triggered: each of those would wait for itself.
thread_local const CaptureGate* t_initializer = nullptr;
thread_local const CaptureGate* t_in_callback = nullptr;
thread_local const CaptureGate* t_in_trigger = nullptr;
thread_local int t_wrapper_depth = 0;

class ThreadMark {
   public:
    ThreadMark(const CaptureGate*& slot, const CaptureGate* gate)
        : slot_(slot), previous_(slot) {
        slot_ = gate;
    }
    ~ThreadMark() { slot_ = previous_; }
    ThreadMark(const ThreadMark&) = delete;
    ThreadMark& operator=(const ThreadMark&) = delete;

   private:
    const CaptureGate*& slot_;
    const CaptureGate* previous_;
};

bool ParseInt(const char* text, int64_t& out) {
    if (!text || !*text) return false;
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text, &end, 10);
    if (errno != 0 || !end || *end != '\0') return false;
    out = value;
    return true;
}

GateStep Call(const std::function<GateStep()>& fn, const char* missing,
              const char* exception) {
    if (!fn) return {false, missing};
    try {
        return fn();
    } catch (...) {
        return {false, exception};
    }
}

std::string Bounded(std::string text) {
    if (text.size() > kMaxReasonBytes) text.resize(kMaxReasonBytes);
    return text;
}

void AppendJsonString(std::string& out, const std::string& value) {
    out += '"';
    for (const unsigned char c : value) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    static const char* kHex = "0123456789abcdef";
                    out += "\\u00";
                    out += kHex[c >> 4];
                    out += kHex[c & 0xf];
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    out += '"';
}

void AppendParticipant(std::string& out, const char* name,
                       const GateParticipant& p) {
    out += ",\"";
    out += name;
    out += "\":{\"expected\":";
    out += p.expected ? "true" : "false";
    out += ",\"returned\":";
    out += p.returned ? "true" : "false";
    out += ",\"ok\":";
    out += p.ok ? "true" : "false";
    out += '}';
}

}  // namespace

const char* GateLifecycleName(const GateLifecycle value) {
    switch (value) {
        case GateLifecycle::NotRequested: return "not_requested";
        case GateLifecycle::DriverInitializing: return "driver_initializing";
        case GateLifecycle::DriverReady: return "driver_ready";
        case GateLifecycle::CaptureInitializing: return "capture_initializing";
        case GateLifecycle::Finished: return "finished";
    }
    return "unknown";
}

const char* GateOwnerName(const GateOwner value) {
    switch (value) {
        case GateOwner::None: return "none";
        case GateOwner::Gate: return "gate";
        case GateOwner::Callback: return "callback";
        case GateOwner::Manual: return "manual";
    }
    return "unknown";
}

const char* GateReadinessName(const GateReadiness value) {
    switch (value) {
        case GateReadiness::Pending: return "pending";
        case GateReadiness::Ready: return "ready";
        case GateReadiness::Failed: return "failed";
    }
    return "unknown";
}

GateConfig GateConfigFromEnv(
    const std::function<const char*(const char*)>& getenv_fn) {
    const auto get = [&](const char* name) -> const char* {
        const char* value = getenv_fn ? getenv_fn(name) : std::getenv(name);
        return value && *value ? value : nullptr;
    };
    GateConfig config;

    int64_t delay = kDefaultDelayMs;
    bool delay_set = false;
    if (int64_t parsed = 0;
        ParseInt(get(env::kInjectInitDelayMs), parsed) && parsed >= 0 &&
        parsed <= std::numeric_limits<int>::max()) {
        delay = parsed;
        delay_set = true;
    }
    config.delay_ms = static_cast<int>(delay);

    if (const char* manual = get(env::kInjectManualInit);
        manual && std::strcmp(manual, "1") == 0) {
        config.mode = GateMode::Manual;
    } else if (int64_t warmup = 0;
               ParseInt(get(env::kInjectWarmupMs), warmup) && warmup > 0 &&
               warmup <= std::numeric_limits<int>::max()) {
        config.mode = GateMode::Warmup;
        config.delay_ms = static_cast<int>(warmup);
    }

    if (const char* wait = get(env::kInjectLaunchWaitMs)) {
        int64_t parsed = 0;
        if (ParseInt(wait, parsed) && parsed >= 1 && parsed <= kMaxWaitMs) {
            config.wait_ns = parsed * 1'000'000LL;
        } else {
            config.wait_ns = kDefaultWaitMs * 1'000'000LL;
            config.wait_config_invalid = true;
        }
    }

    if (const char* protocol = get(env::kInjectProtocol)) {
        config.protocol = protocol;
    }
    // An older launcher wrote --warmup into the init delay. Without the
    // protocol variable a long delay may have been one, and a warmup run is
    // not a qualified capture.
    config.legacy_launcher_delay = config.protocol.empty() &&
                                   config.mode == GateMode::Automatic &&
                                   delay_set && delay > kDefaultDelayMs;
    return config;
}

CaptureGate::CaptureGate(GateConfig config, GateHooks hooks)
    : config_(std::move(config)), hooks_(std::move(hooks)) {
    state_.wait_config_invalid = config_.wait_config_invalid;
    state_.protocol = config_.protocol;
    if (config_.mode == GateMode::Warmup) {
        state_.coverage_notes.insert("warmup_unqualified");
    }
    if (config_.legacy_launcher_delay) {
        state_.coverage_notes.insert("legacy_launcher_delay");
    }
}

bool CaptureGate::Settled() const {
    return settled_.load(std::memory_order_acquire);
}

GateSnapshot CaptureGate::Snapshot() const {
    std::lock_guard lock(mu_);
    return state_;
}

void CaptureGate::ElectLocked(const GateOwner owner) {
    state_.owner = owner;
    state_.lifecycle = GateLifecycle::DriverInitializing;
    const int64_t now = hooks_.now_ns ? hooks_.now_ns() : 0;
    // One immutable budget: the worker's requested delay (the readiness delay,
    // or a --warmup) once, plus the initialization allowance. Nothing renews it.
    const int64_t max = std::numeric_limits<int64_t>::max();
    const int64_t delay_ns =
        owner == GateOwner::Manual ? 0 : static_cast<int64_t>(config_.delay_ms) * 1'000'000LL;
    const int64_t budget = delay_ns > max - config_.wait_ns ? max : delay_ns + config_.wait_ns;
    state_.deadline_ns = now > max - budget ? max : now + budget;
}

void CaptureGate::LatchFailedLocked(const std::string& reason) {
    if (state_.readiness != GateReadiness::Pending) return;
    state_.readiness = GateReadiness::Failed;
    state_.reason = Bounded(reason);
    settled_.store(true, std::memory_order_release);
}

void CaptureGate::ResolveLocked() {
    const GateParticipant* participants[] = {&state_.trigger, &state_.worker,
                                             &state_.handshake};
    bool any = false;
    bool all_returned = true;
    bool all_ok = true;
    for (const GateParticipant* p : participants) {
        if (!p->expected) continue;
        any = true;
        all_returned = all_returned && p->returned;
        all_ok = all_ok && p->ok;
    }
    if (!any || !all_returned) return;
    // Every participant has stopped using initialization-owned state.
    state_.lifecycle = GateLifecycle::Finished;
    if (state_.readiness != GateReadiness::Pending) return;
    if (!all_ok) {
        LatchFailedLocked("init_failed");
        return;
    }
    const int64_t now = hooks_.now_ns ? hooks_.now_ns() : 0;
    if (now > state_.deadline_ns) {
        // No waiter has to be present for the budget to apply.
        LatchFailedLocked("capture_init_deadline_exceeded");
        return;
    }
    state_.readiness = GateReadiness::Ready;
    settled_.store(true, std::memory_order_release);
}

void CaptureGate::PublishLocked() {
    // Under mu_: a state is on disk before any thread can act on it, a newer
    // record can never be replaced by an older one, and nothing is published
    // after the process saw Finished and went on into teardown.
    ++state_.seq;
    if (hooks_.publish) hooks_.publish(state_);
}

bool CaptureGate::WaitLocked(std::unique_lock<std::mutex>& lock,
                             const bool until_finished) {
    for (;;) {
        const bool done = until_finished
                              ? state_.lifecycle == GateLifecycle::Finished
                              : state_.readiness != GateReadiness::Pending;
        if (done) return true;
        const int64_t now = hooks_.now_ns ? hooks_.now_ns() : 0;
        if (now >= state_.deadline_ns) {
            if (state_.readiness == GateReadiness::Pending) {
                LatchFailedLocked("capture_wait_timeout");
                PublishLocked();
                cv_.notify_all();
            }
            return until_finished
                       ? state_.lifecycle == GateLifecycle::Finished
                       : false;
        }
        cv_.wait_for(lock, std::chrono::nanoseconds(state_.deadline_ns - now));
    }
}

void CaptureGate::StartWorkerLocked() {
    worker_started_ = true;
    state_.worker.expected = true;
}

void CaptureGate::Enter(const std::function<GateStep()>& trigger) {
    if (Settled()) return;
    if (t_initializer == this || t_in_callback == this || t_in_trigger == this) {
        return;
    }
    if (t_wrapper_depth > 1) return;  // an inner wrapper of an outer one
    if (config_.mode == GateMode::Warmup) return;

    std::unique_lock lock(mu_);
    if (config_.mode == GateMode::Manual) {
        // The host must finish the handshake before capture-relevant work.
        if (state_.readiness == GateReadiness::Pending &&
            state_.coverage_notes.insert("manual_protocol_violation").second) {
            PublishLocked();
            lock.unlock();
        }
        return;
    }

    if (state_.lifecycle == GateLifecycle::NotRequested) {
        ElectLocked(GateOwner::Gate);
        state_.trigger.expected = true;
        PublishLocked();
        lock.unlock();

        GateStep step;
        {
            ThreadMark mark(t_in_trigger, this);
            step = Call(trigger, "trigger_missing", "trigger_exception");
        }

        lock.lock();
        state_.trigger.returned = true;
        state_.trigger.ok = step.ok;
        if (!step.ok) {
            LatchFailedLocked(step.reason.empty() ? "trigger_failed" : step.reason);
        } else if (!worker_started_) {
            // Driver init succeeded but never reached InitializeInjection.
            LatchFailedLocked("no_injection_callback");
        }
        ResolveLocked();
        PublishLocked();
        cv_.notify_all();
    }
    WaitLocked(lock, /*until_finished=*/false);
}

void CaptureGate::OnInjectionCallback() {
    ThreadMark mark(t_in_callback, this);
    std::unique_lock lock(mu_);
    ++state_.callbacks_seen;
    bool spawn = false;
    if (config_.mode != GateMode::Manual) {
        if (state_.lifecycle == GateLifecycle::NotRequested) {
            // Driver init came from a route no gate owns.
            ElectLocked(GateOwner::Callback);
            if (config_.mode == GateMode::Automatic) {
                state_.coverage_notes.insert("unsupported_first_touch");
            }
            StartWorkerLocked();
            spawn = true;
        } else if (state_.owner == GateOwner::Gate && !worker_started_ &&
                   !state_.trigger.returned) {
            // The gate owner's trigger is on the stack: arrange its worker.
            StartWorkerLocked();
            spawn = true;
        }
    }
    PublishLocked();
    lock.unlock();
    if (spawn) {
        if (!hooks_.spawn || !hooks_.spawn([this] { RunWorker(); })) {
            std::unique_lock relock(mu_);
            state_.worker.returned = true;
            LatchFailedLocked("worker_spawn_unavailable");
            ResolveLocked();
            PublishLocked();
            cv_.notify_all();
            relock.unlock();
        }
    }
}

void CaptureGate::RunWorker() {
    ThreadMark mark(t_initializer, this);
    const GateStep barrier =
        Call(hooks_.driver_barrier, "driver_barrier_missing",
             "driver_barrier_exception");
    {
        std::unique_lock lock(mu_);
        if (!barrier.ok) {
            state_.worker.returned = true;
            state_.worker.ok = false;
            LatchFailedLocked(barrier.reason.empty() ? "cuinit_failed"
                                                     : barrier.reason);
            ResolveLocked();
            PublishLocked();
            cv_.notify_all();
            lock.unlock();
            return;
        }
        if (state_.lifecycle == GateLifecycle::DriverInitializing) {
            state_.lifecycle = GateLifecycle::DriverReady;
        }
        PublishLocked();
        lock.unlock();
    }
    {
        // The delay is not CUDA work: an exit that starts during it stops the
        // worker here, before CUPTI, so ordinary teardown stays safe.
        std::unique_lock lock(mu_);
        cv_.wait_for(lock, std::chrono::milliseconds(config_.delay_ms),
                     [this] { return exit_requested_; });
        if (exit_requested_) {
            state_.worker.returned = true;
            state_.worker.ok = false;
            LatchFailedLocked("exit_before_capture");
            ResolveLocked();
            PublishLocked();
            cv_.notify_all();
            return;
        }
        state_.lifecycle = GateLifecycle::CaptureInitializing;
        PublishLocked();
    }
    const GateStep init =
        Call(hooks_.initialize, "initialize_missing", "init_exception");
    std::unique_lock lock(mu_);
    state_.worker.returned = true;
    state_.worker.ok = init.ok;
    if (!init.ok) {
        LatchFailedLocked(init.reason.empty() ? "init_failed" : init.reason);
    }
    ResolveLocked();
    PublishLocked();
    cv_.notify_all();
    lock.unlock();
}

int CaptureGate::ManualHandshake(const std::function<GateStep()>& driver_trigger) {
    if (config_.mode != GateMode::Manual) {
        // Without manual mode the handshake is one more outer entry.
        Enter(driver_trigger);
        std::lock_guard lock(mu_);
        return state_.readiness == GateReadiness::Ready ? 1 : 0;
    }
    std::unique_lock lock(mu_);
    if (t_initializer == this) {
        ++state_.handshake_reentries;
        return 0;
    }
    if (state_.lifecycle == GateLifecycle::NotRequested) {
        ElectLocked(GateOwner::Manual);
        state_.handshake.expected = true;
        PublishLocked();
        lock.unlock();

        GateStep result;
        {
            ThreadMark mark(t_initializer, this);
            result = Call(hooks_.driver_barrier, "driver_barrier_missing",
                          "driver_barrier_exception");
            if (result.ok) {
                lock.lock();
                state_.lifecycle = GateLifecycle::CaptureInitializing;
                PublishLocked();
                lock.unlock();
                result = Call(hooks_.initialize, "initialize_missing",
                              "init_exception");
            }
        }

        lock.lock();
        state_.handshake.returned = true;
        state_.handshake.ok = result.ok;
        if (!result.ok) {
            LatchFailedLocked(result.reason.empty() ? "init_failed" : result.reason);
        }
        ResolveLocked();
        PublishLocked();
        cv_.notify_all();
    } else {
        WaitLocked(lock, /*until_finished=*/false);
    }
    return state_.readiness == GateReadiness::Ready &&
                   state_.coverage_notes.count("manual_protocol_violation") == 0
               ? 1
               : 0;
}

bool CaptureGate::AwaitAtExit() {
    std::unique_lock lock(mu_);
    if (state_.lifecycle == GateLifecycle::NotRequested ||
        state_.lifecycle == GateLifecycle::Finished) {
        return true;
    }
    exit_requested_ = true;  // a worker still in its delay stands down
    cv_.notify_all();
    if (WaitLocked(lock, /*until_finished=*/true)) return true;
    state_.coverage_notes.insert("exit_while_initializing");
    PublishLocked();
    lock.unlock();
    return false;
}

void CaptureGate::AddCoverageNote(const std::string& note) {
    std::unique_lock lock(mu_);
    if (!state_.coverage_notes.insert(note).second) return;
    PublishLocked();
    lock.unlock();
}

WrapperScope::WrapperScope() { ++t_wrapper_depth; }
WrapperScope::~WrapperScope() { --t_wrapper_depth; }

void AppendGateCapabilities(const GateSnapshot& s, CaptureCapabilitiesEvent& evt) {
    std::vector<std::string> reasons;
    if (s.readiness == GateReadiness::Pending) {
        reasons.emplace_back("readiness_pending");
    } else if (s.readiness == GateReadiness::Failed) {
        reasons.emplace_back("readiness_failed");
    }
    for (const std::string& note : s.coverage_notes) reasons.push_back(note);
    // Rows the timestamp counters marked (see capture_capability_resolver).
    for (const CaptureCapability& cap : evt.capabilities) {
        const std::string& code = cap.reason_code;
        const auto ends_with = [&code](const char* suffix) {
            const std::size_t n = std::strlen(suffix);
            return code.size() >= n && code.compare(code.size() - n, n, suffix) == 0;
        };
        if (ends_with("_timestamps_unavailable") || ends_with("_timestamps_invalid") ||
            ends_with("_timestamps_partial")) {
            reasons.push_back(code);
        }
    }

    CaptureCapability readiness;
    readiness.feature = "capture_readiness";
    readiness.requested = true;
    readiness.status = GateReadinessName(s.readiness);
    readiness.mode = GateOwnerName(s.owner);
    readiness.reason_code = s.reason;
    readiness.message = std::string("lifecycle=") + GateLifecycleName(s.lifecycle) +
                        " callbacks=" + std::to_string(s.callbacks_seen) +
                        " protocol=" + (s.protocol.empty() ? "absent" : s.protocol);
    evt.capabilities.push_back(std::move(readiness));

    CaptureCapability coverage;
    coverage.feature = "capture_coverage";
    coverage.requested = true;
    coverage.mode = "injection_gate";
    if (reasons.empty()) {
        coverage.status = "no_known_loss";
        coverage.message =
            "No known capture loss. This is not a completeness proof: a kernel "
            "that produced no record cannot be counted.";
    } else {
        coverage.status = "incomplete";
        for (const std::string& reason : reasons) {
            if (!coverage.reason_code.empty()) coverage.reason_code += ',';
            coverage.reason_code += reason;
        }
        coverage.message = "The timeline of this session may be incomplete.";
    }
    evt.capabilities.push_back(std::move(coverage));
}

std::string GateStatusJson(const GateSnapshot& s, const std::string& nonce,
                           const long pid) {
    std::string out;
    out.reserve(1024);
    out += "{\"schema\":1,\"nonce\":";
    AppendJsonString(out, nonce);
    out += ",\"pid\":" + std::to_string(pid);
    out += ",\"seq\":" + std::to_string(s.seq);
    out += ",\"lifecycle\":\"";
    out += GateLifecycleName(s.lifecycle);
    out += "\",\"owner\":\"";
    out += GateOwnerName(s.owner);
    out += "\",\"readiness\":\"";
    out += GateReadinessName(s.readiness);
    out += "\",\"reason\":";
    AppendJsonString(out, s.reason);
    out += ",\"coverage_notes\":[";
    bool first = true;
    for (const std::string& note : s.coverage_notes) {
        if (!first) out += ',';
        first = false;
        AppendJsonString(out, note);
    }
    out += "],\"deadline_monotonic_ns\":" + std::to_string(s.deadline_ns);
    AppendParticipant(out, "trigger", s.trigger);
    AppendParticipant(out, "worker", s.worker);
    AppendParticipant(out, "handshake", s.handshake);
    out += ",\"callbacks_seen\":" + std::to_string(s.callbacks_seen);
    out += ",\"handshake_reentries\":" + std::to_string(s.handshake_reentries);
    out += ",\"wait_config_invalid\":";
    out += s.wait_config_invalid ? "true" : "false";
    out += ",\"protocol\":";
    AppendJsonString(out, s.protocol.substr(0, 16));
    out += "}\n";
    return out;
}

bool WriteGateStatusFile(const std::string& path, const std::string& json) {
#if defined(_WIN32)
    (void)path;
    (void)json;
    return false;
#else
    if (path.empty() || json.size() > 4096) return false;
    const std::string temp = path + ".tmp";
    const int fd = ::open(temp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC,
                          0600);
    if (fd < 0) return false;
    std::size_t written = 0;
    bool ok = true;
    while (written < json.size()) {
        const ssize_t n = ::write(fd, json.data() + written, json.size() - written);
        if (n < 0) {
            if (errno == EINTR) continue;
            ok = false;
            break;
        }
        written += static_cast<std::size_t>(n);
    }
    ok = ok && ::fsync(fd) == 0;
    ok = ::close(fd) == 0 && ok;
    ok = ok && ::rename(temp.c_str(), path.c_str()) == 0;
    if (!ok) ::unlink(temp.c_str());
    return ok;
#endif
}

}  // namespace gpufl::inject

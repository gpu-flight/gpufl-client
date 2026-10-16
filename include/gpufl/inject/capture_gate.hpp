#pragma once

// Capture readiness for Linux CUDA injection.
//
// The injected library starts CUPTI from a worker thread after the driver's
// own initialization. Without a gate, the application thread continues into
// context creation and module loading while that worker attaches. On an L4
// (2026-10-16) whole Trace sessions received every kernel record with zero
// timestamps; that is consistent with this overlap, but the CUPTI-internal
// cause is not established. The gate makes the application's first supported
// CUDA call wait until capture is ready, within one bounded deadline, and
// records every other outcome.
//
// Limits: status publication runs under the gate mutex, so a slow filesystem
// can hold waiters past the nominal deadline (the launcher watchdog bounds
// that only after its first accepted record). Coverage is a weaker claim
// than completeness, and nothing here is evidence about real driver locks.
//
// Three separate facts, see the design (agent_comm l4-qualification-20261015,
// client-capture-gate-design-v3.md + v3.1):
//   * lifecycle - what the initializer is doing (diagnostics only);
//   * readiness - latched once: Ready or Failed(reason);
//   * coverage notes - reasons the run cannot be reported as no_known_loss.
//
// The component is pure: driver calls, the worker thread, the clock and the
// status file are hooks, so the state machine runs on CPU in unit tests.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <set>
#include <string>

namespace gpufl {
struct CaptureCapabilitiesEvent;
}  // namespace gpufl

namespace gpufl::inject {

enum class GateLifecycle : int {
    NotRequested = 0,
    DriverInitializing,
    DriverReady,
    CaptureInitializing,
    Finished,
};

enum class GateOwner : int { None = 0, Gate, Callback, Manual };

enum class GateReadiness : int { Pending = 0, Ready, Failed };

enum class GateMode : int {
    // The first supported CUDA call triggers driver init and waits.
    Automatic = 0,
    // The host calls GpuFlightInitializeInjectionAfterCuda after cuInit.
    Manual,
    // --warmup: capture starts late on purpose; nothing waits.
    Warmup,
};

struct GateConfig {
    GateMode mode = GateMode::Automatic;
    int64_t wait_ns = 15'000'000'000LL;
    bool wait_config_invalid = false;
    // Worker sleep between the driver barrier and CUPTI setup: the readiness
    // delay, or the warmup in Warmup mode.
    int delay_ms = 1;
    // The launcher's protocol variable was absent while a legacy delay above
    // the default was set: it may have been a warmup.
    bool legacy_launcher_delay = false;
    // Protocol the launcher announced ("" when absent).
    std::string protocol;
};

/** Reads the gate configuration from the environment (getenv by default). */
GateConfig GateConfigFromEnv(
    const std::function<const char*(const char*)>& getenv_fn);

struct GateStep {
    bool ok = false;
    // Machine-readable cause when !ok, e.g. "cuinit_failed:100".
    std::string reason;
};

struct GateParticipant {
    bool expected = false;
    bool returned = false;
    bool ok = false;
};

struct GateSnapshot {
    uint64_t seq = 0;
    GateLifecycle lifecycle = GateLifecycle::NotRequested;
    GateOwner owner = GateOwner::None;
    GateReadiness readiness = GateReadiness::Pending;
    std::string reason;
    std::set<std::string> coverage_notes;
    // CLOCK_MONOTONIC ns; 0 until an owner is elected.
    int64_t deadline_ns = 0;
    GateParticipant trigger;
    GateParticipant worker;
    GateParticipant handshake;
    uint64_t callbacks_seen = 0;
    uint64_t handshake_reentries = 0;
    bool wait_config_invalid = false;
    std::string protocol;
};

struct GateHooks {
    // CLOCK_MONOTONIC in ns (clock_gettime), the value the launcher compares.
    std::function<int64_t()> now_ns;
    // Starts `body` on a new detached thread; false when no thread could start.
    std::function<bool(std::function<void()> body)> spawn;
    // The worker's checked post-driver barrier (cuInit).
    std::function<GateStep()> driver_barrier;
    // gpufl::init plus the Trace facility check. Never throws out.
    std::function<GateStep()> initialize;
    // Publishes the state (the status file), with the gate mutex held so a
    // record is written before anyone acts on it. Must not call back into the
    // gate or into CUDA; snapshots carry a strictly increasing seq.
    std::function<void(const GateSnapshot&)> publish;
};

const char* GateLifecycleName(GateLifecycle value);
const char* GateOwnerName(GateOwner value);
const char* GateReadinessName(GateReadiness value);

class CaptureGate {
   public:
    CaptureGate(GateConfig config, GateHooks hooks);
    CaptureGate(const CaptureGate&) = delete;
    CaptureGate& operator=(const CaptureGate&) = delete;

    /**
     * A supported CUDA entry point, before its real call. `trigger` performs
     * that route's driver initialization (runtime: the real
     * cudaGetDeviceCount; driver: the checked cuInit) and is called at most
     * once per process, by the elected gate owner.
     *
     * Returns at once when readiness is settled, on gpufl's own threads, inside
     * the injection callback, inside the trigger, and in Manual or Warmup mode.
     * Otherwise waits until readiness settles or the deadline passes.
     */
    void Enter(const std::function<GateStep()>& trigger);

    /** NVIDIA's InitializeInjection (Linux). Never waits. */
    void OnInjectionCallback();

    /**
     * GpuFlightInitializeInjectionAfterCuda. In Manual mode it is the only
     * initializer; returns 1 only for Ready with no protocol violation.
     * `driver_trigger` is used only in Automatic mode, as an outer entry.
     */
    int ManualHandshake(const std::function<GateStep()>& driver_trigger);

    /**
     * The atexit guard. true: ordinary teardown may continue (nothing ran, or
     * every participant returned). false: an initializer is still live after
     * the deadline; the caller must end the process without teardown.
     */
    bool AwaitAtExit();

    /** Adds a coverage note decided outside the gate (e.g. at exit). */
    void AddCoverageNote(const std::string& note);

    GateSnapshot Snapshot() const;

    /** One acquire load: readiness settled or the deadline already passed. */
    bool Settled() const;

   private:
    // All under mu_.
    void ElectLocked(GateOwner owner);
    void LatchFailedLocked(const std::string& reason);
    void ResolveLocked();
    void PublishLocked();
    bool WaitLocked(std::unique_lock<std::mutex>& lock, bool until_finished);
    void RunWorker();
    void StartWorkerLocked();

    const GateConfig config_;
    const GateHooks hooks_;

    mutable std::mutex mu_;
    std::condition_variable cv_;
    GateSnapshot state_;
    bool worker_started_ = false;
    bool exit_requested_ = false;
    std::atomic<bool> settled_{false};
};

/**
 * Counts nested gpufl CUDA wrappers on this thread. Only the outermost one
 * gates: an inner one is the real call of an outer wrapper reaching another
 * wrapped symbol. This cannot see CUDA work an unwrapped entry started.
 */
class WrapperScope {
   public:
    WrapperScope();
    ~WrapperScope();
    WrapperScope(const WrapperScope&) = delete;
    WrapperScope& operator=(const WrapperScope&) = delete;
};

/**
 * Appends capture_readiness and capture_coverage rows. Coverage is
 * "no_known_loss" only when readiness is Ready, no coverage note is set and no
 * kernel or memcpy row reports rejected timestamps. That is weaker than
 * completeness: a kernel that produced no record cannot be counted.
 */
void AppendGateCapabilities(const GateSnapshot& snapshot,
                            CaptureCapabilitiesEvent& evt);

/** The snapshot as the bootstrap status JSON (schema 1, at most 4 KiB). */
std::string GateStatusJson(const GateSnapshot& snapshot, const std::string& nonce,
                           long pid);

/**
 * Writes the status JSON to `path` with temp file + fsync + rename. Only
 * open/write/fsync/rename/close, so it needs neither the logger nor the
 * driver. Returns false when any step failed.
 */
bool WriteGateStatusFile(const std::string& path, const std::string& json);

}  // namespace gpufl::inject

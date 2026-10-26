#pragma once

#include "bootstrap_status.hpp"
#include "cli_parse.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace gpufl::launcher {

namespace fs = std::filesystem;

// The injected library ends the process with this status when an initializer
// is still inside CUDA/CUPTI at exit after its deadline (capture_gate).
constexpr int kInitWedgedExitCode = 86;

struct TraceProcessResult {
    int rc = 1;
    bool signaled = false;
    int signal = 0;
    bool launcher_error = false;
    // The launcher stopped the target at the window deadline (intentional —
    // not a crash and not a failure). rc/signal reflect the kill, not health.
    bool window_stopped = false;
    // The initialization watchdog killed the target: capture initialization
    // never finished within the library's published deadline plus the grace.
    bool init_watchdog_killed = false;
    // The child's last bootstrap status (invalid when none was read).
    BootstrapStatus bootstrap;
    // Empty when the read after exit confirmed `bootstrap`; otherwise why it
    // did not (the record above is then historical).
    std::string bootstrap_final_error;
    std::string error;
};

// How long the launcher lets a target run before stopping it. The window is a
// `gpufl trace`-only mechanism for bounding capture of a process that never
// exits on its own. run_ms == 0 means "wait for the target's natural exit"
// (the historical behavior).
struct RunOptions {
    int64_t run_ms = 0;            // stop the target after this wall-clock (0 = unbounded)
    int stop_grace_ms = 5000;     // after the stop signal, hard-kill if still alive
    // Per-child bootstrap status directory and nonce (empty: no watchdog).
    fs::path bootstrap_dir;
    std::string bootstrap_nonce;
    // Allowed past the library's initialization deadline before the kill.
    int64_t watchdog_grace_ms = 5000;
};

class TracePlatform {
   public:
    virtual ~TracePlatform() = default;

    virtual const char* platformName() const = 0;
    virtual const char* injectLibraryName() const = 0;

    virtual fs::path selfExe() const = 0;
    virtual std::vector<fs::path> injectLibCandidates(const fs::path& exe) const = 0;
    virtual fs::path defaultOutputDir(const std::string& tag) const = 0;
    virtual std::string defaultAppName(const std::string& command0) const = 0;

    virtual bool setEnv(const char* key, const std::string& value,
                        std::string& error) const = 0;
    /**
     * Remove a variable from the environment the target will inherit.
     *
     * Needed because some GPUFL variables are triggers that install by
     * EXISTING, not by their value: not setting one is not the same as the
     * target not seeing one. Not-present is the only way to say "off", so a
     * mode that owns a trigger has to be able to take the other one away.
     *
     * Removing a variable that is not set is success, not an error.
     */
    virtual bool unsetEnv(const char* key, std::string& error) const = 0;
    virtual bool prepareInjectionEnv(const fs::path& inject_lib,
                                     std::string& error) const = 0;
    virtual TraceProcessResult runProcess(
            const std::vector<std::string>& command,
            const RunOptions& opts) const = 0;
};

/** @brief platform.setEnv, printing the platform's reason on failure. */
bool setEnvOrPrint(const TracePlatform& platform, const char* key,
                   const std::string& value);
/** @brief platform.unsetEnv, printing the platform's reason on failure. */
bool unsetEnvOrPrint(const TracePlatform& platform, const char* key);

/**
 * @brief Publish the deep-window options into the target's environment.
 *
 * Separate from runTraceCommon so the trigger-ownership rule can be tested
 * without launching a process: which variable a mode sets is only half of it,
 * and the half that bit - which one it REMOVES - is invisible in any test that
 * only inspects what was set.
 *
 * @return false when the platform refused an environment change; the reason is
 *         already on stderr.
 */
bool applyDeepWindowEnv(const TraceArgs& args, const TracePlatform& platform);

/**
 * @brief Publish or scrub the launcher-owned segmentation environment.
 *
 * A non-segmented invocation removes all three internal variables so stale
 * parent-shell state cannot turn an ordinary trace into a segmented run.
 */
bool applySegmentationEnv(const TraceArgs& args, const std::string& run_id,
                          const TracePlatform& platform);

/**
 * @brief Publish the launch id every process of this trace inherits.
 *
 * `inherited` is GPUFL_LAUNCH_ID from the launcher's own environment (null
 * when unset). A valid value - a harness joining its records to these
 * sessions - is kept; an invalid one is refused, because replacing it would
 * silently break that join; an absent or empty one is generated as a UUIDv4.
 *
 * @return false when the inherited value is invalid or the platform refused
 *         the change; the reason is already on stderr.
 */
bool applyLaunchIdEnv(const char* inherited, const TracePlatform& platform);

int runTraceCommon(const TraceArgs& args, const TracePlatform& platform);

}  // namespace gpufl::launcher

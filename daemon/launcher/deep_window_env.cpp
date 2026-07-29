// Publishing the deep-window options into the target's environment.
//
// Its own translation unit so a test can link it without the rest of the trace
// command, which pulls in zlib, the agent launcher and the uploader. What is
// being tested here is a decision, not a launch.

#include <cstdio>
#include <string>

#include "cli_parse.hpp"
#include "gpufl/core/env_vars.hpp"
#include "trace_command_common.hpp"

namespace gpufl::launcher {

bool setEnvOrPrint(const TracePlatform& platform, const char* key,
                   const std::string& value) {
    std::string error;
    if (platform.setEnv(key, value, error)) return true;
    std::fprintf(stderr, "gpufl: %s\n", error.c_str());
    return false;
}

bool unsetEnvOrPrint(const TracePlatform& platform, const char* key) {
    std::string error;
    if (platform.unsetEnv(key, error)) return true;
    std::fprintf(stderr, "gpufl: %s\n", error.c_str());
    return false;
}

// --deep-*: bound how long the DEEP engines stay armed inside a target that
// keeps running. Distinct from --window, which bounds the target's lifetime.
// Asking for a deep window implies window-only arming, or the engines would be
// armed from the first kernel and the window would bound nothing.
bool applyDeepWindowEnv(const TraceArgs& args, const TracePlatform& platform) {
    if (!args.deep_requested) {
        // Not this run's business. A `--passes` run sets neither trigger and
        // scrubs neither: configuring a window purely through the environment
        // is the supported way to reach an engine the adaptive plan does not
        // select yet.
        return true;
    }
    if (!setEnvOrPrint(platform, env::kDeepArm, "window")) return false;

    // Both trigger variables install by EXISTING, whatever their value, so the
    // run has to say which one it owns and REMOVE the other. Merely not
    // setting one leaves whatever the parent shell had: a --deep-when run
    // under an exported GPUFL_DEEP_AFTER_MS opened a scheduled window at t=0
    // and the rule spent the run refused behind it, and a --deep-after run
    // under an exported GPUFL_DEEP_WHEN installed a rule nobody asked for. The
    // CLI rejects the two flags together; this is the same rule applied to the
    // environment the target inherits.
    if (!args.deep_when.empty()) {
        if (!unsetEnvOrPrint(platform, env::kDeepAfterMs)) return false;
        if (!setEnvOrPrint(platform, env::kDeepWhen, args.deep_when)) {
            return false;
        }
    } else {
        if (!unsetEnvOrPrint(platform, env::kDeepWhen)) return false;
        if (!setEnvOrPrint(platform, env::kDeepAfterMs,
                           std::to_string(args.deep_after_ms))) {
            return false;
        }
    }

    if (args.deep_for_ms > 0 &&
        !setEnvOrPrint(platform, env::kDeepWindowMs,
                       std::to_string(args.deep_for_ms))) {
        return false;
    }
    if (args.deep_launches > 0 &&
        !setEnvOrPrint(platform, env::kDeepWindowMaxLaunches,
                       std::to_string(args.deep_launches))) {
        return false;
    }
    if (args.deep_cooldown_ms > 0 &&
        !setEnvOrPrint(platform, env::kDeepWindowCooldownMs,
                       std::to_string(args.deep_cooldown_ms))) {
        return false;
    }
    return true;
}

}  // namespace gpufl::launcher

#pragma once

#include <cstdint>
#include <string>

namespace gpufl {
struct DeviceSample;
}

namespace gpufl::detail {

/**
 * @brief Owns the one conditional rule for the session.
 *
 * A facade rather than an object the caller holds, because the feed points sit
 * in the launch callback and the sampler while the evaluation sits on the
 * collector - three call sites that have no way to pass an instance between
 * them.
 *
 * One rule for the MVP. Multi-rule arbitration is deferred rather than guessed:
 * two rules that both want a window need a policy for who wins, and inventing
 * one before anybody has asked is how it ends up wrong.
 */
class DeepWindowRules {
public:
    /**
     * @brief Read GPUFL_DEEP_WHEN and friends. Called once from init().
     *
     * NEVER fails init(), whatever the configuration says. Config is parsed
     * during init, so a hard failure here would leave no session and no
     * telemetry writer - nowhere to record the very outcome a rejected rule has
     * to report. An invalid rule disables the trigger and nothing else; the
     * profiling session runs normally.
     */
    static void InstallFromEnv();

    /** @brief Advance the rule. Called on the collector beat. */
    static void Service();

    /** @brief A kernel launch was observed at the host launch API. */
    static void NoteKernelLaunch(int64_t ts_ns);
    /** @brief A completed kernel's duration, for recent_kernel_ms. */
    static void NoteKernelDuration(int64_t ts_ns, double duration_ms);
    /** @brief A successful device measurement. Never called on a poll. */
    static void NoteDeviceSample(const DeviceSample& sample, int64_t ts_ns);

    /**
     * @brief Write the rule summary. Called once during shutdown.
     *
     * Emitted even when the rule never fired: "log once" is invisible in the
     * UI, and no record at all is indistinguishable from a rule that was simply
     * never true.
     */
    static void Finish();

    /**
     * @brief Write the session's counter data-quality summary, if any.
     *
     * Called at shutdown beside Finish(), but independent of it: the event
     * reports what the APPLICATION sent (refused registrations, failed reads,
     * negative deltas) and exists whether or not a rule was configured.
     * Advances the bridge's session baseline, so an embedded re-init reports
     * only its own session's problems.
     */
    static void EmitCounterQuality();

    /** @brief True when a rule is installed - valid or refused. */
    static bool Installed();

    /** @brief Cheap enough for the launch callback to ask every launch. */
    static bool WantsLaunchFeed();

    static void ResetForTesting();
};

}  // namespace gpufl::detail

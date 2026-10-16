#pragma once

// The launcher's side of the injected library's bootstrap status file
// (gpufl/inject/capture_gate.hpp): reading it, and the initialization
// watchdog that kills a child whose capture initialization never finished.

#include <cstdint>
#include <filesystem>
#include <set>
#include <string>

namespace gpufl::launcher {

struct BootstrapStatus {
    // False: missing, unreadable, oversized, malformed, foreign or stale. Never
    // read as success.
    bool valid = false;
    std::string error;
    int64_t seq = 0;
    std::string lifecycle;
    std::string owner;
    std::string readiness;
    std::string reason;
    std::set<std::string> coverage_notes;
    int64_t deadline_monotonic_ns = 0;
    std::string protocol;
};

/** status-<pid>.json inside the launcher-owned per-child directory. */
std::filesystem::path BootstrapStatusPath(const std::filesystem::path& dir,
                                          long pid);

/**
 * Reads at most 4 KiB and requires schema 1, `nonce` and `pid`. A file that
 * fails any of those is returned invalid with the reason in `error`.
 */
BootstrapStatus ReadBootstrapStatus(const std::filesystem::path& dir, long pid,
                                    const std::string& nonce);

/** CLOCK_MONOTONIC in ns: the clock the library publishes its deadline in. */
int64_t MonotonicNowNs();

/**
 * Arms on the first accepted record that carries a deadline, keeps that first
 * deadline (a later value never renews the budget), and disarms for good once
 * the record says every initialization participant returned. A record with a
 * lower seq than one already seen is stale and ignored.
 */
class InitWatchdog {
   public:
    explicit InitWatchdog(int64_t grace_ns) : grace_ns_(grace_ns) {}

    /** Feeds one read; true means the child must be killed now. */
    bool Observe(const BootstrapStatus& status, int64_t now_ns);

    bool armed() const { return armed_; }
    bool disarmed() const { return disarmed_; }
    const BootstrapStatus& last() const { return last_; }
    bool seen_valid() const { return seen_valid_; }

   private:
    int64_t grace_ns_;
    bool armed_ = false;
    bool disarmed_ = false;
    bool seen_valid_ = false;
    int64_t deadline_ns_ = 0;
    int64_t last_seq_ = -1;
    BootstrapStatus last_;
};

/**
 * The last accepted record plus whether the read after exit confirmed it.
 * `final_error` is empty only when that read was valid and current; otherwise
 * it names what went wrong (missing, malformed, foreign_*, stale_final_seq),
 * and the accepted record is historical, not a fresh observation.
 */
struct FinalBootstrapObservation {
    BootstrapStatus accepted;
    std::string final_error;
};

FinalBootstrapObservation CombineFinalBootstrap(const InitWatchdog& watchdog,
                                                const BootstrapStatus& final_read);

/** One line for the run summary, e.g. "ready owner=gate" or "unavailable (...)". */
std::string DescribeBootstrapStatus(const BootstrapStatus& status);

/** A fresh 128-bit hex nonce. */
std::string MakeBootstrapNonce();

}  // namespace gpufl::launcher

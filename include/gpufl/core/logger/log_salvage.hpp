#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <system_error>

namespace gpufl {

struct LogSalvageResult {
    /** Windows published into the session directory by this pass. */
    int salvaged = 0;
    /**
     * Artifacts left in `.tmp` for a later pass or for manual recovery.
     * Nonzero keeps `.tmp` alive, so the session still looks unfinished -
     * that is the point: the data may still be recoverable.
     */
    int deferred = 0;
    /**
     * Windows whose events are GONE and can never be recovered, so the
     * artifact was discarded to let the session finish. A TERMINAL state,
     * deliberately not folded into `deferred` (which would pin `.tmp`
     * forever) and never into `salvaged`. Callers must surface it: an
     * unreported loss here looks exactly like a clean session downstream.
     */
    int lost_windows = 0;
    /**
     * Session directories deliberately not touched because another live
     * process owns their OS lock. This is not a salvage failure and must not
     * be treated as an orphan or completion signal.
     */
    int active_sessions_skipped = 0;
};

/**
 * Result of moving one completed spool artifact into its published name
 * without ever replacing an existing destination.
 */
enum class MoveFileNoReplaceResult {
    Moved,
    DestinationExists,
    Failed,
};

/**
 * Atomically claim `to` and move `from` there without replacement.
 *
 * `std::filesystem::rename` replaces an existing destination on POSIX and
 * therefore cannot enforce the transport-window uniqueness contract. This
 * helper uses a platform no-replace primitive (or a same-filesystem
 * hard-link/unlink fallback) so a concurrent publisher degrades to a visible
 * collision instead of destroying the older window.
 */
MoveFileNoReplaceResult moveFileNoReplace(
    const std::filesystem::path& from,
    const std::filesystem::path& to,
    std::error_code& ec);

/**
 * True when `path` is a non-empty file that decodes cleanly as gzip all the
 * way to EOF. Existence is NOT proof: a zero-length file decodes as a clean
 * empty stream, and a truncated one only fails partway through - so any code
 * about to delete a raw source because "the .gz is already there" must ask
 * this first.
 */
bool isValidGzipFile(const std::filesystem::path& path);

/**
 * Count durable terminal-loss markers in one session directory.
 *
 * Markers deliberately live outside `.tmp`: salvage may remove `.tmp` to
 * preserve liveness after an unrecoverable window, but the loss must remain
 * visible to a later uploader/agent and session-complete gate.
 */
std::size_t transportLossMarkerCount(
    const std::filesystem::path& session_dir);

/**
 * Return the next append-style window index for `channel` in a session.
 * Both published root files and unpublished `.tmp` staging files count, so
 * a failed publish cannot be overwritten by the next rotation.
 */
std::size_t nextLogWindowIndex(const std::filesystem::path& session_dir,
                               const std::string& channel);

/**
 * Remove oldest published windows once more than `max_files` exist, and
 * return how many were deleted. A nonzero return is DATA LOSS for any
 * window the agent had not uploaded yet - callers surface it loudly
 * (short rotation cadences reach the cap in minutes: 100 files at a 10 s
 * cadence is ~17 min of agent/backend outage tolerance).
 */
std::size_t pruneLogWindows(const std::filesystem::path& session_dir,
                            const std::string& channel,
                            std::size_t max_files);

/** Publish staged `.tmp/*.log.gz` files and export non-empty `.tmp/*.log`. */
LogSalvageResult salvageSessionTempDir(
    const std::filesystem::path& session_dir);

/**
 * Salvage one session while its writer-owned SessionOwnershipLock is held by
 * the caller. Only FileLogSink's clean-shutdown path may use this bypass.
 */
LogSalvageResult salvageOwnedSessionTempDir(
    const std::filesystem::path& session_dir);

/** Apply salvageSessionTempDir() to each session directory under `root`. */
LogSalvageResult salvageSessionTempDirs(
    const std::filesystem::path& root);

/** True when a session `.tmp` directory still holds uploadable data. */
bool sessionTempDirHasDeferredData(
    const std::filesystem::path& session_dir);

}  // namespace gpufl

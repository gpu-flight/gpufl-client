#pragma once

// Deferred bulk upload of a profiling session's NDJSON logs.
//
// Replaces the per-batch HttpLogSink streaming model. The C++ client
// writes everything to local NDJSON files during the session (always —
// regardless of upload mode); when the user wants to ship data to the
// backend, they call uploadLogs() AFTER gpufl::shutdown() has returned.
//
// Key property: this function never runs while the main GPU workload is
// active. All network I/O happens post-shutdown, so transient cert
// errors, TLS failures, and backend timeouts cannot affect the host
// process's exit code or perceived performance.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gpufl {

/**
 * Configuration for a single uploadLogs() invocation.
 *
 * All fields are independent. Callers typically populate at minimum
 * `log_path`, `backend_url`, and `api_key`; the rest have sensible
 * defaults tuned for "dev mode" — short total budget, one quick retry,
 * fail loud on auth errors.
 */
struct UploadOptions {
    /**
     * Same shape as `InitOptions::log_path`. Both the directory AND
     * filename-prefix of the session's NDJSON output. For example,
     * passing "/tmp/runs/pytorch_demo" causes uploadLogs to look for
     *
     *   /tmp/runs/pytorch_demo.device.log         (active)
     *   /tmp/runs/pytorch_demo.scope.log
     *   /tmp/runs/pytorch_demo.system.log
     *   /tmp/runs/pytorch_demo.device.1.log.gz   (rotated, oldest = highest N)
     *   /tmp/runs/pytorch_demo.scope.1.log.gz
     *   …
     *
     * If the path ends in ".log" the suffix is stripped before glob
     * (matches LogFileRotator::basePrefix() semantics).
     */
    std::string log_path;

    /** Backend host, e.g. "https://api.gpuflight.com". No trailing slash. */
    std::string backend_url;

    /** Bearer token sent as `Authorization: Bearer <api_key>`. */
    std::string api_key;

    /**
     * Reverse-proxy path mount. Empty resolves to "/api/v1". The final
     * POST URL is `{backend_url}{api_path}/events/{eventType}`.
     */
    std::string api_path;

    /**
     * Total wall-clock budget for the entire upload. If the budget
     * expires mid-stream, uploadLogs returns with success=false and a
     * warning. Default 5 minutes — generous for ~500 MB sessions on a
     * typical home connection, short enough that an unreachable backend
     * doesn't hang the caller indefinitely.
     */
    int total_timeout_ms = 5 * 60 * 1000;

    /** Per-POST connect timeout. */
    int connect_timeout_ms = 10 * 1000;

    /** Per-POST read timeout. */
    int read_timeout_ms = 30 * 1000;

    /**
     * Number of retries per failing POST. Plan calls for "one quick
     * retry per POST, then give up" — that catches transient flakes
     * without amplifying a sustained outage into a retry storm.
     * Default 1.
     */
    int max_retries = 1;

    /** Delay before the retry, in milliseconds. */
    int retry_delay_ms = 1000;

    /**
     * Filename of the cursor file written into the log directory.
     * Tracks (a) rotated files that have been successfully uploaded so
     * a re-run resumes rather than re-sending, and (b) the set of
     * `session_id`s that have completed a successful upload at least
     * once — used by the `force` flag below to refuse accidental
     * re-uploads.
     *
     * Schema is auto-migrated: a v1 cursor file (just `uploaded_files`)
     * is read as v2 with empty `completed_sessions`; the next write
     * upgrades it.
     */
    std::string cursor_filename = ".gpufl-upload-cursor.json";

    // ── Session selection ───────────────────────────────────────────────
    //
    // Default behavior (both fields below at their defaults): find every
    // `job_start` event across the discovered files, pick the one with
    // the highest `ts_ns`, upload only that session. "Upload" = "upload
    // what I just did."
    //
    // The two fields below override that:

    /**
     * Upload only the session whose `session_id` matches this value.
     * Empty (default) means "no explicit filter — fall through to
     * latest-by-default or all_sessions". If non-empty AND the session
     * isn't found in any discovered file, uploadLogs() returns
     * success=false with a warning. Mutually exclusive with
     * `all_sessions` — setting both is rejected with success=false.
     */
    std::string session_id_filter;

    /**
     * Upload every distinct session found in the discovered files
     * (the pre-1.0.4 behavior). Each session's events are streamed
     * with its own per-session lifecycle order: job_start(A) →
     * batches(A) → shutdown(A) → job_start(B) → … . Sessions are
     * processed oldest-first by `job_start.ts_ns`.
     *
     * Sessions already listed in the cursor's `completed_sessions`
     * map are silently skipped (unless `force=true`) — batch
     * semantics imply "do what's still pending."
     *
     * Mutually exclusive with `session_id_filter`.
     */
    bool all_sessions = false;

    /**
     * Re-upload sessions even when the cursor's `completed_sessions`
     * map says they've already shipped. Without this, the default
     * single-session modes (latest-by-default and session_id_filter)
     * refuse with a clear "already uploaded" message and exit code 2,
     * and all_sessions silently skips completed ones.
     */
    bool force = false;

    /**
     * If true, log a progress line every progress_log_interval_ms or
     * progress_log_interval_bytes (whichever comes first). Disabled by
     * setting to false (useful for the CLI's --quiet mode).
     */
    bool report_progress = true;
    int progress_log_interval_ms = 5 * 1000;
    std::size_t progress_log_interval_bytes = 50 * 1024 * 1024;  // 50 MB
};

/**
 * Outcome of an uploadLogs() call. Inspect `.success` first; on failure
 * `.warnings` lists every per-POST / per-file error encountered.
 */
struct UploadResult {
    /**
     * True iff every event from every discovered file was acknowledged
     * by the backend with a 2xx response inside the total wall budget.
     * False on:
     *   - any 401/403 (auth error) — short-circuits remaining work
     *   - total_timeout_ms exhausted
     *   - log_path's directory missing
     *   - cursor file unreadable AND unrepairable
     *   - any unrecoverable per-POST failure after max_retries
     */
    bool success = false;

    /** Count of distinct .log / .log.gz files processed (or skipped via cursor). */
    std::size_t files_processed = 0;

    /** Count of files that were skipped because the cursor said so. */
    std::size_t files_skipped_by_cursor = 0;

    /** Number of NDJSON events POSTed successfully (2xx). */
    std::size_t events_uploaded = 0;

    /** Sum of NDJSON-line bytes successfully uploaded (uncompressed). */
    std::size_t bytes_uploaded = 0;

    /** Wall time of the upload in milliseconds. */
    long long elapsed_ms = 0;

    /**
     * One entry per per-POST/per-file failure. Each line is plain
     * English suitable for surfacing in a CLI or log. Empty on a clean
     * success.
     */
    std::vector<std::string> warnings;
};

/**
 * Synchronous bulk upload. Never throws — all errors flow through
 * `UploadResult.warnings` and the `success` flag. Safe to call from any
 * thread, but typically called from the caller's main thread after
 * `gpufl::shutdown()` has returned.
 *
 * Session selection (see UploadOptions for details):
 *   - default: upload only the latest session found in the files
 *     (highest job_start.ts_ns).
 *   - session_id_filter set: upload only that session.
 *   - all_sessions=true: upload every session, oldest-first.
 *
 * Lifecycle ordering is enforced **per session**: for each selected
 * session, `job_start` is POSTed first, all batch events in file
 * order, then `shutdown` last — before any subsequent session begins.
 * This keeps the backend's session-lifecycle bookkeeping consistent
 * even when multiple sessions share a log directory.
 *
 * Files visited in (channel, rotation-index DESC, active) order so
 * per-channel chronological order is preserved within each session.
 *
 * Cursor file:
 *   - Tracks rotated `.log.gz` files that have shipped (skip on
 *     re-run); active `.log` files are always re-streamed.
 *   - Tracks `session_id`s that have completed a successful upload.
 *     Default and session_id_filter modes refuse to re-upload a
 *     completed session unless `force=true`. all_sessions mode
 *     silently skips completed sessions unless `force=true`.
 *
 * Local NDJSON files are NEVER deleted by this call.
 */
UploadResult uploadLogs(const UploadOptions& opts);

}  // namespace gpufl

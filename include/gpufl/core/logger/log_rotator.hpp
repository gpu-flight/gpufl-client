#pragma once

#include <cstddef>
#include <string>

#include "gpufl/core/logger/file_compressor.hpp"
#include "gpufl/core/logger/window_metadata.hpp"

namespace gpufl {

struct LogRotationOptions {
    /**
     * Parent directory for all session subdirectories. Trailing ".log"
     * (if present, e.g. when a caller passed a legacy-style log_path)
     * is stripped - base_path is treated as a directory, not a prefix.
     */
    std::string base_path;
    /**
     * Session ID. The active file lives at
     * `<base_path>/<session_id>/<channel_name>.log`; rotated files at
     * `<base_path>/<session_id>/<channel_name>.<N>.log[.gz]`. Required
     * on v1.2+; an empty session_id is treated as a configuration
     * error (the rotator falls back to writing directly under
     * base_path, which is the pre-v1.2 layout the uploader rejects).
     */
    std::string session_id;
    std::string channel_name;
    bool compress_rotated = true;
};

class LogFileRotator {
   public:
    /**
     * What actually happened to the window. Callers must branch on this -
     * treating every rotate() as a success once double-counted stats,
     * reset the window age on a DEFERRED export (delaying the retry by a
     * full cadence), and hid real publish failures.
     */
    enum class ExportWindowResult {
        NoData,            // active file empty/missing - nothing exported
        Published,         // window is a finished file in the session root
        StagedForSalvage,  // exported into .tmp staging; publish rename
                           // failed - the salvage pass finishes it. The
                           // active file WAS truncated: a new window began.
        DeferredInActive,  // compress/truncate failed - the data is still
                           // the active file's current window. Retry later.
    };

    LogFileRotator(LogRotationOptions opt, IFileCompressor* compressor);

    /** Outcome of the fast cutover half of a rotation. */
    enum class RetireResult {
        NoData,   // active file empty/missing - no window to retire
        Retired,  // active window is now an immutable `.tmp/<ch>.<N>.log`
        Blocked,  // rename denied (a holder) - data stays the active window
    };

    /**
     * The active file lives in the session's TEMP subdir
     * (`<session>/.tmp/<channel>.log`), never in the session root. The
     * root only ever receives FINISHED `.gz` files, exported atomically -
     * so consumers (uploader, analyzer, UI) can read the session at any
     * time without seeing half-written or stale-duplicate files, and an
     * un-deletable leftover is an obviously-garbage `.tmp` dir instead of
     * a data file that shadows real content.
     */
    [[nodiscard]] std::string activePath() const;

    /**
     * Export the current window synchronously for shutdown: retire the
     * active file to an indexed raw window, compress through an unrecognized
     * `.part` artifact, atomically promote the completed gzip, remove or
     * truncate the raw authority, then publish. This is the same crash-safe
     * transaction used by the asynchronous mid-run path.
     *
     * SYNCHRONOUS - compression and publish retries happen on the calling
     * thread. Used by the shutdown path only. Mid-run rotation splits this
     * into retireActiveWindow() + exportRetiredWindow() so no collector or
     * writer thread ever waits on gzip.
     */
    ExportWindowResult rotate() const;

    /** Next append-style window index for this channel (root + `.tmp`). */
    [[nodiscard]] std::size_t nextWindowIndex() const;

    /**
     * Fast half of a rotation: rename the active file to an immutable
     * `.tmp/<channel>.<index>.log`. Metadata-only, so it is safe to call
     * under the channel lock from the collector beat or a writer.
     *
     * A retired file is durable and self-describing: if the process dies
     * before it is exported, the salvage pass compresses and publishes it
     * (log_salvage.cpp handles plain indexed windows in `.tmp`).
     *
     * `Blocked` (a holder denied the rename) leaves the data as the
     * current active window - the caller keeps the window's age so the
     * next beat retries rather than waiting out a fresh cadence.
     */
    RetireResult retireActiveWindow(std::size_t index) const;

    /**
     * Slow half: compress the retired window and publish it into the
     * session root, then prune. Touches only files nothing writes to any
     * more, so it runs off the caller's thread with no lock held; the
     * publish retry/backoff for transient holders lives here.
     */
    ExportWindowResult exportRetiredWindow(std::size_t index,
                                           const WindowTiming& timing = {}) const;

    /**
     * Finalize this channel on clean shutdown (FileLogSink::close →
     * Logger::close → gpufl::shutdown): export the last window like
     * rotate() and drop the channel's active file. The shared `.tmp` dir
     * is removed separately via removeTempDir() once all channels closed.
     *
     * On crash (shutdown never runs) the active `.log` stays in `.tmp` -
     * the launcher repair / uploader salvage exports it on first sight.
     */
    /**
     * Finalize the active window using the index owned by FileChannel.
     * The shutdown path must not re-scan the filesystem: published files may
     * already have been acknowledged and removed, but their indices remain
     * consumed for the lifetime of the session.
     */
    ExportWindowResult compressActive(
        std::size_t index, const WindowTiming& timing = {}) const;

    /**
     * The session temp dir: `<base_path>/<session_id>/.tmp`. Removed
     * wholesale by FileLogSink::close() after every channel finalized.
     */
    [[nodiscard]] std::string tempDir() const;

   private:
    /**
     * The session directory: `<base_path>/<session_id>`. Created by
     * the FileLogSink before opening any channel's stream.
     */
    [[nodiscard]] std::string sessionDir() const;
    [[nodiscard]] std::string rotatedPath(std::size_t index) const;
    /** A retired-but-not-yet-exported window: `.tmp/<channel>.<N>.log`. */
    [[nodiscard]] std::string retiredPath(std::size_t index) const;
    /** staging → session root, with backoff for transient holders. */
    [[nodiscard]] bool publishWithRetry_(const std::string& from,
                                         const std::string& to) const;

    /** Shared body of rotate()/compressActive(). */
    ExportWindowResult exportWindow_() const;
    ExportWindowResult exportWindowAt_(std::size_t index) const;

    LogRotationOptions opt_;
    IFileCompressor* compressor_ = nullptr;  // non-owning
};

}  // namespace gpufl

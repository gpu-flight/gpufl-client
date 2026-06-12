#pragma once

#include <cstddef>
#include <string>

#include "gpufl/core/logger/file_compressor.hpp"

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
    std::size_t max_files = 100;
    bool compress_rotated = true;
};

class LogFileRotator {
   public:
    LogFileRotator(LogRotationOptions opt, IFileCompressor* compressor);

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
     * Export the current window. The active file is never renamed or
     * deleted - operations on it are limited to READ (gzip) and TRUNCATE,
     * which work even while another process holds the file:
     *   1. gzip active → `.tmp/<channel>.<N>.log.gz` (staging, pure read)
     *   2. truncate active (restart the window)
     *   3. move staging → `<session>/<channel>.<N>.log.gz` (fresh name at
     *      a monotonic index - no shifting, no overwrite hazard)
     * Any failed step logs and defers: data stays exactly-once (in the
     * active file or in staging, both inside `.tmp`, which consumers
     * ignore), and the next write or the launcher's salvage pass retries.
     */
    void rotate() const;

    /**
     * Finalize this channel on clean shutdown (FileLogSink::close →
     * Logger::close → gpufl::shutdown): export the last window like
     * rotate() and drop the channel's active file. The shared `.tmp` dir
     * is removed separately via removeTempDir() once all channels closed.
     *
     * On crash (shutdown never runs) the active `.log` stays in `.tmp` -
     * the launcher repair / uploader salvage exports it on first sight.
     */
    void compressActive() const;

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

    enum class ExportWindowResult {
        NoData,
        Published,
        StagedForSalvage,
        DeferredInActive,
    };

    /** Shared body of rotate()/compressActive(). */
    ExportWindowResult exportWindow_() const;

    LogRotationOptions opt_;
    IFileCompressor* compressor_ = nullptr;  // non-owning
};

}  // namespace gpufl

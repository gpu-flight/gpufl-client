#pragma once

#include <cstddef>
#include <string>

#include "gpufl/core/logger/file_compressor.hpp"

namespace gpufl {

struct LogRotationOptions {
    /**
     * Parent directory for all session subdirectories. Trailing ".log"
     * (if present, e.g. when a caller passed a legacy-style log_path)
     * is stripped — base_path is treated as a directory, not a prefix.
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

    [[nodiscard]] std::string activePath() const;
    void rotate()const;

    /**
     * Compress the active `.log` file in place to `.log.gz` and remove
     * the original. Called on clean shutdown (FileLogSink::close →
     * Logger::close → gpufl::shutdown) so a finished session leaves
     * behind only compressed files. No-op when:
     *   - compress_rotated is false (caller opted out)
     *   - the active file doesn't exist (never opened)
     *   - a compressor isn't configured (defense)
     *
     * On crash (i.e. shutdown never runs), the uncompressed .log is
     * left intact — the uploader lazily compresses on first read so
     * the on-wire format is uniform.
     */
    void compressActive() const;

   private:
    /**
     * The session directory: `<base_path>/<session_id>`. Created by
     * the FileLogSink before opening any channel's stream.
     */
    [[nodiscard]] std::string sessionDir() const;
    [[nodiscard]] std::string rotatedPath(std::size_t index) const;

    LogRotationOptions opt_;
    IFileCompressor* compressor_ = nullptr;  // non-owning
};

}  // namespace gpufl

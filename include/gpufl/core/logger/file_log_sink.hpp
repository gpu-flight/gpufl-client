#pragma once

#include <cstddef>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>

#include "gpufl/core/logger/log_sink.hpp"
#include "gpufl/core/logger/logger.hpp"

namespace gpufl {

class IFileCompressor;
class LogFileRotator;

/**
 * Sink that writes NDJSON lines to per-channel files on disk.
 *
 * Behavior is preserved bit-for-bit from the pre-refactor Logger's
 * internal LogChannel: three separate std::ofstream per channel
 * (Device / Scope / System), line-level flush after every write,
 * size-based rotation with optional gzip compression of rotated files.
 *
 * This sink is the default for every session. It's also the durable
 * fallback if other sinks (e.g. HttpLogSink) drop lines due to
 * network pressure — the data is still on disk for a monitor daemon
 * to ship later.
 */
class FileLogSink final : public ILogSink {
   public:
    /**
     * Construct a file sink using the Logger::Options already parsed
     * by the caller. Passing `Options` (instead of the narrower set
     * of fields this sink needs) lets us keep the existing
     * configuration surface unchanged.
     */
    explicit FileLogSink(const Logger::Options& opt);
    ~FileLogSink() override;

    FileLogSink(const FileLogSink&) = delete;
    FileLogSink& operator=(const FileLogSink&) = delete;

    void write(Channel ch, std::string_view json) override;
    void close() override;

   private:
    // One stream per channel, matching the existing file layout.
    class FileChannel {
       public:
        FileChannel(std::string name, Logger::Options opt);
        ~FileChannel();

        void write(std::string_view line);
        void close();
        bool isOpen() const;

       private:
        void ensureOpenLocked();
        void rotateLocked();
        void closeLocked();

        std::string name_;
        Logger::Options opt_;
        std::unique_ptr<IFileCompressor> compressor_;
        std::unique_ptr<LogFileRotator> rotator_;

        std::ofstream stream_;
        size_t current_bytes_ = 0;

        mutable std::mutex mu_;
        bool opened_ = false;
    };

    FileChannel* resolveChannel(Channel ch) const;

    std::unique_ptr<FileChannel> chanDevice_;
    std::unique_ptr<FileChannel> chanScope_;
    std::unique_ptr<FileChannel> chanSystem_;
};

}  // namespace gpufl

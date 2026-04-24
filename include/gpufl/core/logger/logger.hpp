#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "gpufl/core/model/serializable.hpp"

namespace gpufl {

class ILogSink;

/**
 * Central log dispatcher — every event emitted via the runtime goes
 * through Logger::write(), which broadcasts the serialized NDJSON
 * line to every registered {@link ILogSink}.
 *
 * The default session uses a single {@link FileLogSink} (preserves
 * the original per-channel file layout on disk). Additional sinks
 * (e.g. {@link HttpLogSink} for direct-to-backend upload) can be
 * attached via {@link addSink}.
 *
 * Thread safety: {@link write} is safe to call from any thread.
 * The sink vector itself is only modified from the init() / shutdown()
 * path (no concurrent addSink during writes in practice) and is
 * locked with a mutex on reads anyway.
 */
class Logger {
   public:
    struct Options {
        std::string base_path;
        std::size_t rotate_bytes = 64 * 1024 * 1024;  // 64 MiB default
        std::size_t max_files = 100;
        bool compress_rotated = true;
        bool flush_always = false;
        int system_sample_rate_ms = 0;
    };

    Logger();
    ~Logger();

    /**
     * Initialize the logger with the default FileLogSink attached.
     * Returns false if the base_path is empty (the file sink cannot
     * open without a path).
     */
    bool open(const Options& opt);

    /**
     * Close and release all attached sinks. Safe to call multiple
     * times; safe to call on an already-closed logger.
     */
    void close();

    /**
     * Attach an additional sink. Ownership transfers to the Logger.
     * Call before the first write() to avoid lost lines — sinks added
     * after events have already flowed will not back-fill.
     */
    void addSink(std::unique_ptr<ILogSink> sink);

    /**
     * Serialize the model and broadcast to all sinks on the model's
     * declared channel(s).
     */
    void write(const IJsonSerializable& model);

   private:
    Options opt_;
    std::vector<std::unique_ptr<ILogSink>> sinks_;
    mutable std::mutex sinks_mu_;
};

}  // namespace gpufl

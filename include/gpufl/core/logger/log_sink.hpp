#pragma once

#include <string_view>

#include "gpufl/core/model/serializable.hpp"

namespace gpufl {

/**
 * Abstract destination for log lines emitted by {@link Logger}.
 *
 * The Logger holds a vector of sinks and broadcasts every write to all
 * of them. Each sink decides what to do with the line — write it to
 * disk, POST it to a backend, put it on a message bus, etc.
 *
 * Ownership: Logger owns sinks via std::unique_ptr. A sink MUST be
 * safe to destroy after close() has returned.
 *
 * Thread-safety: Logger serializes calls to write() at the Channel
 * level (per-channel mutex) today, but a sink should not assume this
 * — newer broadcast paths may parallelize. Implement write() to be
 * re-entrant for different channel arguments. The default contract:
 * write() may be called concurrently from multiple threads; flush()
 * and close() are only called from Logger::close() (single-threaded
 * teardown).
 */
class ILogSink {
   public:
    virtual ~ILogSink() = default;

    /**
     * Deliver a single NDJSON line on the given channel.
     *
     * @param ch    Channel the line belongs to. Sinks that ignore
     *              channels (e.g. a network sink that fans everything
     *              to one endpoint) can treat this as informational.
     * @param json  Serialized event. MAY NOT include a trailing newline
     *              (the sink is responsible for whatever framing it
     *              needs — file sinks append '\n', HTTP sinks put the
     *              bytes directly into a POST body).
     */
    virtual void write(Channel ch, std::string_view json) = 0;

    /**
     * Called from Logger::close() before destruction. Implementations
     * should flush any pending in-memory buffers, wait a bounded amount
     * of time for background workers to drain, and release resources.
     * Must not throw.
     */
    virtual void close() = 0;
};

}  // namespace gpufl

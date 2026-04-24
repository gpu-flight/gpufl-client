#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>

#include "gpufl/core/logger/log_sink.hpp"

namespace gpufl {

/**
 * Sink that POSTs NDJSON lines to the GPUFlight backend ingestion API.
 *
 * The sink's write() call is non-blocking from the producer's
 * perspective — it enqueues the line into a bounded in-memory FIFO
 * and returns immediately. A dedicated worker thread drains the
 * queue and POSTs each line to
 * {@code <base_url>/api/v1/events/<type>} with
 * {@code Authorization: Bearer <api_key>} header.
 *
 * Design goals:
 *   - Zero producer-thread blocking: the hot path must never wait on
 *     network.
 *   - Bounded memory: if the backend is slow, the queue fills up to
 *     {@code queue_capacity} and then starts dropping the OLDEST
 *     line per arrival (with a deduped GFL_LOG_ERROR warning).
 *   - Best-effort delivery: up to 3 retries with exponential backoff
 *     (50ms, 200ms, 1s) per line, then drop and move on. The NDJSON
 *     is also on disk via FileLogSink, so persistent failures don't
 *     cause data loss — they just prevent live upload.
 *   - Auth-failure short-circuit: a 401/403 response disables the
 *     sink for the rest of the session (API key is invalid; retries
 *     won't help). File sink continues normally.
 *
 * Not thread-safe for concurrent open/close. Safe to call
 * write() from many producer threads — the queue mutex guards pushes.
 */
class HttpLogSink final : public ILogSink {
   public:
    struct Options {
        /**
         * Backend base URL, e.g. "https://api.gpuflight.com" or
         * "http://localhost:8080". A trailing slash is tolerated
         * but stripped. The sink appends "/api/v1/events/<type>" to
         * form the target for each line.
         */
        std::string base_url;

        /**
         * API key used in {@code Authorization: Bearer <api_key>}.
         * Must match an active workspace key on the backend.
         */
        std::string api_key;

        /** Bounded in-memory queue depth. Overflow drops oldest. */
        std::size_t queue_capacity = 4096;

        /** Per-request connect + read timeout. */
        int connect_timeout_ms = 5000;
        int read_timeout_ms    = 10000;

        /** Retry budget per line (0 = no retries). */
        int max_retries = 3;

        /** Shutdown drain grace period. Logger::close() waits up to
         *  this long for the worker to drain the queue before it is
         *  forcibly stopped. */
        int shutdown_drain_ms = 5000;
    };

    explicit HttpLogSink(Options opts);
    ~HttpLogSink() override;

    HttpLogSink(const HttpLogSink&) = delete;
    HttpLogSink& operator=(const HttpLogSink&) = delete;

    /** Non-blocking: push the line into the upload queue. */
    void write(Channel ch, std::string_view json) override;

    /** Stop accepting new writes, drain for up to shutdown_drain_ms,
     *  then join the worker thread. Safe to call multiple times. */
    void close() override;

    // --- Diagnostics (used by tests + debug logging) ---

    /** Total lines enqueued since construction. */
    std::size_t enqueuedCount() const {
        return enqueued_.load(std::memory_order_relaxed);
    }
    /** Total lines dropped due to queue overflow. */
    std::size_t droppedCount() const {
        return dropped_.load(std::memory_order_relaxed);
    }
    /** Total lines successfully POSTed (2xx). */
    std::size_t uploadedCount() const {
        return uploaded_.load(std::memory_order_relaxed);
    }
    /** Total lines that exhausted retries and were abandoned. */
    std::size_t failedCount() const {
        return failed_.load(std::memory_order_relaxed);
    }

   private:
    void workerLoop();
    bool postOnce(std::string_view event_type, std::string_view body);
    static std::string_view extractType(std::string_view json);

    Options opts_;
    std::string host_;       // e.g. "api.gpuflight.com"
    std::string scheme_;     // "http" or "https"
    int port_ = -1;          // -1 = default per scheme
    bool ssl_supported_ = false;

    // Bounded producer-consumer queue. std::deque + mutex + condvar is
    // simpler than a lock-free ring and performs well for our volume
    // (a few hundred events per scope at most); replace with a
    // lock-free SPSC queue if profiling shows this is a bottleneck.
    mutable std::mutex         queue_mu_;
    std::condition_variable    queue_cv_;
    std::deque<std::string>    queue_;

    std::atomic<bool>          auth_failed_{false};
    std::atomic<bool>          running_{false};
    std::atomic<bool>          closing_{false};
    std::thread                worker_;

    // Observability counters.
    std::atomic<std::size_t>   enqueued_{0};
    std::atomic<std::size_t>   dropped_{0};
    std::atomic<std::size_t>   uploaded_{0};
    std::atomic<std::size_t>   failed_{0};

    // Deduped-warning state so we don't spam the log when the queue
    // is persistently full or auth keeps failing.
    std::atomic<bool>          overflow_warned_{false};
    std::atomic<bool>          auth_warned_{false};
};

}  // namespace gpufl

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "gpufl/core/logger/log_sink.hpp"
#include "gpufl/core/logger/logger.hpp"
#include "gpufl/core/logger/window_metadata.hpp"

namespace gpufl {

class IFileCompressor;
class LogFileRotator;
class SessionOwnershipLock;

/**
 * Sink that writes NDJSON lines to per-channel files on disk.
 *
 * Behavior is preserved bit-for-bit from the pre-refactor Logger's
 * internal LogChannel: three separate std::ofstream per channel
 * (Device / Scope / System), line-level flush after every write,
 * size-based rotation with optional gzip compression of rotated files.
 *
 * This sink is the default for every session. The NDJSON files it
 * writes are the source of truth for both `gpufl::uploadLogs()` (the
 * in-process deferred upload path) and the standalone `gpufl-agent`
 * JVM service (production fleet sync). Data is always recoverable
 * from these files even if upload fails.
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
    void setSerializedBytesCallbackBeforeFirstWrite(
        std::function<void(std::uint64_t)> callback) override;
    void close() override;

    /**
     * True if at least one of the per-channel files actually opened
     * successfully. Used by Logger::open() to surface filesystem
     * failures (permission denied on a Docker volume mount with
     * stale UID ownership is the canonical case) up through
     * gpufl::init() rather than letting init silently "succeed" with
     * a sink that drops every write.
     */
    bool anyChannelOpen() const;

    /**
     * Rotation outcomes summed across channels. An IN-MEMORY test and
     * measurement seam - it is NOT durable and is not exported anywhere;
     * files alone can't say WHY a window was published, and failed
     * exports leave no file at all.
     *
     * by_size / by_time count durable CUTOVERS keyed by which trigger fired
     * first. published / staged / export_failed describe the later worker
     * outcome. Spool saturation is terminal: new events are dropped only
     * after a durable marker makes the incomplete session visible.
     */
    struct RotationStats {
        // Windows CUT OVER by each trigger. A cutover is the boundary
        // itself: the window is immutable and durable in `.tmp` from
        // this moment, whether or not the export has run yet.
        std::size_t by_size = 0;
        std::size_t by_time = 0;
        // Cutover blocked (a holder denied the rename): the data is still
        // the active window and its age is preserved, so the next beat
        // retries instead of waiting out a fresh cadence.
        std::size_t cutover_blocked = 0;
        // Export outcomes, counted on the retirement worker.
        std::size_t published = 0;      // finished file in the session root
        std::size_t staged = 0;         // compressed, publish blocked
        std::size_t export_failed = 0;  // compression failed
        // Slowest single export, for sizing the compression cost that the
        // worker now absorbs instead of the collector.
        std::int64_t max_export_ms = 0;
        // Retirement-worker backlog: immutable raw windows still waiting for
        // gzip/publish.
        std::size_t pending_exports = 0;
        std::size_t max_pending_exports = 0;
        std::uint64_t pending_export_bytes = 0;
        std::uint64_t max_pending_export_bytes = 0;
        // Windows whose events are GONE - discarded at close() because
        // nothing on disk could still yield them. Terminal, unlike
        // `staged`/`export_failed`, which salvage still recovers.
        std::size_t lost_windows = 0;
        bool spool_saturated = false;
        std::uint64_t spool_bytes_at_saturation = 0;
        std::uint64_t dropped_events = 0;
        std::uint64_t dropped_bytes = 0;
    };
    RotationStats rotationStats() const;

    /**
     * Block until every retired window has been exported. Called at
     * close() before the temp dir is swept, and by tests that need the
     * asynchronous export to have landed.
     */
    void waitForPendingExports();

    /**
     * Rotate every channel whose non-empty window is older than
     * rotate_after_ms RIGHT NOW - the deadline path for channels that
     * went quiet. Called from the collector's periodic beat; without
     * it the time trigger only ever fires on the next write, and a
     * channel that wrote once then stalled would sit in `.tmp`
     * indefinitely. No-op for empty windows (idle channels never
     * publish empty files) and when the time trigger is off.
     */
    void rotateDueWindows() override;

   private:
    // One stream per channel, matching the existing file layout.
    class FileChannel {
       public:
        FileChannel(std::string name, Logger::Options opt, FileLogSink* owner);
        ~FileChannel();

        bool write(std::string_view line);
        void close();
        bool isOpen() const;
        std::uint64_t currentBytes() const;
        RotationStats rotationStats() const;
        void rotateIfDue();
        /**
         * Compress + publish a window this channel already retired. Runs
         * on the retirement worker with NO channel lock held - it only
         * touches files nothing writes to any more - and takes the lock
         * just to fold the outcome into the stats.
         */
        void exportRetired(std::size_t index, WindowTiming timing);

       private:
        void ensureOpenLocked();
        enum class RotateTrigger { Size, Time };
        void rotateLocked(RotateTrigger trigger);
        bool timeDueLocked(std::int64_t now) const;
        void closeLocked();
        std::int64_t nowMs() const;

        std::string name_;
        Logger::Options opt_;
        FileLogSink* owner_ = nullptr;  // non-owning; outlives the channel
        std::unique_ptr<IFileCompressor> compressor_;
        std::unique_ptr<LogFileRotator> rotator_;

        std::ofstream stream_;
        size_t current_bytes_ = 0;
        // Next window index to hand out, seeded ONCE from the filesystem and
        // then owned by this channel. It must not be re-derived per rotation:
        // nextWindowIndex() scans `.tmp` and the session root, and the
        // retirement worker moves files between exactly those two directories
        // with no lock held, so a window in flight can be missed by both
        // scans - handing its index out twice, and fs::rename replaces its
        // destination silently. 0 = not seeded yet.
        std::size_t next_window_index_ = 0;
        // Monotonic time of the current window's FIRST write; -1 = the
        // window is empty. Recorded for every non-empty window so its
        // immutable metadata has real timing even when time rotation is
        // disabled. It also drives rotate_after_ms: an empty window has no
        // age, so it can never rotate.
        std::int64_t window_first_write_ms_ = -1;
        RotationStats rotation_stats_;

        mutable std::mutex mu_;
        bool opened_ = false;
    };

    FileChannel* resolveChannel(Channel ch) const;

    /**
     * Hand a retired window to the export worker. Called from a channel
     * with its lock held, so it must not block: it appends and notifies.
     * Starts the worker on first use, so sessions that never rotate never
     * spawn a thread.
     */
    void enqueueRetired(FileChannel* channel, std::size_t index,
                        std::uint64_t bytes, WindowTiming timing);
    void stopRetirementWorker();
    void checkSpoolBudget(bool force = false,
                          std::uint64_t incoming_bytes = 0);
    void markSpoolSaturated(std::uint64_t spool_bytes,
                            std::uint64_t available_bytes,
                            const char* reason);
    std::uint64_t spoolBytesOnDisk() const;
    std::int64_t spoolNowMs() const;

    std::unique_ptr<FileChannel> chanDevice_;
    std::unique_ptr<FileChannel> chanScope_;
    std::unique_ptr<FileChannel> chanSystem_;
    std::unique_ptr<FileChannel> chanSass_;
    // Held for the complete writer lifetime. Root windows remain readable to
    // the agent, but no other process may salvage `.tmp` or finalize this
    // session until the handle is released.
    std::unique_ptr<SessionOwnershipLock> session_ownership_;
    // The shared session `.tmp` dir, removed once in close() after every
    // channel has finalized (its own actives would block earlier removal).
    std::string temp_dir_;
    std::filesystem::path session_dir_;
    std::uint64_t max_spool_bytes_ = 0;
    std::uint64_t min_free_bytes_ = 0;
    std::function<std::int64_t()> spool_now_ms_;
    std::function<void(std::uint64_t bytes)> on_serialized_bytes_;
    mutable std::mutex spool_budget_mu_;
    std::int64_t last_spool_check_ms_ = -1;
    std::atomic<std::uint64_t> spool_estimated_bytes_{0};
    // Conservative free-space estimate from the last fs::space() scan,
    // decremented by accepted writes. Agent deletions can only make the real
    // value larger; approaching the reserve forces a fresh scan.
    std::atomic<std::uint64_t> available_bytes_estimate_{
        std::numeric_limits<std::uint64_t>::max()};
    std::atomic<bool> spool_saturated_{false};
    std::atomic<std::uint64_t> spool_bytes_at_saturation_{0};
    std::atomic<std::uint64_t> dropped_events_{0};
    std::atomic<std::uint64_t> dropped_bytes_{0};

    // Retirement queue: cutover happens on whichever thread hit the
    // boundary (fast, metadata-only), compression and publish retries
    // happen here. Without this split a 64 MiB gzip - or 700 ms of
    // publish backoff - would run inside the CUPTI collector beat or a
    // writer, stalling ring drain and every other logger write.
    struct RetiredWindow {
        FileChannel* channel;
        std::size_t index;
        std::uint64_t bytes;
        WindowTiming timing;
    };
    mutable std::mutex retire_mu_;
    std::condition_variable retire_cv_;
    std::deque<RetiredWindow> retire_queue_;
    std::size_t exports_in_flight_ = 0;
    std::size_t pending_exports_ = 0;
    std::size_t max_pending_exports_ = 0;
    std::uint64_t pending_export_bytes_ = 0;
    std::uint64_t max_pending_export_bytes_ = 0;
    // Sink-level like the pending_* counters (the channels are already gone
    // when close() learns this): written by close(), read by rotationStats().
    std::size_t lost_windows_ = 0;
    bool retire_stop_ = false;
    std::thread retire_worker_;
};

}  // namespace gpufl

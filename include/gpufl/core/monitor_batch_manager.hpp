#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "gpufl/core/batch_buffer.hpp"
#include "gpufl/core/dictionary_manager.hpp"
#include "gpufl/core/events.hpp"

namespace gpufl {

class Logger;

namespace detail {

/**
 * @brief Small imperative shell for monitor batching and dictionary state.
 */
class MonitorBatchManager {
public:
    enum class FlushMode { Fast, Full };

    void reset();
    void bindFlushSink(Logger* logger, std::string session_id);
    void clearFlushSink();
    void setSourceCollectionEnabled(bool enabled);
    void flushAll(FlushMode mode = FlushMode::Fast);

    uint32_t internKernel(const std::string& name);
    uint32_t internScopeName(const std::string& name);
    uint32_t internFunction(const std::string& name,
                            const std::string& func_symbol = std::string());
    uint32_t internMetric(const std::string& name);
    uint32_t internSourceFile(const std::string& path);

    void enqueueDisassembly(uint64_t crc, const uint8_t* data, size_t size);
    void flushDisassembly();

    uint64_t allocateScopeInstanceId();
    uint32_t activeScopeNameId() const;
    /** @brief How many scopes are open right now; the depth a new one nests at. */
    int openScopeDepth() const;
    /** @brief Capture and publish a close timestamp as one ordered operation. */
    int64_t captureScopeCloseTimestamp(uint64_t instance_id);
    /** @brief Publish an already-chosen close timestamp before state transition. */
    void markScopeClosePending(uint64_t instance_id, int64_t end_ns);

    /**
     * @brief Publish the point below which no future sample can be attributed.
     *
     * Contract: no subsequent SUCCESSFUL decode will return a sample with a
     * timestamp at or below @p ts_ns. Completed scopes that ended before it can
     * therefore never match again and are dropped.
     *
     * Deliberately NOT wall-clock. A stalled collector lets `now` run on while
     * undecoded samples still sit in the buffer, and trimming against it would
     * discard the scopes those samples need. It is also not "last decode time"
     * nor "oldest decoded timestamp" - both can move ahead of samples that are
     * still to come.
     *
     * Monotonic: a lower value is ignored. The caller must simply not advance it
     * when a decode fails or the buffer overflowed, since either means samples
     * were lost rather than delivered.
     */
    void publishScopeRetentionWatermark(int64_t ts_ns);

    /** @brief Mark the wall-clock boundary from which PM samples may be pending. */
    void beginPmScopeAttribution(int64_t start_ns);
    /** @brief Mark that the final PM decode completed. */
    void endPmScopeAttribution();
    /** @brief Completed scope history evicted while PM attribution was active. */
    uint64_t scopeAttributionTruncated() const;
    /** @brief PM metric rows that have passed through scope attribution. */
    uint64_t pmSampleRowsSeen() const;

    /** @brief Test seam: resolve a batch exactly as the drain path does. */
    void resolveScopeIdsForTesting(std::vector<PmSampleBatchRow>& rows, uint32_t fallback_id);
    /** @brief Test seam: the original per-sample resolver, kept as the reference
     *  the batch sweep is checked against. */
    uint32_t resolveScopeIdForTesting(int64_t ts_ns) const;
    /** @brief Test seam: completed scopes still retained. */
    size_t retainedCompletedScopesForTesting() const;

    bool pushKernel(const KernelBatchRow& row, const KernelDetailRow* detail = nullptr);
    bool pushMemcpy(const MemcpyBatchRow& row);
    void pushTraceScopeRows(const ScopeBatchRow& begin_row, const ScopeBatchRow& end_row);
    void pushTrackedScopeRow(const ScopeBatchRow& row);
    bool pushProfileSample(const ProfileSampleBatchRow& row);
    void pushProfileSamples(const std::vector<ProfileSampleBatchRow>& rows);
    void pushPmSamplesResolvingScopes(const std::vector<PmSampleBatchRow>& rows);
    bool pushMemoryAlloc(const MemoryAllocEventBatchRow& row);
    void pushSynchronization(const SynchronizationEventBatchRow& row);

private:
    struct FlushSink {
        Logger* logger = nullptr;
        std::string session_id;

        bool available() const { return logger != nullptr; }
    };

    struct ScopeWindow {
        int64_t start_ns = 0;
        int64_t end_ns = 0;
        uint64_t instance_id = 0;
        uint32_t name_id = 0;
        int depth = 0;

        /** @brief True when this scope should win over @p other for a sample
         *  both contain. Deepest, then latest start, then instance id. */
        bool outranks(const ScopeWindow& other) const;
    };

    struct OpenScopeWindow {
        int64_t start_ns = 0;
        uint32_t name_id = 0;
        int depth = 0;
    };

    uint32_t resolveScopeIdLocked(int64_t ts_ns) const;

    /**
     * @brief Attribute a whole batch of samples in one pass.
     *
     * Sorts the samples and a snapshot of the candidate scopes once, then
     * sweeps them together, rather than re-scanning every scope for every
     * sample. The snapshot includes scopes that are STILL OPEN, given a
     * provisional end: PM drains mid-run, so the scope covering a sample is
     * routinely still open when that sample is decoded.
     *
     * Candidates cannot be kept pre-sorted. A scope's close timestamp is taken
     * before PushScopeRow acquires the lock, so two threads closing at once
     * append out of order.
     *
     * Selection matches resolveScopeIdLocked exactly: the interval contains the
     * timestamp (both ends inclusive), then greatest depth, then latest start.
     */
    std::vector<ScopeWindow> snapshotScopeCandidatesLocked(
            const std::vector<PmSampleBatchRow>& rows) const;
    static void resolveScopeIdsForBatch(std::vector<ScopeWindow>& candidates,
                                        std::vector<PmSampleBatchRow>& rows,
                                        uint32_t fallback_id);

    /** @brief Drop completed scopes that can no longer match any future sample. */
    void trimCompletedScopesLocked();

    /**
     * @brief Apply the hard cap, counting whatever it drops.
     *
     * Called on every scope close, not only when PM samples arrive: the
     * retention watermark is the real bound, but only PM publishes one. A run
     * without PM sampling would otherwise never trim at all.
     */
    void enforceScopeCapLocked();
    void clearPendingScopeCloseLocked(uint64_t instance_id);

    FlushSink flushSink_;
    DictionaryManager dictManager_;

    BatchBuffer<KernelBatchRow> kernelBatch_;
    BatchBuffer<MemcpyBatchRow> memcpyBatch_;
    uint64_t kernelBatchId_ = 0;
    uint64_t memcpyBatchId_ = 0;
    std::vector<KernelDetailRow> pendingDetails_;

    BatchBuffer<ScopeBatchRow> scopeBatch_;
    BatchBuffer<ProfileSampleBatchRow> profileBatch_;
    BatchBuffer<PmSampleBatchRow> pmSampleBatch_;
    uint64_t scopeBatchId_ = 0;
    uint64_t profileBatchId_ = 0;
    uint64_t pmSampleBatchId_ = 0;
    mutable std::mutex scopeBatchMu_;
    std::atomic<uint64_t> nextScopeInstanceId_{1};
    // Cached top of scopeNameStack_. Read without the mutex on the sample hot
    // path, written only while holding it.
    std::atomic<uint32_t> activeScopeNameId_{0};
    // Scopes currently open, innermost last. A stack rather than a single
    // value because scopes nest: a deep window opens inside the process scope
    // and must hand the name back on close, or every sample after it keeps the
    // window's name. Entries carry their instance id so a close that is not
    // strictly LIFO - the collector can close a deep window while an
    // application scope is open - removes the right one.
    std::vector<std::pair<uint64_t, uint32_t>> scopeNameStack_;
    std::unordered_map<uint64_t, OpenScopeWindow> openScopeWindows_;
    // A close timestamp is captured before scopeBatchMu_ can be acquired.
    // Publishing it separately lets a concurrent PM snapshot cap an otherwise
    // still-open interval at its real end instead of extending it to the batch
    // watermark.
    mutable std::mutex pendingScopeCloseMu_;
    std::unordered_map<uint64_t, int64_t> pendingScopeCloseNs_;
    // Closed scopes still needed to attribute samples that have not been
    // decoded yet. Unordered - see snapshotScopeCandidatesLocked.
    std::deque<ScopeWindow> completedScopeWindows_;
    // Below this, no future sample can arrive; see
    // publishScopeRetentionWatermark. 0 = nothing published yet, so nothing is
    // dropped by the watermark. The hard cap still applies on every close.
    int64_t scopeRetentionWatermarkNs_ = 0;
    // Backstop only. The watermark is what should bound this; the cap exists so
    // a source that never advances it cannot grow the deque without limit.
    static constexpr size_t kMaxCompletedScopes = 65536;
    int64_t pmScopeAttributionStartNs_ = 0;
    bool scopeHistoryEvictionLogged_ = false;
    uint64_t scopeAttributionTruncated_ = 0;
    uint64_t pmSampleRowsSeen_ = 0;

    BatchBuffer<SynchronizationEventBatchRow> syncBatch_;
    BatchBuffer<MemoryAllocEventBatchRow> memAllocBatch_;
    uint64_t syncBatchId_ = 0;
    uint64_t memAllocBatchId_ = 0;
};

}  // namespace detail
}  // namespace gpufl

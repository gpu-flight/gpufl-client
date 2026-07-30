#include "gpufl/core/monitor_batch_manager.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <queue>
#include <set>
#include <utility>

#include "gpufl/core/common.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/logger/logger.hpp"
#include "gpufl/core/model/batch_models.hpp"
#include "gpufl/core/runtime.hpp"
#include "gpufl/core/segment_runtime.hpp"

namespace gpufl::detail {

void MonitorBatchManager::reset() {
    clearFlushSink();
    dictManager_.reset();
    kernelBatch_.clear();
    memcpyBatch_.clear();
    {
        std::lock_guard lk(scopeBatchMu_);
        scopeBatch_.clear();
        profileBatch_.clear();
        pmSampleBatch_.clear();
        scopeNameStack_.clear();
        openScopeWindows_.clear();
        completedScopeWindows_.clear();
        scopeRetentionWatermarkNs_ = 0;
        pmScopeAttributionStartNs_ = 0;
        scopeHistoryEvictionLogged_ = false;
        scopeAttributionTruncated_ = 0;
        pmSampleRowsSeen_ = 0;
        {
            std::lock_guard pending_lk(pendingScopeCloseMu_);
            pendingScopeCloseNs_.clear();
        }
    }
    syncBatch_.clear();
    memAllocBatch_.clear();
    pendingDetails_.clear();

    kernelBatchId_ = 0;
    memcpyBatchId_ = 0;
    scopeBatchId_ = 0;
    profileBatchId_ = 0;
    pmSampleBatchId_ = 0;
    syncBatchId_ = 0;
    memAllocBatchId_ = 0;

    nextScopeInstanceId_.store(1);
    activeScopeNameId_.store(0);
}

void MonitorBatchManager::bindFlushRuntime(Runtime* runtime) {
    flushSink_.runtime = runtime;
}

void MonitorBatchManager::clearFlushSink() {
    flushSink_ = {};
}

void MonitorBatchManager::setSourceCollectionEnabled(bool enabled) {
    dictManager_.enable_source_collection = enabled;
}

void MonitorBatchManager::flushAll(FlushMode mode) {
    if (!flushSink_.available()) {
        // No logger bound (bindFlushSink never ran). Every flush is then a
        // silent no-op and batches grow unbounded — surface it when there is
        // actually buffered data to drop instead of losing events without a
        // trace. Checks only collector-owned batches (lock-free; flushAll
        // never runs concurrently with the collector).
        if (!kernelBatch_.empty() || !memcpyBatch_.empty() || !syncBatch_.empty() ||
            !memAllocBatch_.empty() || !pendingDetails_.empty()) {
            GFL_LOG_ERROR("MonitorBatchManager::flushAll: no logger bound, dropping buffered events");
        }
        return;
    }

    const auto context = flushSink_.runtime->acquireSegmentContext();
    if (!context || !context->logger) {
        GFL_LOG_ERROR(
            "MonitorBatchManager::flushAll: active segment context is missing");
        return;
    }
    Logger& logger = *context->logger;
    const std::string& session_id = context->session_id;
    uint64_t logical_rows = 0;
    const auto flushDictionary = [&] {
        if (context->dictionary) {
            context->dictionary->flush(dictManager_, logger, session_id);
        } else {
            dictManager_.flushDictionary(logger, session_id);
        }
    };

    // Dictionary MUST be written before any batch that references its IDs.
    flushDictionary();
    if (mode == FlushMode::Full) {
        dictManager_.flushSourceContent(logger, session_id);
        dictManager_.flushDisassembly(logger, session_id);
    }

    if (!kernelBatch_.empty()) {
        logical_rows += kernelBatch_.rows().size();
        flushDictionary();
        logger.write(model::KernelEventBatchModel(kernelBatch_, session_id, ++kernelBatchId_));
        kernelBatch_.clear();
        for (const auto& d : pendingDetails_) {
            logger.write(model::KernelDetailModel(d));
        }
        pendingDetails_.clear();
    }

    if (!memcpyBatch_.empty()) {
        logical_rows += memcpyBatch_.rows().size();
        flushDictionary();
        logger.write(model::MemcpyEventBatchModel(memcpyBatch_, session_id, ++memcpyBatchId_));
        memcpyBatch_.clear();
    }

    if (!syncBatch_.empty()) {
        logical_rows += syncBatch_.rows().size();
        flushDictionary();
        logger.write(model::SynchronizationEventBatchModel(syncBatch_, session_id, ++syncBatchId_));
        syncBatch_.clear();
    }

    if (!memAllocBatch_.empty()) {
        logical_rows += memAllocBatch_.rows().size();
        logger.write(model::MemoryAllocEventBatchModel(memAllocBatch_, session_id, ++memAllocBatchId_));
        memAllocBatch_.clear();
    }

    {
        std::lock_guard lk(scopeBatchMu_);
        if (!scopeBatch_.empty() || !profileBatch_.empty() || !pmSampleBatch_.empty()) {
            flushDictionary();
        }
        if (!scopeBatch_.empty()) {
            logical_rows += scopeBatch_.rows().size();
            logger.write(model::ScopeEventBatchModel(scopeBatch_, session_id, ++scopeBatchId_));
            scopeBatch_.clear();
        }
        if (!profileBatch_.empty()) {
            logical_rows += profileBatch_.rows().size();
            logger.write(model::ProfileSampleBatchModel(profileBatch_, session_id, ++profileBatchId_));
            profileBatch_.clear();
        }
        if (!pmSampleBatch_.empty()) {
            logical_rows += pmSampleBatch_.rows().size();
            logger.write(model::PmSampleBatchModel(pmSampleBatch_, session_id, ++pmSampleBatchId_));
            pmSampleBatch_.clear();
        }
    }
    if (logical_rows > 0 && flushSink_.runtime->segment_runtime) {
        const int64_t steady_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count();
        flushSink_.runtime->segment_runtime->noteRows(
            context->segment_index, logical_rows, steady_ns,
            GetTimestampNs());
    }
}

void MonitorBatchManager::flushDictionarySnapshot(
    SegmentDictionaryEmitter& emitter, Logger& logger,
    const std::string& session_id) {
    emitter.flush(dictManager_, logger, session_id);
}

uint32_t MonitorBatchManager::internKernel(const std::string& name) {
    return dictManager_.internKernel(name);
}

uint32_t MonitorBatchManager::internScopeName(const std::string& name) {
    return dictManager_.internScopeName(name);
}

uint32_t MonitorBatchManager::internFunction(const std::string& name,
                                             const std::string& func_symbol) {
    return dictManager_.internFunction(name, func_symbol);
}

uint32_t MonitorBatchManager::internMetric(const std::string& name) {
    return dictManager_.internMetric(name);
}

uint32_t MonitorBatchManager::internSourceFile(const std::string& path) {
    return dictManager_.internSourceFile(path);
}

void MonitorBatchManager::enqueueDisassembly(uint64_t crc, const uint8_t* data, size_t size) {
    dictManager_.enqueueDisassembly(crc, data, size);
}

void MonitorBatchManager::flushDisassembly() {
    if (!flushSink_.available()) return;
    const auto context = flushSink_.runtime->acquireSegmentContext();
    if (!context || !context->logger) return;
    dictManager_.flushDisassembly(*context->logger, context->session_id);
}

uint64_t MonitorBatchManager::allocateScopeInstanceId() {
    return nextScopeInstanceId_.fetch_add(1, std::memory_order_relaxed);
}

uint32_t MonitorBatchManager::activeScopeNameId() const {
    return activeScopeNameId_.load(std::memory_order_relaxed);
}

int MonitorBatchManager::openScopeDepth() const {
    std::lock_guard lk(scopeBatchMu_);
    return static_cast<int>(scopeNameStack_.size());
}

int64_t MonitorBatchManager::captureScopeCloseTimestamp(uint64_t instance_id) {
    // The timestamp and its publication share the same lock observed by the PM
    // snapshot. If snapshot wins, its batch predates this close; if close wins,
    // snapshot sees the exact end and cannot provisionally extend past it.
    std::lock_guard lk(pendingScopeCloseMu_);
    const int64_t end_ns = GetTimestampNs();
    pendingScopeCloseNs_[instance_id] = end_ns;
    return end_ns;
}

void MonitorBatchManager::markScopeClosePending(uint64_t instance_id, int64_t end_ns) {
    std::lock_guard lk(pendingScopeCloseMu_);
    const auto [it, inserted] = pendingScopeCloseNs_.emplace(instance_id, end_ns);
    if (!inserted && end_ns < it->second) it->second = end_ns;
}

bool MonitorBatchManager::pushKernel(const KernelBatchRow& row,
                                     const KernelDetailRow* detail) {
    kernelBatch_.push(row);
    if (detail) {
        pendingDetails_.push_back(*detail);
    }
    return kernelBatch_.needsFlush();
}

bool MonitorBatchManager::pushMemcpy(const MemcpyBatchRow& row) {
    memcpyBatch_.push(row);
    return memcpyBatch_.needsFlush();
}

void MonitorBatchManager::pushTraceScopeRows(const ScopeBatchRow& begin_row,
                                             const ScopeBatchRow& end_row) {
    std::lock_guard lk(scopeBatchMu_);
    scopeBatch_.push(begin_row);
    scopeBatch_.push(end_row);
}

void MonitorBatchManager::pushTrackedScopeRow(const ScopeBatchRow& row) {
    // Publish a close's already-captured timestamp before waiting for the main
    // scope lock. A concurrent PM snapshot can then stop the open interval at
    // this timestamp instead of extending it through the whole sample batch.
    if (row.event_type != 0) {
        markScopeClosePending(row.scope_instance_id, row.ts_ns);
    }

    std::lock_guard lk(scopeBatchMu_);
    if (row.event_type == 0) {
        scopeNameStack_.emplace_back(row.scope_instance_id, row.name_id);
        activeScopeNameId_.store(row.name_id, std::memory_order_relaxed);
        openScopeWindows_[row.scope_instance_id] = {
            row.original_start_ns == 0 ? row.ts_ns : row.original_start_ns,
            row.name_id, row.depth, row.repeat, row.warmup};
    } else {
        // Search from the back: the common case is closing the innermost
        // scope, and an unmatched id leaves the stack alone rather than
        // popping somebody else's scope.
        for (auto it = scopeNameStack_.rbegin(); it != scopeNameStack_.rend(); ++it) {
            if (it->first == row.scope_instance_id) {
                scopeNameStack_.erase(std::next(it).base());
                break;
            }
        }
        activeScopeNameId_.store(
            scopeNameStack_.empty() ? 0 : scopeNameStack_.back().second,
            std::memory_order_relaxed);

        if (const auto it = openScopeWindows_.find(row.scope_instance_id);
            it != openScopeWindows_.end()) {
            completedScopeWindows_.push_back(
                    {it->second.start_ns, row.ts_ns, row.scope_instance_id, row.name_id, it->second.depth});
            openScopeWindows_.erase(it);
            // Bound it here rather than only when PM samples arrive. Nothing
            // publishes a retention watermark unless PM is actually sampling,
            // so a Trace-only run would grow this for the life of the process.
            enforceScopeCapLocked();
        }
        clearPendingScopeCloseLocked(row.scope_instance_id);
    }
    scopeBatch_.push(row);
}

std::pair<std::vector<ScopeBatchRow>, std::vector<ScopeBatchRow>>
MonitorBatchManager::snapshotScopeContinuations(
    const int64_t boundary_ns) const {
    std::lock_guard lk(scopeBatchMu_);
    std::vector<ScopeBatchRow> closes;
    std::vector<ScopeBatchRow> opens;
    closes.reserve(openScopeWindows_.size());
    opens.reserve(openScopeWindows_.size());
    for (const auto& [instance_id, open] : openScopeWindows_) {
        ScopeBatchRow close;
        close.ts_ns = boundary_ns;
        close.scope_instance_id = instance_id;
        close.name_id = open.name_id;
        close.event_type = 3;
        close.depth = open.depth;
        close.original_start_ns = open.start_ns;
        closes.push_back(close);

        ScopeBatchRow next = close;
        next.event_type = 2;
        next.repeat = open.repeat;
        next.warmup = open.warmup;
        opens.push_back(next);
    }
    const auto by_depth_then_id = [](const ScopeBatchRow& lhs,
                                     const ScopeBatchRow& rhs) {
        if (lhs.depth != rhs.depth) return lhs.depth < rhs.depth;
        return lhs.scope_instance_id < rhs.scope_instance_id;
    };
    std::sort(closes.begin(), closes.end(), by_depth_then_id);
    std::sort(opens.begin(), opens.end(), by_depth_then_id);
    return {std::move(closes), std::move(opens)};
}

void MonitorBatchManager::writeScopeRows(
    Logger& logger, const std::string& session_id,
    const std::vector<ScopeBatchRow>& rows) {
    if (rows.empty()) return;
    BatchBuffer<ScopeBatchRow> batch;
    for (const auto& row : rows) batch.push(row);
    uint64_t batch_id = 0;
    {
        std::lock_guard lk(scopeBatchMu_);
        batch_id = ++scopeBatchId_;
    }
    logger.write(model::ScopeEventBatchModel(batch, session_id, batch_id));
}

bool MonitorBatchManager::pushProfileSample(const ProfileSampleBatchRow& row) {
    std::lock_guard lk(scopeBatchMu_);
    profileBatch_.push(row);
    return profileBatch_.needsFlush();
}

void MonitorBatchManager::pushProfileSamples(const std::vector<ProfileSampleBatchRow>& rows) {
    std::lock_guard lk(scopeBatchMu_);
    for (const auto& row : rows) {
        profileBatch_.push(row);
    }
}

void MonitorBatchManager::pushPmSamplesResolvingScopes(const std::vector<PmSampleBatchRow>& rows) {
    if (rows.empty()) return;

    std::vector resolved(rows.begin(), rows.end());

    // Snapshot under the lock, sweep outside it. Holding scopeBatchMu_ for the
    // whole sort-and-sweep would block every scope close for the duration, and
    // a close that is already holding its end timestamp and waiting here is
    // exactly what widens the window below.
    std::vector<ScopeWindow> candidates;
    {
        std::lock_guard lk(scopeBatchMu_);
        pmSampleRowsSeen_ += rows.size();
        trimCompletedScopesLocked();
        candidates = snapshotScopeCandidatesLocked(rows);
    }

    // No fallback to the currently active scope. A sample no interval covers is
    // left unattributed, and that is the point of this path: handing it
    // whichever scope happens to be open at DECODE time answers "what is
    // running now", not "what was running when this was sampled", and the two
    // differ by however long the sample sat in the buffer - exactly the error
    // this resolver exists to remove. Open scopes are already candidates, so a
    // sample inside a still-running scope resolves properly rather than by luck.
    resolveScopeIdsForBatch(candidates, resolved, /*fallback_id=*/0);

    {
        std::lock_guard lk(scopeBatchMu_);
        for (const auto& row : resolved) pmSampleBatch_.push(row);
    }
}

void MonitorBatchManager::publishScopeRetentionWatermark(int64_t ts_ns) {
    std::lock_guard lk(scopeBatchMu_);
    // Monotonic. A caller that regressed - a decode that failed, a buffer that
    // overflowed - must not be able to un-retire scopes it already released.
    if (ts_ns > scopeRetentionWatermarkNs_) scopeRetentionWatermarkNs_ = ts_ns;
}

void MonitorBatchManager::beginPmScopeAttribution(int64_t start_ns) {
    std::lock_guard lk(scopeBatchMu_);
    pmScopeAttributionStartNs_ = start_ns;
}

void MonitorBatchManager::endPmScopeAttribution() {
    std::lock_guard lk(scopeBatchMu_);
    pmScopeAttributionStartNs_ = 0;
}

uint64_t MonitorBatchManager::scopeAttributionTruncated() const {
    std::lock_guard lk(scopeBatchMu_);
    return scopeAttributionTruncated_;
}

uint64_t MonitorBatchManager::pmSampleRowsSeen() const {
    std::lock_guard lk(scopeBatchMu_);
    return pmSampleRowsSeen_;
}

void MonitorBatchManager::resolveScopeIdsForTesting(std::vector<PmSampleBatchRow>& rows,
                                                    uint32_t fallback_id) {
    std::vector<ScopeWindow> candidates;
    {
        std::lock_guard lk(scopeBatchMu_);
        trimCompletedScopesLocked();
        candidates = snapshotScopeCandidatesLocked(rows);
    }
    resolveScopeIdsForBatch(candidates, rows, fallback_id);
}

uint32_t MonitorBatchManager::resolveScopeIdForTesting(int64_t ts_ns) const {
    std::lock_guard lk(scopeBatchMu_);
    return resolveScopeIdLocked(ts_ns);
}

size_t MonitorBatchManager::retainedCompletedScopesForTesting() const {
    std::lock_guard lk(scopeBatchMu_);
    return completedScopeWindows_.size();
}

void MonitorBatchManager::clearPendingScopeCloseLocked(uint64_t instance_id) {
    // Called with scopeBatchMu_ held. Snapshot takes locks in the same order:
    // scopeBatchMu_ first, pendingScopeCloseMu_ second.
    std::lock_guard lk(pendingScopeCloseMu_);
    pendingScopeCloseNs_.erase(instance_id);
}

void MonitorBatchManager::enforceScopeCapLocked() {
    // Runs on every close, not only when PM samples arrive. The watermark is
    // the real bound, but nothing publishes one unless PM is actually
    // sampling - so a Trace-only run, or one where PM never initialised, would
    // otherwise grow this deque for the life of the process with no cap and no
    // telemetry to show for it.
    if (completedScopeWindows_.size() <= kMaxCompletedScopes) return;
    const size_t excess = completedScopeWindows_.size() - kMaxCompletedScopes;
    uint64_t attribution_risk = 0;
    if (pmScopeAttributionStartNs_ > 0) {
        for (size_t i = 0; i < excess; ++i) {
            if (completedScopeWindows_[i].end_ns >= pmScopeAttributionStartNs_) {
                ++attribution_risk;
            }
        }
    }
    completedScopeWindows_.erase(completedScopeWindows_.begin(),
                                 completedScopeWindows_.begin() + static_cast<long>(excess));
    // Trace-only history is unused by PM attribution. Count an eviction as a
    // data-quality risk only when it overlaps the current PM collection
    // boundary. This also handles mixed sessions: a long Trace warmup cannot
    // make a later Deep PM window look partial merely because old, pre-window
    // entries are evicted while PM happens to be active.
    scopeAttributionTruncated_ += attribution_risk;
    if (!scopeHistoryEvictionLogged_) {
        scopeHistoryEvictionLogged_ = true;
        GFL_LOG_ERROR("[MonitorBatchManager] scope_history_evicted: hard cap reached; ",
                      excess,
                      " completed scope record(s) dropped (further messages suppressed)");
    }
}

void MonitorBatchManager::trimCompletedScopesLocked() {
    // The deque is NOT ordered by end_ns, so this cannot stop at the first
    // survivor. It is a full pass, but only over what the watermark has not
    // already retired, and it runs once per drain rather than once per sample.
    if (scopeRetentionWatermarkNs_ > 0) {
        const int64_t cutoff = scopeRetentionWatermarkNs_;
        const auto it = std::remove_if(
            completedScopeWindows_.begin(), completedScopeWindows_.end(),
            [cutoff](const ScopeWindow& w) { return w.end_ns < cutoff; });
        completedScopeWindows_.erase(it, completedScopeWindows_.end());
    }

    // Backstop. Reaching this means the watermark is not advancing, so the
    // entries dropped here may still have been needed: record it rather than
    // let the samples quietly go unattributed.
    enforceScopeCapLocked();
}

std::vector<MonitorBatchManager::ScopeWindow>
MonitorBatchManager::snapshotScopeCandidatesLocked(
        const std::vector<PmSampleBatchRow>& rows) const {
    // Candidates = closed scopes still retained, PLUS scopes that are still
    // open. The open ones matter: PM drains mid-run, so a sample is routinely
    // decoded while the scope covering it is still running. Giving them a
    // provisional end at the batch's newest sample keeps them eligible for
    // every sample in this batch without inventing a close that has not
    // happened.
    //
    int64_t provisional_end = (std::numeric_limits<int64_t>::min)();
    for (const auto& row : rows) {
        provisional_end = (std::max)(provisional_end, row.ts_ns);
    }

    std::vector<ScopeWindow> candidates;
    candidates.reserve(completedScopeWindows_.size() + openScopeWindows_.size());
    candidates.assign(completedScopeWindows_.begin(), completedScopeWindows_.end());
    {
        // Close publishes pending first and never holds this mutex while
        // waiting for scopeBatchMu_, so this lock order cannot deadlock.
        std::lock_guard pending_lk(pendingScopeCloseMu_);
        for (const auto& [instance_id, open] : openScopeWindows_) {
            int64_t effective_end = provisional_end;
            if (const auto close = pendingScopeCloseNs_.find(instance_id);
                close != pendingScopeCloseNs_.end()) {
                effective_end = (std::min)(effective_end, close->second);
            }
            if (open.start_ns > effective_end) continue;
            candidates.push_back(ScopeWindow{open.start_ns, effective_end, instance_id,
                                             open.name_id, open.depth});
        }
    }
    return candidates;
}

void MonitorBatchManager::resolveScopeIdsForBatch(std::vector<ScopeWindow>& candidates,
                                                  std::vector<PmSampleBatchRow>& rows,
                                                  uint32_t fallback_id) {
    if (candidates.empty()) {
        for (auto& row : rows) row.scope_name_id = fallback_id;
        return;
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const ScopeWindow& a, const ScopeWindow& b) { return a.start_ns < b.start_ns; });

    std::vector<size_t> order(rows.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&rows](size_t a, size_t b) { return rows[a].ts_ns < rows[b].ts_ns; });

    // Two structures over the active set rather than one flat list. Rescanning
    // it per sample would keep the cost at O(samples x concurrent scopes),
    // which is the shape this replaces. `ranked` is ordered so the winner is
    // its last element; `expiry` surfaces the soonest end so retirement costs
    // a peek instead of a pass.
    struct ByRank {
        bool operator()(const ScopeWindow* a, const ScopeWindow* b) const {
            return b->outranks(*a);
        }
    };
    struct ByEnd {
        bool operator()(const ScopeWindow* a, const ScopeWindow* b) const {
            return a->end_ns > b->end_ns;   // min-heap on end_ns
        }
    };
    std::set<const ScopeWindow*, ByRank> ranked;
    std::priority_queue<const ScopeWindow*, std::vector<const ScopeWindow*>, ByEnd> expiry;

    size_t next_candidate = 0;
    for (const size_t idx : order) {
        const int64_t ts = rows[idx].ts_ns;

        // Admit everything that has started. Candidates are start-sorted, so
        // each is admitted once across the whole batch.
        while (next_candidate < candidates.size() && candidates[next_candidate].start_ns <= ts) {
            const ScopeWindow* w = &candidates[next_candidate++];
            ranked.insert(w);
            expiry.push(w);
        }
        // Retire what has ended. Samples are visited in time order, so an
        // expired scope can never be wanted again. Both ends are inclusive,
        // hence `end_ns < ts` rather than `<=`.
        while (!expiry.empty() && expiry.top()->end_ns < ts) {
            ranked.erase(expiry.top());
            expiry.pop();
        }

        rows[idx].scope_name_id = ranked.empty() ? fallback_id : (*ranked.rbegin())->name_id;
    }
}

bool MonitorBatchManager::pushMemoryAlloc(const MemoryAllocEventBatchRow& row) {
    memAllocBatch_.push(row);
    return memAllocBatch_.needsFlush();
}

void MonitorBatchManager::pushSynchronization(const SynchronizationEventBatchRow& row) {
    syncBatch_.push(row);
}

// Ranking shared by both resolvers. Depth first, then latest start; the
// instance id breaks a remaining tie so the answer does not depend on
// container order. Without it two scopes at the same depth and start could
// resolve differently between the reference and the batch path, since neither
// std::sort nor an unordered_map preserves any order for equal keys.
//
// The tertiary key is also load-bearing for the sweep, not merely tidy: it
// orders a std::set, and a comparator that ever reports equivalence would make
// that set silently keep one of the two scopes and drop the other. Instance ids
// are unique, so no two entries can compare equal.
bool MonitorBatchManager::ScopeWindow::outranks(const ScopeWindow& other) const {
    if (depth != other.depth) return depth > other.depth;
    if (start_ns != other.start_ns) return start_ns > other.start_ns;
    return instance_id > other.instance_id;
}

uint32_t MonitorBatchManager::resolveScopeIdLocked(int64_t ts_ns) const {
    const ScopeWindow* best = nullptr;
    for (const auto& w : completedScopeWindows_) {
        if (ts_ns < w.start_ns || ts_ns > w.end_ns) continue;
        if (!best || w.outranks(*best)) best = &w;
    }
    return best ? best->name_id : 0;
}

}  // namespace gpufl::detail

#include "gpufl/core/monitor_batch_manager.hpp"

#include <utility>

#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/logger/logger.hpp"
#include "gpufl/core/model/batch_models.hpp"

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
        openScopeWindows_.clear();
        completedScopeWindows_.clear();
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

void MonitorBatchManager::bindFlushSink(Logger* logger, std::string session_id) {
    flushSink_.logger = logger;
    flushSink_.session_id = std::move(session_id);
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

    Logger& logger = *flushSink_.logger;
    const std::string& session_id = flushSink_.session_id;

    // Dictionary MUST be written before any batch that references its IDs.
    dictManager_.flushDictionary(logger, session_id);
    if (mode == FlushMode::Full) {
        dictManager_.flushSourceContent(logger, session_id);
        dictManager_.flushDisassembly(logger, session_id);
    }

    if (!kernelBatch_.empty()) {
        dictManager_.flushDictionary(logger, session_id);
        logger.write(model::KernelEventBatchModel(kernelBatch_, session_id, ++kernelBatchId_));
        kernelBatch_.clear();
        for (const auto& d : pendingDetails_) {
            logger.write(model::KernelDetailModel(d));
        }
        pendingDetails_.clear();
    }

    if (!memcpyBatch_.empty()) {
        dictManager_.flushDictionary(logger, session_id);
        logger.write(model::MemcpyEventBatchModel(memcpyBatch_, session_id, ++memcpyBatchId_));
        memcpyBatch_.clear();
    }

    if (!syncBatch_.empty()) {
        dictManager_.flushDictionary(logger, session_id);
        logger.write(model::SynchronizationEventBatchModel(syncBatch_, session_id, ++syncBatchId_));
        syncBatch_.clear();
    }

    if (!memAllocBatch_.empty()) {
        logger.write(model::MemoryAllocEventBatchModel(memAllocBatch_, session_id, ++memAllocBatchId_));
        memAllocBatch_.clear();
    }

    {
        std::lock_guard lk(scopeBatchMu_);
        if (!scopeBatch_.empty() || !profileBatch_.empty() || !pmSampleBatch_.empty()) {
            dictManager_.flushDictionary(logger, session_id);
        }
        if (!scopeBatch_.empty()) {
            logger.write(model::ScopeEventBatchModel(scopeBatch_, session_id, ++scopeBatchId_));
            scopeBatch_.clear();
        }
        if (!profileBatch_.empty()) {
            logger.write(model::ProfileSampleBatchModel(profileBatch_, session_id, ++profileBatchId_));
            profileBatch_.clear();
        }
        if (!pmSampleBatch_.empty()) {
            logger.write(model::PmSampleBatchModel(pmSampleBatch_, session_id, ++pmSampleBatchId_));
            pmSampleBatch_.clear();
        }
    }
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
    dictManager_.flushDisassembly(*flushSink_.logger, flushSink_.session_id);
}

uint64_t MonitorBatchManager::allocateScopeInstanceId() {
    return nextScopeInstanceId_.fetch_add(1, std::memory_order_relaxed);
}

uint32_t MonitorBatchManager::activeScopeNameId() const {
    return activeScopeNameId_.load(std::memory_order_relaxed);
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
    if (row.event_type == 0) {
        activeScopeNameId_.store(row.name_id, std::memory_order_relaxed);
    }

    std::lock_guard lk(scopeBatchMu_);
    if (row.event_type == 0) {
        openScopeWindows_[row.scope_instance_id] = {row.ts_ns, row.name_id, row.depth};
    } else {
        if (const auto it = openScopeWindows_.find(row.scope_instance_id);
            it != openScopeWindows_.end()) {
            completedScopeWindows_.push_back(
                    {it->second.start_ns, row.ts_ns, row.scope_instance_id, row.name_id, it->second.depth});
            openScopeWindows_.erase(it);
        }
    }
    scopeBatch_.push(row);
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
    const uint32_t fallback_id = activeScopeNameId_.load(std::memory_order_relaxed);
    std::lock_guard lk(scopeBatchMu_);
    for (const auto& sample : rows) {
        PmSampleBatchRow row = sample;
        row.scope_name_id = resolveScopeIdLocked(row.ts_ns);
        if (row.scope_name_id == 0) row.scope_name_id = fallback_id;
        pmSampleBatch_.push(row);
    }
}

bool MonitorBatchManager::pushMemoryAlloc(const MemoryAllocEventBatchRow& row) {
    memAllocBatch_.push(row);
    return memAllocBatch_.needsFlush();
}

void MonitorBatchManager::pushSynchronization(const SynchronizationEventBatchRow& row) {
    syncBatch_.push(row);
}

uint32_t MonitorBatchManager::resolveScopeIdLocked(int64_t ts_ns) const {
    uint32_t best_id = 0;
    int best_depth = -1;
    int64_t best_start = 0;
    for (auto it = completedScopeWindows_.rbegin(); it != completedScopeWindows_.rend(); ++it) {
        if (ts_ns < it->start_ns || ts_ns > it->end_ns) continue;
        if (it->depth > best_depth || (it->depth == best_depth && it->start_ns >= best_start)) {
            best_id = it->name_id;
            best_depth = it->depth;
            best_start = it->start_ns;
        }
    }
    return best_id;
}

}  // namespace gpufl::detail

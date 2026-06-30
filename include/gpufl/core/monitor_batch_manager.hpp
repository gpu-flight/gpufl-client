#pragma once

#include <atomic>
#include <cstdint>
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
    };

    struct OpenScopeWindow {
        int64_t start_ns = 0;
        uint32_t name_id = 0;
        int depth = 0;
    };

    uint32_t resolveScopeIdLocked(int64_t ts_ns) const;

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
    std::atomic<uint32_t> activeScopeNameId_{0};
    std::unordered_map<uint64_t, OpenScopeWindow> openScopeWindows_;
    std::vector<ScopeWindow> completedScopeWindows_;

    BatchBuffer<SynchronizationEventBatchRow> syncBatch_;
    BatchBuffer<MemoryAllocEventBatchRow> memAllocBatch_;
    uint64_t syncBatchId_ = 0;
    uint64_t memAllocBatchId_ = 0;
};

}  // namespace detail
}  // namespace gpufl

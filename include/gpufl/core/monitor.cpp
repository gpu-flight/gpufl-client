#include "gpufl/core/monitor.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <map>
#include <mutex>
#include <stack>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "gpufl/core/activity_record.hpp"
#include "gpufl/core/batch_buffer.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/dictionary_manager.hpp"
#include "gpufl/core/logger/logger.hpp"
#include "gpufl/core/model/batch_models.hpp"
#include "gpufl/core/model/memcpy_event_model.hpp"
#include "gpufl/core/model/nvtx_marker_model.hpp"
#include "gpufl/core/model/synchronization_event_model.hpp"
#include "gpufl/core/model/memory_alloc_event_model.hpp"
#include "gpufl/core/model/graph_launch_event_model.hpp"
#include "gpufl/core/model/lifecycle_model.hpp"
#include "gpufl/core/monitor_adapter.hpp"
#include "gpufl/core/ring_buffer.hpp"
#include "gpufl/core/runtime.hpp"
#include "gpufl/core/stack_registry.hpp"
#include "gpufl/core/stack_trace.hpp"

namespace gpufl {

// Global ring buffer (as declared in monitor.hpp)
RingBuffer<ActivityRecord, kMonitorBufferSize> g_monitorBuffer;

namespace {

/**
 * @brief Encapsulates all batching and dictionary flushing logic.
 */
class BatchManager {
public:
    void reset() {
        dictManager.reset();
        kernelBatch.clear();
        memcpyBatch.clear();
        scopeBatch.clear();
        profileBatch.clear();
        pmSampleBatch.clear();
        syncBatch.clear();
        memAllocBatch.clear();
        pendingDetails.clear();

        kernelBatchId = 0;
        memcpyBatchId = 0;
        scopeBatchId = 0;
        profileBatchId = 0;
        pmSampleBatchId = 0;
        syncBatchId = 0;
        memAllocBatchId = 0;

        nextScopeInstanceId.store(1);
        activeScopeNameId.store(0);
    }

    enum class FlushMode { Fast, Full };

    void flushAll(Logger& logger, const std::string& session_id, FlushMode mode = FlushMode::Fast) {
        // Dictionary MUST be written before any batch that references its IDs.
        dictManager.flushDictionary(logger, session_id);
        if (mode == FlushMode::Full) {
            dictManager.flushSourceContent(logger, session_id);
            dictManager.flushDisassembly(logger, session_id);
        }

        if (!kernelBatch.empty()) {
            dictManager.flushDictionary(logger, session_id);
            logger.write(model::KernelEventBatchModel(kernelBatch, session_id, ++kernelBatchId));
            kernelBatch.clear();
            for (const auto& d : pendingDetails) {
                logger.write(model::KernelDetailModel(d));
            }
            pendingDetails.clear();
        }

        if (!memcpyBatch.empty()) {
            dictManager.flushDictionary(logger, session_id);
            logger.write(model::MemcpyEventBatchModel(memcpyBatch, session_id, ++memcpyBatchId));
            memcpyBatch.clear();
        }

        if (!syncBatch.empty()) {
            dictManager.flushDictionary(logger, session_id);
            logger.write(model::SynchronizationEventBatchModel(syncBatch, session_id, ++syncBatchId));
            syncBatch.clear();
        }

        if (!memAllocBatch.empty()) {
            logger.write(model::MemoryAllocEventBatchModel(memAllocBatch, session_id, ++memAllocBatchId));
            memAllocBatch.clear();
        }

        {
            std::lock_guard lk(scopeBatchMu);
            if (!scopeBatch.empty() || !profileBatch.empty() || !pmSampleBatch.empty()) {
                dictManager.flushDictionary(logger, session_id);
            }
            if (!scopeBatch.empty()) {
                logger.write(model::ScopeEventBatchModel(scopeBatch, session_id, ++scopeBatchId));
                scopeBatch.clear();
            }
            if (!profileBatch.empty()) {
                logger.write(model::ProfileSampleBatchModel(profileBatch, session_id, ++profileBatchId));
                profileBatch.clear();
            }
            if (!pmSampleBatch.empty()) {
                logger.write(model::PmSampleBatchModel(pmSampleBatch, session_id, ++pmSampleBatchId));
                pmSampleBatch.clear();
            }
        }
    }

    DictionaryManager dictManager;
    
    BatchBuffer<KernelBatchRow> kernelBatch;
    BatchBuffer<MemcpyBatchRow> memcpyBatch;
    uint64_t kernelBatchId = 0;
    uint64_t memcpyBatchId = 0;
    std::vector<KernelDetailRow> pendingDetails;

    BatchBuffer<ScopeBatchRow> scopeBatch;
    BatchBuffer<ProfileSampleBatchRow> profileBatch;
    BatchBuffer<PmSampleBatchRow> pmSampleBatch;
    uint64_t scopeBatchId = 0;
    uint64_t profileBatchId = 0;
    uint64_t pmSampleBatchId = 0;
    std::mutex scopeBatchMu;
    std::atomic<uint64_t> nextScopeInstanceId{1};
    std::atomic<uint32_t> activeScopeNameId{0};

    BatchBuffer<SynchronizationEventBatchRow> syncBatch;
    BatchBuffer<MemoryAllocEventBatchRow> memAllocBatch;
    uint64_t syncBatchId = 0;
    uint64_t memAllocBatchId = 0;
};

/**
 * @brief Manages metadata joins, demangling, and execution signatures.
 */
class MetadataManager {
public:
    void reset() {
        kernelNameDemangleCache.clear();
        functionKeyDemangleCache.clear();
        launchMetaByCorr.clear();
        emittedKernelCorrIds.clear();
        syncStackByCorr.clear();
        execSignatureByScope.clear();
    }

    const std::string& demangleKernelName(const char* raw) {
        static const std::string kFallback = "kernel_launch";
        if (!raw || raw[0] == '\0') return kFallback;
        auto it = kernelNameDemangleCache.find(raw);
        if (it != kernelNameDemangleCache.end()) return it->second;
        auto [ins, _] = kernelNameDemangleCache.emplace(raw, gpufl::core::DemangleName(raw));
        return ins->second;
    }

    const std::string& demangleFunctionKey(const std::string& key) {
        auto it = functionKeyDemangleCache.find(key);
        if (it != functionKeyDemangleCache.end()) return it->second;
        auto [ins, _] = functionKeyDemangleCache.emplace(key, gpufl::core::DemangleFunctionKey(key));
        return ins->second;
    }

    void accumulateSignature(const ActivityRecord& rec) {
        if (!rec.has_details) return;
        std::string key = rec.name;
        key += '\x1f';
        key += std::to_string(rec.grid_x);  key += ',';
        key += std::to_string(rec.grid_y);  key += ',';
        key += std::to_string(rec.grid_z);  key += '\x1f';
        key += std::to_string(rec.block_x); key += ',';
        key += std::to_string(rec.block_y); key += ',';
        key += std::to_string(rec.block_z); key += '\x1f';
        key += std::to_string(rec.dyn_shared);
        const std::string scope = (rec.user_scope[0] != '\0') ? rec.user_scope : std::string();
        ++execSignatureByScope[scope][key];
    }

    void emitSignatures(Runtime* rt) {
        if (execSignatureByScope.empty() || !(rt && rt->logger)) return;
        const int64_t ts = detail::GetTimestampNs();
        for (const auto& [scope, kernels] : execSignatureByScope) {
            std::string buf;
            uint64_t launch_count = 0;
            for (const auto& [k, cnt] : kernels) {
                buf += k; buf += '='; buf += std::to_string(cnt); buf += ';';
                launch_count += cnt;
            }
            ExecutionSignatureEvent ev;
            ev.session_id = rt->session_id;
            ev.ts_ns = ts;
            ev.scope_name = scope;
            ev.signature = Fnv1a64(buf);
            ev.launch_count = launch_count;
            ev.distinct_kernels = static_cast<uint32_t>(kernels.size());
            rt->logger->write(model::ExecutionSignatureModel(ev));
        }
        execSignatureByScope.clear();
    }

    void joinLaunchMeta(ActivityRecord& rec) {
        auto it = launchMetaByCorr.find(rec.corr_id);
        if (it == launchMetaByCorr.end()) return;
        const ActivityRecord& m = it->second;
        rec.scope_depth = m.scope_depth;
        rec.stack_id = m.stack_id;
        std::memcpy(rec.user_scope, m.user_scope, sizeof(rec.user_scope));
        rec.api_start_ns = m.api_start_ns;
        rec.api_exit_ns = m.api_exit_ns;
        rec.const_bytes = m.const_bytes;
        launchMetaByCorr.erase(it);
    }

private:
    static uint64_t Fnv1a64(const std::string& s) {
        uint64_t h = 1469598103934665603ULL;
        for (const unsigned char c : s) {
            h ^= c; h *= 1099511628211ULL;
        }
        return h;
    }

public:
    std::unordered_map<std::string, std::string> kernelNameDemangleCache;
    std::unordered_map<std::string, std::string> functionKeyDemangleCache;
    std::unordered_map<uint64_t, ActivityRecord> launchMetaByCorr;
    std::unordered_set<uint64_t> emittedKernelCorrIds;
    std::unordered_map<uint64_t, size_t> syncStackByCorr;
    std::map<std::string, std::map<std::string, uint64_t>> execSignatureByScope;
};

struct MonitorState {
    std::unique_ptr<IMonitorAdapter> adapter;
    std::atomic<bool> initialized{false};
    std::thread collectorThread;
    std::atomic<bool> collectorRunning{false};
    
    BatchManager batches;
    MetadataManager metadata;

    bool suppressOrphanSyntheticKernels = false;
    bool drainSyntheticMidRun = false;

    struct ScopeWindow {
        int64_t start_ns = 0, end_ns = 0;
        uint64_t instance_id = 0;
        uint32_t name_id = 0;
        int depth = 0;
    };
    struct OpenScopeWindow {
        int64_t start_ns = 0;
        uint32_t name_id = 0;
        int depth = 0;
    };
    std::unordered_map<uint64_t, OpenScopeWindow> openScopeWindows;
    std::vector<ScopeWindow> completedScopeWindows;

    uint32_t resolveScopeId(int64_t ts_ns) {
        uint32_t best_id = 0;
        int best_depth = -1;
        int64_t best_start = 0;
        for (auto it = completedScopeWindows.rbegin(); it != completedScopeWindows.rend(); ++it) {
            if (ts_ns < it->start_ns || ts_ns > it->end_ns) continue;
            if (it->depth > best_depth || (it->depth == best_depth && it->start_ns >= best_start)) {
                best_id = it->name_id; best_depth = it->depth; best_start = it->start_ns;
            }
        }
        return best_id;
    }
};

static MonitorState g_state;
static thread_local std::stack<void*> g_rangeStack;

// --- Helper Functions ---

static bool isSyntheticNonKernelLaunchName(const char* name) {
    if (!name || name[0] == '\0') return false;
    auto startsWith = [&](const char* prefix) {
        return std::strncmp(name, prefix, std::strlen(prefix)) == 0;
    };
    return startsWith("cudaMemcpy") || startsWith("cuMemcpy") ||
           startsWith("cudaMemset") || startsWith("cuMemset") ||
           startsWith("mem_transfer");
}

static void emitKernelRecord(const ActivityRecord& rec, const std::string& stack_trace, Runtime* rt) {
    const uint32_t kernel_id = g_state.batches.dictManager.internKernel(
        g_state.metadata.demangleKernelName(rec.name).c_str());

    KernelBatchRow row;
    row.start_ns = rec.cpu_start_ns;
    row.kernel_id = kernel_id;
    row.stream_id = static_cast<uint32_t>(rec.stream);
    row.duration_ns = rec.duration_ns;
    row.corr_id = rec.corr_id;
    row.dyn_shared = rec.dyn_shared;
    row.num_regs = rec.num_regs;
    row.has_details = rec.has_details ? 1 : 0;
    row.external_kind = rec.external_kind;
    row.external_id = rec.external_id;
    g_state.batches.kernelBatch.push(row);

    if (rec.has_details) {
        KernelDetailRow detail;
        detail.corr_id = rec.corr_id;
        detail.session_id = rt->session_id;
        detail.pid = detail::GetPid();
        detail.app = rt->app_name;
        detail.grid_x = rec.grid_x; detail.grid_y = rec.grid_y; detail.grid_z = rec.grid_z;
        detail.block_x = rec.block_x; detail.block_y = rec.block_y; detail.block_z = rec.block_z;
        detail.static_shared = rec.static_shared;
        detail.local_bytes = rec.local_bytes;
        detail.const_bytes = rec.const_bytes;
        detail.occupancy = rec.occupancy;
        detail.reg_occupancy = rec.reg_occupancy;
        detail.smem_occupancy = rec.smem_occupancy;
        detail.warp_occupancy = rec.warp_occupancy;
        detail.block_occupancy = rec.block_occupancy;
        std::memcpy(detail.limiting_resource, rec.limiting_resource, sizeof(detail.limiting_resource));
        detail.max_active_blocks = rec.max_active_blocks;
        detail.local_mem_total = rec.local_mem_total;
        detail.local_mem_per_thread = rec.local_mem_per_thread;
        detail.cache_config_requested = rec.cache_config_requested;
        detail.cache_config_executed = rec.cache_config_executed;
        detail.shared_mem_executed = rec.shared_mem_executed;
        detail.user_scope = rec.user_scope;
        detail.stack_trace = stack_trace;
        g_state.batches.pendingDetails.push_back(std::move(detail));
    }

    if (g_state.batches.kernelBatch.needsFlush()) {
        g_state.batches.flushAll(*rt->logger, rt->session_id);
    }
}

static void drainSyntheticKernels(Runtime* rt, int64_t maxApiStartNs = INT64_MAX) {
    auto& metaMap = g_state.metadata.launchMetaByCorr;
    if (metaMap.empty()) return;
    if (g_state.suppressOrphanSyntheticKernels) {
        if (maxApiStartNs == INT64_MAX) metaMap.clear();
        return;
    }
    if (!(rt && rt->logger)) return;
    
    const int64_t flushNs = detail::GetTimestampNs();
    std::vector<uint64_t> orderedCorr;
    orderedCorr.reserve(metaMap.size());
    for (const auto& [corr, _] : metaMap) orderedCorr.push_back(corr);
    std::sort(orderedCorr.begin(), orderedCorr.end(), [&](uint64_t a, uint64_t b) {
        return metaMap[a].api_start_ns < metaMap[b].api_start_ns;
    });

    std::vector<uint64_t> drained;
    for (size_t i = 0; i < orderedCorr.size(); ++i) {
        const uint64_t corr = orderedCorr[i];
        const ActivityRecord& meta = metaMap[corr];
        if (meta.api_start_ns > maxApiStartNs) continue;
        drained.push_back(corr);

        if (isSyntheticNonKernelLaunchName(meta.name)) continue;

        ActivityRecord out = meta;
        out.type = TraceType::KERNEL;
        out.stream = 0;
        out.cpu_start_ns = out.api_start_ns;
        const int64_t nextEnterNs = (i + 1 < orderedCorr.size()) ? metaMap[orderedCorr[i+1]].api_start_ns : flushNs;
        out.duration_ns = std::max<int64_t>(0, nextEnterNs - out.api_start_ns);
        out.corr_id = static_cast<unsigned>(corr);
        if (out.api_exit_ns <= 0) out.api_exit_ns = flushNs;

        g_state.metadata.emittedKernelCorrIds.insert(corr);
        const std::string stack_trace = (out.stack_id != 0) ? StackRegistry::instance().get(out.stack_id) : "";
        emitKernelRecord(out, stack_trace, rt);
    }
    for (uint64_t corr : drained) metaMap.erase(corr);
}

// --- Record Dispatcher ---

struct RecordProcessor {
    static bool processNext() {
        ActivityRecord rec{};
        if (!g_monitorBuffer.Consume(rec)) return false;

        switch (rec.type) {
            case TraceType::KERNEL_LAUNCH_META:
                handleKernelLaunchMeta(rec);
                return true;
            case TraceType::KERNEL_API_EXIT:
                handleKernelApiExit(rec);
                return true;
            case TraceType::SYNC_META:
                g_state.metadata.syncStackByCorr[rec.corr_id] = rec.stack_id;
                return true;
            default:
                break;
        }

        Runtime* rt = runtime();
        if (!(rt && rt->logger)) return true;

        switch (rec.type) {
            case TraceType::KERNEL:
            case TraceType::MEMCPY:
            case TraceType::MEMSET:
                handleGpuActivity(rec, rt);
                break;
            case TraceType::RANGE:
                handleRange(rec);
                break;
            case TraceType::PC_SAMPLE:
                handlePcSample(rec, rt);
                break;
            case TraceType::NVTX_MARKER:
                handleNvtxMarker(rec, rt);
                break;
            case TraceType::GRAPH_LAUNCH:
                handleGraphLaunch(rec, rt);
                break;
            case TraceType::MEMORY_ALLOC:
                handleMemoryAlloc(rec, rt);
                break;
            case TraceType::SYNCHRONIZATION:
                handleSynchronization(rec);
                break;
            default:
                break;
        }
        return true;
    }

private:
    static void handleKernelLaunchMeta(const ActivityRecord& rec) {
        const auto it = g_state.metadata.launchMetaByCorr.find(rec.corr_id);
        bool countForSignature = false;
        if (it == g_state.metadata.launchMetaByCorr.end()) {
            g_state.metadata.launchMetaByCorr.emplace(rec.corr_id, rec);
            countForSignature = rec.has_details;
        } else if (!it->second.has_details && rec.has_details) {
            it->second = rec;
            countForSignature = true;
        }
        if (countForSignature) g_state.metadata.accumulateSignature(rec);
    }

    static void handleKernelApiExit(const ActivityRecord& rec) {
        if (const auto it = g_state.metadata.launchMetaByCorr.find(rec.corr_id);
            it != g_state.metadata.launchMetaByCorr.end()) {
            it->second.api_exit_ns = rec.api_exit_ns;
        }
    }

    static void handleGpuActivity(ActivityRecord& rec, Runtime* rt) {
        g_state.metadata.joinLaunchMeta(rec);
        const std::string stack_trace = (rec.stack_id != 0) ? StackRegistry::instance().get(rec.stack_id) : "";

        if (rec.type == TraceType::KERNEL) {
            if (rec.corr_id != 0 && !g_state.metadata.emittedKernelCorrIds.insert(rec.corr_id).second) return;
            emitKernelRecord(rec, stack_trace, rt);
        } else if (rec.type == TraceType::MEMCPY) {
            MemcpyBatchRow row;
            row.start_ns = rec.cpu_start_ns;
            row.stream_id = static_cast<uint32_t>(rec.stream);
            row.duration_ns = rec.duration_ns;
            row.bytes = rec.bytes;
            row.copy_kind = rec.copy_kind;
            row.corr_id = rec.corr_id;
            g_state.batches.memcpyBatch.push(row);
            if (g_state.batches.memcpyBatch.needsFlush()) {
                g_state.batches.flushAll(*rt->logger, rt->session_id);
            }
        } else { // MEMSET
            MemsetEvent be;
            be.platform = g_state.adapter ? g_state.adapter->platformName() : "unknown";
            be.device_id = rec.device_id;
            be.stream_id = static_cast<uint32_t>(rec.stream);
            be.session_id = rt->session_id;
            be.pid = detail::GetPid();
            be.app = rt->app_name;
            be.name = rec.name;
            be.start_ns = rec.cpu_start_ns;
            be.end_ns = rec.cpu_start_ns + rec.duration_ns;
            be.api_start_ns = rec.api_start_ns;
            be.api_exit_ns = rec.api_exit_ns;
            be.user_scope = rec.user_scope;
            be.scope_depth = rec.scope_depth;
            be.corr_id = rec.corr_id;
            be.stack_trace = stack_trace;
            be.bytes = rec.bytes;
            rt->logger->write(model::MemsetEventModel(be));
        }
    }

    static void handleRange(const ActivityRecord& rec) {
        const uint32_t name_id = g_state.batches.dictManager.internScopeName(rec.name);
        const uint64_t instance_id = g_state.batches.nextScopeInstanceId.fetch_add(1, std::memory_order_relaxed);
        
        ScopeBatchRow begin_row, end_row;
        begin_row.ts_ns = rec.cpu_start_ns;
        begin_row.scope_instance_id = instance_id;
        begin_row.name_id = name_id;
        begin_row.event_type = 0;
        begin_row.depth = rec.scope_depth;

        end_row.ts_ns = rec.cpu_start_ns + rec.duration_ns;
        end_row.scope_instance_id = instance_id;
        end_row.name_id = name_id;
        end_row.event_type = 1;
        end_row.depth = rec.scope_depth;

        std::lock_guard lk(g_state.batches.scopeBatchMu);
        g_state.batches.scopeBatch.push(begin_row);
        g_state.batches.scopeBatch.push(end_row);
    }

    static void handlePcSample(const ActivityRecord& rec, Runtime* rt) {
        uint8_t kind = rec.metric_name[0] != '\0' ||
                    (rec.sample_kind[0] != '\0' && rec.sample_kind[0] == 's') ? 1 : 0;
        const std::string func_key = std::string(rec.function_name) + "@" + rec.source_file;
        const uint32_t function_id = g_state.batches.dictManager.internFunction(
            g_state.metadata.demangleFunctionKey(func_key), std::string(rec.function_name));
        const std::string metric_key = (rec.metric_name[0] != '\0') ? std::string(rec.metric_name) : rec.reason_name;
        const uint32_t metric_id = g_state.batches.dictManager.internMetric(metric_key);
        const uint32_t scope_name_id = g_state.batches.activeScopeNameId.load(std::memory_order_relaxed);
        const uint32_t source_file_id = g_state.batches.dictManager.internSourceFile(rec.source_file);

        ProfileSampleBatchRow row;
        row.ts_ns = rec.cpu_start_ns;
        row.corr_id = rec.corr_id;
        row.device_id = rec.device_id;
        row.function_id = function_id;
        row.pc_offset = rec.pc_offset;
        row.metric_id = metric_id;
        row.metric_value = (kind == 1) ? rec.metric_value : rec.samples_count;
        row.stall_reason = rec.stall_reason;
        row.sample_kind = kind;
        row.scope_name_id = scope_name_id;
        row.source_file_id = source_file_id;
        row.source_line = rec.source_line;

        bool needs_flush = false;
        {
            std::lock_guard lk(g_state.batches.scopeBatchMu);
            g_state.batches.profileBatch.push(row);
            needs_flush = g_state.batches.profileBatch.needsFlush();
        }
        if (needs_flush) {
            g_state.batches.flushAll(*rt->logger, rt->session_id);
        }
    }

    static void handleNvtxMarker(const ActivityRecord& rec, Runtime* rt) {
        NvtxMarkerEvent ev;
        ev.pid = detail::GetPid();
        ev.app = rt->app_name;
        ev.session_id = rt->session_id;
        ev.name = rec.name;
        ev.domain = rec.user_scope;
        ev.start_ns = rec.cpu_start_ns;
        ev.end_ns = rec.cpu_start_ns + rec.duration_ns;
        ev.duration_ns = rec.duration_ns;
        ev.marker_id = rec.corr_id;
        rt->logger->write(model::NvtxMarkerModel(ev));
    }

    static void handleGraphLaunch(const ActivityRecord& rec, Runtime* rt) {
        GraphLaunchEvent ev;
        ev.pid = detail::GetPid();
        ev.app = rt->app_name;
        ev.session_id = rt->session_id;
        ev.start_ns = rec.cpu_start_ns;
        ev.end_ns = rec.cpu_start_ns + rec.duration_ns;
        ev.duration_ns = rec.duration_ns;
        ev.graph_id = rec.graph_id;
        ev.device_id = rec.device_id;
        ev.stream_id = static_cast<uint32_t>(rec.stream);
        ev.corr_id = rec.corr_id;
        rt->logger->write(model::GraphLaunchEventModel(ev));
    }

    static void handleMemoryAlloc(const ActivityRecord& rec, Runtime* rt) {
        MemoryAllocEventBatchRow row;
        row.start_ns = rec.cpu_start_ns;
        row.duration_ns = rec.duration_ns;
        row.memory_op = rec.memory_op;
        row.memory_kind = rec.memory_kind;
        row.address = rec.address;
        row.bytes = rec.bytes;
        row.device_id = rec.device_id;
        row.stream_id = static_cast<uint32_t>(rec.stream);
        row.corr_id = rec.corr_id;
        g_state.batches.memAllocBatch.push(row);
        if (g_state.batches.memAllocBatch.needsFlush()) {
            g_state.batches.flushAll(*rt->logger, rt->session_id);
        }
    }

    static void handleSynchronization(ActivityRecord& rec) {
        if (auto it = g_state.metadata.syncStackByCorr.find(rec.corr_id);
            it != g_state.metadata.syncStackByCorr.end()) {
            rec.stack_id = it->second;
            g_state.metadata.syncStackByCorr.erase(it);
        }
        SynchronizationEventBatchRow row;
        row.start_ns = rec.cpu_start_ns;
        row.duration_ns = rec.duration_ns;
        row.sync_type = rec.sync_type;
        row.stream_id = static_cast<uint32_t>(rec.stream);
        row.event_id = rec.sync_event_id;
        row.context_id = rec.context_id;
        row.corr_id = rec.corr_id;
        row.function_id = (rec.stack_id != 0) 
            ? g_state.batches.dictManager.internFunction(StackRegistry::instance().get(rec.stack_id))
            : 0;
        g_state.batches.syncBatch.push(row);
    }
};

} // anonymous namespace

// --- Collector Loop ---

void CollectorLoop() {
    GFL_LOG_DEBUG("[CollectorLoop] START");
    // Local (not static): the flush cadence resets per CollectorLoop invocation,
    // so an Initialize->Shutdown->Initialize cycle doesn't inherit a stale
    // timestamp from the previous session.
    auto lastFlush = std::chrono::steady_clock::now();

    while (g_state.collectorRunning.load()) {
        if (!RecordProcessor::processNext()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        if (std::chrono::steady_clock::now() - lastFlush > std::chrono::milliseconds(250)) {
            if (Runtime* rt = runtime(); rt && rt->logger) {
                if (g_state.drainSyntheticMidRun) {
                    constexpr int64_t kMidRunSyntheticGraceNs = 100'000'000;
                    drainSyntheticKernels(rt, detail::GetTimestampNs() - kMidRunSyntheticGraceNs);
                }
                g_state.batches.flushAll(*rt->logger, rt->session_id);
            }
            if (g_state.adapter) g_state.adapter->drainProfilingData();
            lastFlush = std::chrono::steady_clock::now();
        }
    }

    while (RecordProcessor::processNext()) {}

    if (Runtime* rt = runtime(); rt && rt->logger) {
        drainSyntheticKernels(rt);
        g_state.metadata.emitSignatures(rt);
        g_state.batches.flushAll(*rt->logger, rt->session_id, BatchManager::FlushMode::Full);
    }
    GFL_LOG_DEBUG("[CollectorLoop] END");
}

// --- Public Monitor API ---

void Monitor::Initialize(const MonitorOptions& opts) {
    if (g_state.initialized.exchange(true)) return;

    g_monitorBuffer.resetDroppedCount();
    g_state.batches.reset();
    g_state.metadata.reset();
    g_state.batches.dictManager.enable_source_collection = opts.enable_source_collection;
    
    {
        std::lock_guard lk(g_state.batches.scopeBatchMu);
        g_state.openScopeWindows.clear();
        g_state.completedScopeWindows.clear();
    }

    DebugLogger::setEnabled(opts.enable_debug_output);
    g_state.adapter = CreateMonitorAdapter(opts);
    if (g_state.adapter) g_state.adapter->initialize(opts);

    g_state.collectorRunning.store(true);
    g_state.collectorThread = std::thread(CollectorLoop);
}

void Monitor::Shutdown() {
    if (!g_state.initialized.exchange(false)) return;

    if (g_state.adapter) {
        g_state.adapter->stop();
        g_state.adapter->shutdown();
    }

    g_state.collectorRunning.store(false);
    if (g_state.collectorThread.joinable()) g_state.collectorThread.join();

    while (RecordProcessor::processNext()) {}
    if (Runtime* rt = runtime(); rt && rt->logger) {
        drainSyntheticKernels(rt);
        g_state.metadata.emitSignatures(rt);
        g_state.batches.flushAll(*rt->logger, rt->session_id, BatchManager::FlushMode::Full);
    }

    g_monitorBuffer.resetDroppedCount();
    g_state.adapter.reset();
}

void Monitor::Start() { if (g_state.adapter) g_state.adapter->start(); }
void Monitor::Stop() { if (g_state.adapter) g_state.adapter->stop(); }

void Monitor::PushRange(const char* name) {
    void* handle = nullptr;
    RecordStart(name, 0, TraceType::RANGE, &handle);
    g_rangeStack.push(handle);
}

void Monitor::PopRange() {
    if (g_rangeStack.empty()) return;
    void* handle = g_rangeStack.top();
    g_rangeStack.pop();
    RecordStop(handle, 0);
}

void Monitor::RecordStart(const char* name, const StreamHandle stream, const TraceType type, void** outHandle) {
    auto* rec = new ActivityRecord();
    strncpy(rec->name, name, 127);
    rec->type = type;
    rec->stream = stream;
    rec->cpu_start_ns = detail::GetTimestampNs();
    rec->duration_ns = 0;
    rec->has_details = false;
    *outHandle = rec;
}

void Monitor::RecordStop(void* handle, StreamHandle) {
    auto* rec = static_cast<ActivityRecord*>(handle);
    rec->duration_ns = detail::GetTimestampNs() - rec->cpu_start_ns;
    g_monitorBuffer.Push(*rec);
    delete rec;
}

void Monitor::BeginProfilerScope(const char* name) { if (auto* b = GetBackend()) b->OnScopeStart(name); }
void Monitor::EndProfilerScope(const char* name) { if (auto* b = GetBackend()) b->OnScopeStop(name); }
void Monitor::BeginPerfScope(const char* name) { if (auto* b = GetBackend()) b->OnPerfScopeStart(name); }
void Monitor::EndPerfScope(const char* name) { if (auto* b = GetBackend()) b->OnPerfScopeStop(name); }

IMonitorBackend* Monitor::GetBackend() { return g_state.adapter ? g_state.adapter->backend() : nullptr; }
uint32_t Monitor::InternScopeName(const std::string& name) { return g_state.batches.dictManager.internScopeName(name); }
void Monitor::EnqueueCubinForDisassembly(uint64_t crc, const uint8_t* data, size_t size) { g_state.batches.dictManager.enqueueDisassembly(crc, data, size); }
void Monitor::FlushDisassemblyNow() {
    if (Runtime* rt = runtime(); rt && rt->logger) {
        g_state.batches.dictManager.flushDisassembly(*rt->logger, rt->session_id);
    }
}
void Monitor::PushActivityRecord(const ActivityRecord& rec) { g_monitorBuffer.Push(rec); }

void Monitor::PushScopeRow(const ScopeBatchRow& row) {
    if (row.event_type == 0) {
        g_state.batches.activeScopeNameId.store(row.name_id, std::memory_order_relaxed);
    }
    std::lock_guard lk(g_state.batches.scopeBatchMu);
    if (row.event_type == 0) {
        g_state.openScopeWindows[row.scope_instance_id] = {row.ts_ns, row.name_id, row.depth};
    } else {
        if (const auto it = g_state.openScopeWindows.find(row.scope_instance_id); it != g_state.openScopeWindows.end()) {
            g_state.completedScopeWindows.push_back({it->second.start_ns, row.ts_ns, row.scope_instance_id, row.name_id, it->second.depth});
            g_state.openScopeWindows.erase(it);
        }
    }
    g_state.batches.scopeBatch.push(row);
}

void Monitor::PushProfileSamples(const std::vector<ProfileSampleInput>& samples) {
    if (samples.empty()) return;
    const uint32_t scope_name_id = g_state.batches.activeScopeNameId.load(std::memory_order_relaxed);
    std::lock_guard lk(g_state.batches.scopeBatchMu);
    std::unordered_map<std::string, std::string> demangle_cache;
    auto demangledKey = [&demangle_cache](const std::string& key) -> const std::string& {
        auto it = demangle_cache.find(key);
        if (it != demangle_cache.end()) return it->second;
        return demangle_cache.emplace(key, gpufl::core::DemangleFunctionKey(key)).first->second;
    };
    for (const auto& s : samples) {
        ProfileSampleBatchRow row;
        row.ts_ns = s.ts_ns; row.corr_id = s.corr_id; row.device_id = s.device_id;
        const std::string funcSymbol = s.function_key.substr(0, s.function_key.find('@'));
        row.function_id = g_state.batches.dictManager.internFunction(demangledKey(s.function_key), funcSymbol);
        row.pc_offset = s.pc_offset;
        row.metric_id = g_state.batches.dictManager.internMetric(s.metric_name);
        row.metric_value = s.metric_value;
        row.stall_reason = s.stall_reason;
        row.sample_kind = s.sample_kind;
        row.scope_name_id = scope_name_id;
        row.source_file_id = g_state.batches.dictManager.internSourceFile(s.source_file);
        row.source_line = s.source_line;
        g_state.batches.profileBatch.push(row);
    }
}

void Monitor::PushPmSamples(const std::vector<PmSampleInput>& samples) {
    if (samples.empty()) return;
    const uint32_t fallback_id = g_state.batches.activeScopeNameId.load(std::memory_order_relaxed);
    std::lock_guard lk(g_state.batches.scopeBatchMu);
    for (const auto& s : samples) {
        if (s.metric_name.empty()) continue;
        PmSampleBatchRow row;
        row.sample_index = s.sample_index; row.ts_ns = s.ts_ns; row.device_id = s.device_id;
        row.metric_id = g_state.batches.dictManager.internMetric(s.metric_name);
        row.value = s.value;
        row.scope_name_id = g_state.resolveScopeId(s.ts_ns);
        if (row.scope_name_id == 0) row.scope_name_id = fallback_id;
        g_state.batches.pmSampleBatch.push(row);
    }
}

void Monitor::EmitPmSamplingConfig(uint32_t device_id, uint32_t interval_us, uint32_t max_samples, const std::string& preset, const std::vector<std::string>& metrics) {
    const Runtime* rt = runtime();
    if (!(rt && rt->logger)) return;
    PmSamplingConfigEvent ev;
    ev.session_id = rt->session_id; ev.ts_ns = detail::GetTimestampNs();
    ev.device_id = device_id; ev.interval_us = interval_us; ev.max_samples = max_samples;
    ev.preset = preset; ev.metrics = metrics;
    rt->logger->write(model::PmSamplingConfigModel(ev));
}

void SetSuppressOrphanSyntheticKernels(const bool suppress) { g_state.suppressOrphanSyntheticKernels = suppress; }
void SetDrainSyntheticKernelsMidRun(const bool enable) { g_state.drainSyntheticMidRun = enable; }

}  // namespace gpufl

#include "gpufl/core/monitor.hpp"

#include "gpufl/core/deep_window_rules.hpp"

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
#include "gpufl/core/counter_provider.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/logger/logger.hpp"
#include "gpufl/core/model/batch_models.hpp"
#include "gpufl/core/model/memcpy_event_model.hpp"
#include "gpufl/core/model/nvtx_marker_model.hpp"
#include "gpufl/core/model/synchronization_event_model.hpp"
#include "gpufl/core/model/memory_alloc_event_model.hpp"
#include "gpufl/core/model/graph_launch_event_model.hpp"
#include "gpufl/core/model/graph_node_definition_model.hpp"
#include "gpufl/core/model/lifecycle_model.hpp"
#include "gpufl/core/monitor_adapter.hpp"
#include "gpufl/core/monitor_batch_manager.hpp"
#include "gpufl/core/monitor_record_builders.hpp"
#include "gpufl/core/ring_buffer.hpp"
#include "gpufl/core/runtime.hpp"
#include "gpufl/core/segment_runtime.hpp"
#include "gpufl/core/stack_registry.hpp"
#include "gpufl/core/stack_trace.hpp"

namespace gpufl {

// Global ring buffer (as declared in monitor.hpp)
RingBuffer<ActivityRecord, kMonitorBufferSize> g_monitorBuffer;

namespace {
// Set once by Monitor::Initialize; see Monitor::ResolvedProfilingEngine.
std::atomic<ProfilingEngine> g_resolvedProfilingEngine{ProfilingEngine::Monitor};
}  // namespace

namespace {

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
        graphExecByCorr.clear();
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
        const auto segment = rt ? rt->acquireSegmentContext() : nullptr;
        if (execSignatureByScope.empty() || !segment || !segment->logger) return;
        const int64_t ts = detail::GetTimestampNs();
        for (const auto& [scope, kernels] : execSignatureByScope) {
            std::string buf;
            uint64_t launch_count = 0;
            for (const auto& [k, cnt] : kernels) {
                buf += k; buf += '='; buf += std::to_string(cnt); buf += ';';
                launch_count += cnt;
            }
            ExecutionSignatureEvent ev;
            ev.session_id = segment->session_id;
            ev.ts_ns = ts;
            ev.scope_name = scope;
            ev.signature = Fnv1a64(buf);
            ev.launch_count = launch_count;
            ev.distinct_kernels = static_cast<uint32_t>(kernels.size());
            segment->logger->write(model::ExecutionSignatureModel(ev));
        }
        execSignatureByScope.clear();
    }

    void joinLaunchMeta(ActivityRecord& rec) {
        const auto it = launchMetaByCorr.find(rec.corr_id);
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
    std::unordered_map<uint64_t, uint64_t> graphExecByCorr;
    std::map<std::string, std::map<std::string, uint64_t>> execSignatureByScope;
};

struct MonitorState {
    std::unique_ptr<IMonitorAdapter> adapter;
    std::atomic<bool> initialized{false};
    std::thread collectorThread;
    std::atomic<bool> collectorRunning{false};
    // On-demand synchronous drain handshake: an app-thread CUDA cleanup callback
    // bumps drainRequest and waits for the collector to match it in drainAck, so
    // the (synthetic) kernels are written before the Windows-injection exit race.
    std::atomic<uint64_t> drainRequest{0};
    std::atomic<uint64_t> drainAck{0};

    detail::MonitorBatchManager batches;
    MetadataManager metadata;

    // Atomic: set by the backend's start() thread and (since the
    // missing-records suppression) by Monitor::Shutdown while the collector
    // thread may still be reading it at a drain site.
    std::atomic<bool> suppressOrphanSyntheticKernels{false};
    std::atomic<bool> drainSyntheticMidRun{false};
};

MonitorState g_state;
thread_local std::stack<void*> g_rangeStack;
// Application GFL_SCOPE rows are pushed outside the collector thread. Hold
// this only for the final drain/snapshot/publication transaction so an open or
// close cannot slip between the scope snapshot and SegmentContext publish.
std::mutex g_segmentScopeBoundaryMu;

// --- Helper Functions ---

bool isSyntheticNonKernelLaunchName(const char* name) {
    if (!name || name[0] == '\0') return false;
    auto startsWith = [&](const char* prefix) {
        return std::strncmp(name, prefix, std::strlen(prefix)) == 0;
    };
    return startsWith("cudaMemcpy") || startsWith("cuMemcpy") ||
           startsWith("cudaMemset") || startsWith("cuMemset") ||
           startsWith("mem_transfer");
}

void emitKernelRecord(const ActivityRecord& rec, const std::string& stack_trace, Runtime* rt) {
    const uint32_t kernel_id = g_state.batches.internKernel(
        g_state.metadata.demangleKernelName(rec.name));
    const KernelBatchRow row = detail::MakeKernelBatchRow(rec, kernel_id);
    const KernelDetailRow detail = rec.has_details
        ? detail::MakeKernelDetailRow(rec, stack_trace, *rt)
        : KernelDetailRow{};
    const KernelDetailRow* detail_ptr = rec.has_details ? &detail : nullptr;

    if (g_state.batches.pushKernel(row, detail_ptr)) {
        g_state.batches.flushAll();
    }
}

void drainSyntheticKernels(Runtime* rt, int64_t maxApiStartNs = INT64_MAX) {
    auto& metaMap = g_state.metadata.launchMetaByCorr;
    if (metaMap.empty()) return;
    if (g_state.suppressOrphanSyntheticKernels) {
        if (maxApiStartNs == INT64_MAX) metaMap.clear();
        return;
    }
    if (!(rt && rt->hasSegmentContext())) return;
    
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
            case TraceType::KERNEL_META_DISCARD:
                // The CUPTI layer dropped this corr's activity record (invalid
                // timestamps). Take the launch meta with it so
                // drainSyntheticKernels can't resurrect the launch as a
                // synthetic row with host-gap "duration" (a CUDA-graph capture
                // pass is ~2k such drops in one go).
                g_state.metadata.launchMetaByCorr.erase(rec.corr_id);
                return true;
            case TraceType::GRAPH_EXEC_LAUNCH:
                g_state.metadata.graphExecByCorr[rec.corr_id] =
                    rec.graph_exec_key;
                return true;
            default:
                break;
        }

        Runtime* rt = runtime();
        if (!(rt && rt->hasSegmentContext())) return true;

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
            case TraceType::GRAPH_NODE_DEFINITION:
                handleGraphNodeDefinition(rec, rt);
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
            if (const MemcpyBatchRow row = detail::MakeMemcpyBatchRow(rec);
                g_state.batches.pushMemcpy(row)) {
                g_state.batches.flushAll();
            }
        } else { // MEMSET
            const auto segment = rt->acquireSegmentContext();
            if (!segment || !segment->logger) return;
            MemsetEvent be;
            be.platform = g_state.adapter ? g_state.adapter->platformName() : "unknown";
            be.device_id = rec.device_id;
            be.stream_id = static_cast<uint32_t>(rec.stream);
            be.session_id = segment->session_id;
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
            segment->logger->write(model::MemsetEventModel(be));
        }
    }

    static void handleRange(const ActivityRecord& rec) {
        const uint32_t name_id = g_state.batches.internScopeName(rec.name);
        const uint64_t instance_id = g_state.batches.allocateScopeInstanceId();
        ScopeBatchRow begin_row = detail::MakeScopeBatchRow(
            rec.cpu_start_ns, instance_id, name_id, 0, rec.scope_depth);
        ScopeBatchRow end_row = detail::MakeScopeBatchRow(
            rec.cpu_start_ns + rec.duration_ns, instance_id, name_id, 1, rec.scope_depth);
        begin_row.original_start_ns = rec.cpu_start_ns;
        end_row.original_start_ns = rec.cpu_start_ns;

        g_state.batches.pushTraceScopeRows(begin_row, end_row);
    }

    static void handlePcSample(const ActivityRecord& rec, Runtime* rt) {
        uint8_t kind = rec.metric_name[0] != '\0' ||
                    (rec.sample_kind[0] != '\0' && rec.sample_kind[0] == 's') ? 1 : 0;
        const std::string func_key = std::string(rec.function_name) + "@" + rec.source_file;
        const uint32_t function_id = g_state.batches.internFunction(
            g_state.metadata.demangleFunctionKey(func_key), std::string(rec.function_name));
        const std::string metric_key = (rec.metric_name[0] != '\0') ? std::string(rec.metric_name) : rec.reason_name;
        const uint32_t metric_id = g_state.batches.internMetric(metric_key);
        const uint32_t scope_name_id = g_state.batches.activeScopeNameId();
        const uint32_t source_file_id = g_state.batches.internSourceFile(rec.source_file);
        const ProfileSampleBatchRow row = detail::MakeProfileSampleBatchRow(
            rec, kind, function_id, metric_id, scope_name_id, source_file_id);

        if (g_state.batches.pushProfileSample(row)) {
            g_state.batches.flushAll();
        }
    }

    static void handleNvtxMarker(const ActivityRecord& rec, Runtime* rt) {
        const auto segment = rt ? rt->acquireSegmentContext() : nullptr;
        if (!segment || !segment->logger) return;
        NvtxMarkerEvent ev;
        ev.pid = detail::GetPid();
        ev.app = rt->app_name;
        ev.session_id = segment->session_id;
        ev.name = rec.name;
        ev.domain = rec.user_scope;
        ev.start_ns = rec.cpu_start_ns;
        ev.end_ns = rec.cpu_start_ns + rec.duration_ns;
        ev.duration_ns = rec.duration_ns;
        ev.marker_id = rec.corr_id;
        segment->logger->write(model::NvtxMarkerModel(ev));
    }

    static void handleGraphLaunch(const ActivityRecord& rec, Runtime* rt) {
        const auto segment = rt ? rt->acquireSegmentContext() : nullptr;
        if (!segment || !segment->logger) return;
        GraphLaunchEvent ev;
        ev.pid = detail::GetPid();
        ev.app = rt->app_name;
        ev.session_id = segment->session_id;
        ev.start_ns = rec.cpu_start_ns;
        ev.end_ns = rec.cpu_start_ns + rec.duration_ns;
        ev.duration_ns = rec.duration_ns;
        ev.graph_id = rec.graph_id;
        ev.device_id = rec.device_id;
        ev.stream_id = static_cast<uint32_t>(rec.stream);
        ev.corr_id = rec.corr_id;
        if (const auto it = g_state.metadata.graphExecByCorr.find(rec.corr_id);
            it != g_state.metadata.graphExecByCorr.end()) {
            ev.graph_exec_key = it->second;
            g_state.metadata.graphExecByCorr.erase(it);
        }
        segment->logger->write(model::GraphLaunchEventModel(ev));
    }

    static void handleGraphNodeDefinition(const ActivityRecord& rec,
                                          Runtime* rt) {
        const auto segment = rt ? rt->acquireSegmentContext() : nullptr;
        if (!segment || !segment->logger || rec.graph_exec_key == 0) return;
        GraphNodeDefinitionEvent event;
        event.session_id = segment->session_id;
        event.graph_exec_key = rec.graph_exec_key;
        event.node_index = rec.graph_node_index;
        event.node_type = rec.graph_node_type;
        event.dependency_count = rec.graph_node_dependency_count;
        segment->logger->write(model::GraphNodeDefinitionModel(event));
    }

    static void handleMemoryAlloc(const ActivityRecord& rec, Runtime* rt) {
        const MemoryAllocEventBatchRow row = detail::MakeMemoryAllocBatchRow(rec);
        if (g_state.batches.pushMemoryAlloc(row)) {
            g_state.batches.flushAll();
        }
    }

    static void handleSynchronization(ActivityRecord& rec) {
        if (const auto it = g_state.metadata.syncStackByCorr.find(rec.corr_id);
            it != g_state.metadata.syncStackByCorr.end()) {
            rec.stack_id = it->second;
            g_state.metadata.syncStackByCorr.erase(it);
        }
        const uint32_t function_id = (rec.stack_id != 0)
            ? g_state.batches.internFunction(StackRegistry::instance().get(rec.stack_id))
            : 0;
        g_state.batches.pushSynchronization(detail::MakeSynchronizationBatchRow(rec, function_id));
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
        // Serve an on-demand drain request (a CUDA cleanup callback on the app
        // thread, waiting for the kernels to be durable before process exit).
        // Drain the ring fully + emit synthetic kernels + flush, then ack. This
        // is the single-consumer drain done HERE, so the requester never touches
        // the ring. Forces the write even when the collector was CPU-starved
        // during a short busy run (the waiter yielded us the CPU).
        if (const uint64_t req = g_state.drainRequest.load(std::memory_order_acquire);
            req != g_state.drainAck.load(std::memory_order_relaxed)) {
            while (RecordProcessor::processNext()) {}
            if (Runtime* rt = runtime(); rt && rt->hasSegmentContext()) {
                drainSyntheticKernels(rt);
                g_state.metadata.emitSignatures(rt);
                g_state.batches.flushAll(detail::MonitorBatchManager::FlushMode::Full);
            }
            g_state.drainAck.store(req, std::memory_order_release);
        }

        // Every iteration, not on the 250ms flush beat below: this is what
        // decides how closely a deep window tracks its deadline, and it is a
        // lock-free check when no window is closing.
        if (g_state.adapter) g_state.adapter->serviceDeepWindow();
        // Immediately after, so a rule that decides to open is serviced on the
        // very next beat rather than a full loop later.
        detail::DeepWindowRules::Service();

        if (!RecordProcessor::processNext()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        if (std::chrono::steady_clock::now() - lastFlush > std::chrono::milliseconds(250)) {
            if (Runtime* rt = runtime(); rt && rt->hasSegmentContext()) {
                if (g_state.drainSyntheticMidRun) {
                    constexpr int64_t kMidRunSyntheticGraceNs = 100'000'000;
                    drainSyntheticKernels(rt, detail::GetTimestampNs() - kMidRunSyntheticGraceNs);
                }
                g_state.batches.flushAll();
                // AFTER the batch flush, so this beat's data lands in the
                // window being closed. This is the deadline path for the
                // time trigger: without it a channel that wrote once and
                // went quiet keeps its window in `.tmp` until the next
                // write or shutdown.
                if (const auto segment = rt->acquireSegmentContext();
                    segment && segment->logger) {
                    segment->logger->rotateDueWindows();
                }
            }
            if (g_state.adapter) g_state.adapter->drainProfilingData();
            // Boundary arbitration shares this collector beat with CUPTI drain
            // and batch flush, so no second flush thread can race the cutover.
            if (Runtime* rt = runtime(); rt && rt->segment_runtime) {
                rt->segment_runtime->service();
            }
            lastFlush = std::chrono::steady_clock::now();
        }
    }

    while (RecordProcessor::processNext()) {}

    if (Runtime* rt = runtime(); rt && rt->hasSegmentContext()) {
        drainSyntheticKernels(rt);
        g_state.metadata.emitSignatures(rt);
        g_state.batches.flushAll(detail::MonitorBatchManager::FlushMode::Full);
    }
    GFL_LOG_DEBUG("[CollectorLoop] END");
}

// --- Public Monitor API ---

void Monitor::Initialize(const MonitorOptions& opts) {
    if (g_state.initialized.exchange(true)) return;

    // Process-lifetime state must not leak a prior session's capture policy.
    // The active backend's start() replaces these defaults before callbacks
    // begin. Monitor/no-backend sessions keep both disabled.
    SetSuppressOrphanSyntheticKernels(false);
    SetDrainSyntheticKernelsMidRun(false);

    // The engine AFTER env overrides. InitOptions::profiling_engine is the
    // pre-override request, and `gpufl trace --passes X` overrides only the
    // MonitorOptions copy - so anything reporting which engine ran must read
    // it from here, not from g_opts.
    g_resolvedProfilingEngine.store(opts.profiling_engine,
                                    std::memory_order_release);

    g_monitorBuffer.resetDroppedCount();
    // Counter slots live for the process, so this session has to baseline them
    // or it would inherit the previous session's ticks plus anything added
    // while gpufl was shut down.
    detail::ActiveCounterProvider()->begin_session();
    g_state.batches.reset();
    g_state.metadata.reset();
    g_state.batches.setSourceCollectionEnabled(opts.enable_source_collection);
    if (Runtime* rt = runtime(); rt && rt->hasSegmentContext()) {
        g_state.batches.bindFlushRuntime(rt);
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
        // stop() has disabled and flushed activity, so "zero kernel records"
        // is final. When a real-record session lost every record (observed:
        // Trace+PM adaptive on Linux, ~1/4 of NVTX-counter-target runs -
        // enables succeed, sync records flow, kernel records never arrive),
        // the teardown drains below would resurrect every launch meta as a
        // synthetic row whose "duration" is the host gap to the next launch,
        // fabricating a full kernel timeline. Suppress instead: no kernel
        // rows, and the capability event already says
        // enabled_but_no_records, so the session stays self-describing.
        if (IMonitorBackend* b = g_state.adapter->backend();
            b && b->kernelActivityExpectedButMissing()) {
            GFL_LOG_ERROR(
                "[Monitor] Kernel activity was enabled and launches happened, "
                "but no kernel activity record arrived this session. Kernel "
                "rows are omitted (a synthetic fallback would carry host-gap "
                "durations, not kernel time).");
        }
        g_state.adapter->shutdown();
    }

    g_state.collectorRunning.store(false);
    if (g_state.collectorThread.joinable()) g_state.collectorThread.join();

    // AFTER the collector has stopped, and before the logger goes away. Writing
    // the summary while the collector still runs would let the evaluator open a
    // window the recorded summary never mentions.
    // Quality BEFORE Finish: Finish releases the rule session, which destroys
    // the metric source whose discard count the quality event reports. The
    // collector is already stopped, so nothing advances between the two.
    detail::DeepWindowRules::EmitCounterQuality();
    detail::DeepWindowRules::Finish();

    while (RecordProcessor::processNext()) {}
    if (Runtime* rt = runtime(); rt && rt->hasSegmentContext()) {
        drainSyntheticKernels(rt);
        g_state.metadata.emitSignatures(rt);
        g_state.batches.flushAll(detail::MonitorBatchManager::FlushMode::Full);
    }

    g_monitorBuffer.resetDroppedCount();
    g_state.batches.clearFlushSink();
    g_state.adapter.reset();
    // Slots keep their values on purpose: a handle held across this stays
    // valid. What stops those adds counting twice is the baseline the next
    // Initialize takes, not clearing them here.
    detail::ActiveCounterProvider()->end_session();
}

void Monitor::DrainAndFinalizeForExit() {
    if (!g_state.initialized.exchange(false)) return;

    // No missing-records suppression here (unlike Shutdown): this path runs
    // BEFORE the backend's stop(), so activity was never flushed and
    // "zero records seen" may just mean a short run whose only buffer is
    // still pending - suppressing would drop the synthetic fallback that is
    // this path's whole reason to exist.
    // The backend's PC sampling cycle thread stops issuing CUPTI reads as soon
    // as process-exit teardown is flagged (PcSamplingEngine::drainData), so it
    // sits idle here and can't fault mid-flush. We deliberately do NOT join it
    // yet - the join (and the fragile CUPTI release) is deferred to
    // ReleaseBackendForExit, which runs AFTER the logger is closed, so a stuck
    // join can't cost the run's data. The collector stop below is a flag + join;
    // the drain/flush/capabilities read already-captured state.
    g_state.collectorRunning.store(false);
    if (g_state.collectorThread.joinable()) g_state.collectorThread.join();

    // Same ordering as Shutdown: the collector is stopped first, so nothing can
    // advance the rule past what this summary reports.
    // Quality BEFORE Finish: Finish releases the rule session, which destroys
    // the metric source whose discard count the quality event reports. The
    // collector is already stopped, so nothing advances between the two.
    detail::DeepWindowRules::EmitCounterQuality();
    detail::DeepWindowRules::Finish();

    while (RecordProcessor::processNext()) {}
    if (Runtime* rt = runtime(); rt && rt->hasSegmentContext()) {
        drainSyntheticKernels(rt);
        g_state.metadata.emitSignatures(rt);
        g_state.batches.flushAll(detail::MonitorBatchManager::FlushMode::Full);
    }

    // Emit the engine's deferred end-of-session data while the logger is still
    // open - ReleaseBackendForExit (which stops/decodes the engine) runs AFTER
    // logger->close(), so anything produced there is lost. Order matters:
    // emitPendingPerfEvents() decodes the Range Profiler kernel-replay metrics
    // (achieved occupancy etc.) FIRST, so the capability report below reflects
    // the decoded ranges instead of "no data".
    if (g_state.adapter) {
        if (IMonitorBackend* b = g_state.adapter->backend()) {
            b->emitPendingPerfEvents();
            b->emitCapabilities();
        }
    }
    g_state.batches.clearFlushSink();
    // Same close-out as Shutdown(). Without it a process that exits through
    // this path leaves the session marked active, and an embedded host that
    // re-initialised afterwards would inherit its ticks.
    detail::ActiveCounterProvider()->end_session();
}

void Monitor::ReleaseBackendForExit() {
    if (g_state.adapter) {
        g_state.adapter->stop();
        g_state.adapter->shutdown();
    }
    g_monitorBuffer.resetDroppedCount();
    g_state.batches.clearFlushSink();
    g_state.adapter.reset();
}

void Monitor::RequestSyntheticDrainAndWait() {
    // Ask the collector to drain the ring + write the (synthetic, launch-derived)
    // kernels NOW, and wait for it. Called from a CUDA cleanup callback (cudaFree)
    // on the app thread before the Windows-injection exit race. We can't drain
    // here - the ring is single-consumer (the collector). Bumping the request and
    // blocking yields the CPU to the (possibly starved) collector so it runs the
    // drain; bounded so a stuck collector can't hang the app. Does NOT stop the
    // collector, so workloads that free mid-run (PyTorch) keep collecting.
    if (!g_state.collectorRunning.load(std::memory_order_acquire)) return;
    const uint64_t req =
        g_state.drainRequest.fetch_add(1, std::memory_order_acq_rel) + 1;
    for (int i = 0; i < 400 &&
                    g_state.drainAck.load(std::memory_order_acquire) < req; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
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
ProfilingEngine Monitor::ResolvedProfilingEngine() {
    return g_resolvedProfilingEngine.load(std::memory_order_acquire);
}

std::string Monitor::ResolvedProfilingEngineWireName() {
    if (const auto* backend = GetBackend()) {
        const auto selected = backend->SelectedEngineWireName();
        if (!selected.empty()) return selected;
    }
    return ProfilingEngineWireName(ResolvedProfilingEngine());
}

void Monitor::BeginDeepWindowScope(const char* name) { if (auto* b = GetBackend()) b->OnDeepWindowStart(name); }
std::vector<std::string> Monitor::EndDeepWindowScope(const char* name) {
    if (auto* b = GetBackend()) return b->OnDeepWindowStop(name);
    return {};
}
void Monitor::BeginDeepWindowPerfScope(const char* name) { if (auto* b = GetBackend()) b->OnDeepWindowPerfStart(name); }
void Monitor::EndDeepWindowPerfScope(const char* name) { if (auto* b = GetBackend()) b->OnDeepWindowPerfStop(name); }
void Monitor::BeginPerfScope(const char* name) { if (auto* b = GetBackend()) b->OnPerfScopeStart(name); }
void Monitor::EndPerfScope(const char* name) { if (auto* b = GetBackend()) b->OnPerfScopeStop(name); }

IMonitorBackend* Monitor::GetBackend() { return g_state.adapter ? g_state.adapter->backend() : nullptr; }
uint32_t Monitor::InternScopeName(const std::string& name) { return g_state.batches.internScopeName(name); }
void Monitor::EnqueueCubinForDisassembly(uint64_t crc, const uint8_t* data, size_t size) { g_state.batches.enqueueDisassembly(crc, data, size); }
void Monitor::FlushDisassemblyNow() {
    g_state.batches.flushDisassembly();
}
bool Monitor::PushActivityRecord(const ActivityRecord& rec) {
    return g_monitorBuffer.Push(rec);
}

void Monitor::PushScopeRow(const ScopeBatchRow& row) {
    std::lock_guard boundary_lock(g_segmentScopeBoundaryMu);
    g_state.batches.pushTrackedScopeRow(row);
}

uint64_t Monitor::AllocateScopeInstanceId() {
    return g_state.batches.allocateScopeInstanceId();
}

int Monitor::OpenScopeDepth() { return g_state.batches.openScopeDepth(); }

int64_t Monitor::CaptureScopeCloseTimestamp(uint64_t instance_id) {
    return g_state.batches.captureScopeCloseTimestamp(instance_id);
}

void Monitor::MarkScopeClosePending(uint64_t instance_id, int64_t end_ns) {
    g_state.batches.markScopeClosePending(instance_id, end_ns);
}

void Monitor::PushProfileSamples(const std::vector<ProfileSampleInput>& samples) {
    if (samples.empty()) return;
    const uint32_t scope_name_id = g_state.batches.activeScopeNameId();
    std::unordered_map<std::string, std::string> demangle_cache;
    auto demangledKey = [&demangle_cache](const std::string& key) -> const std::string& {
        auto it = demangle_cache.find(key);
        if (it != demangle_cache.end()) return it->second;
        return demangle_cache.emplace(key, gpufl::core::DemangleFunctionKey(key)).first->second;
    };
    std::vector<ProfileSampleBatchRow> rows;
    rows.reserve(samples.size());
    for (const auto& s : samples) {
        ProfileSampleBatchRow row;
        row.ts_ns = s.ts_ns; row.corr_id = s.corr_id; row.device_id = s.device_id;
        const std::string funcSymbol = s.function_key.substr(0, s.function_key.find('@'));
        row.function_id = g_state.batches.internFunction(demangledKey(s.function_key), funcSymbol);
        row.pc_offset = s.pc_offset;
        row.metric_id = g_state.batches.internMetric(s.metric_name);
        row.metric_value = s.metric_value;
        row.stall_reason = s.stall_reason;
        row.sample_kind = s.sample_kind;
        row.scope_name_id = scope_name_id;
        row.source_file_id = g_state.batches.internSourceFile(s.source_file);
        row.source_line = s.source_line;
        rows.push_back(row);
    }
    g_state.batches.pushProfileSamples(rows);
}

void Monitor::PushPmSamples(const std::vector<PmSampleInput>& samples) {
    if (samples.empty()) return;
    std::vector<PmSampleBatchRow> rows;
    rows.reserve(samples.size());
    for (const auto& s : samples) {
        if (s.metric_name.empty()) continue;
        PmSampleBatchRow row;
        row.sample_index = s.sample_index; row.ts_ns = s.ts_ns; row.device_id = s.device_id;
        row.metric_id = g_state.batches.internMetric(s.metric_name);
        row.value = s.value;
        rows.push_back(row);
    }
    g_state.batches.pushPmSamplesResolvingScopes(rows);
}

void Monitor::PublishScopeRetentionWatermark(int64_t ts_ns) {
    g_state.batches.publishScopeRetentionWatermark(ts_ns);
}

void Monitor::BeginPmScopeAttribution(int64_t start_ns) {
    g_state.batches.beginPmScopeAttribution(start_ns);
}

void Monitor::EndPmScopeAttribution() {
    g_state.batches.endPmScopeAttribution();
}

uint64_t Monitor::ScopeAttributionTruncated() {
    return g_state.batches.scopeAttributionTruncated();
}

uint64_t Monitor::PmSampleRowsSeen() {
    return g_state.batches.pmSampleRowsSeen();
}

void Monitor::EmitPmSamplingConfig(uint32_t device_id, uint32_t interval_us, uint32_t max_samples, const std::string& preset, const std::vector<std::string>& metrics) {
    const Runtime* rt = runtime();
    const auto segment = rt ? rt->acquireSegmentContext() : nullptr;
    if (!segment || !segment->logger) return;
    PmSamplingConfigEvent ev;
    ev.session_id = segment->session_id; ev.ts_ns = detail::GetTimestampNs();
    ev.device_id = device_id; ev.interval_us = interval_us; ev.max_samples = max_samples;
    ev.preset = preset; ev.metrics = metrics;
    segment->logger->write(model::PmSamplingConfigModel(ev));
}

void Monitor::FlushForSegmentBoundary() {
    while (RecordProcessor::processNext()) {}
    if (Runtime* rt = runtime(); rt && rt->hasSegmentContext()) {
        if (g_state.drainSyntheticMidRun) {
            constexpr int64_t kMidRunSyntheticGraceNs = 100'000'000;
            drainSyntheticKernels(
                rt, detail::GetTimestampNs() - kMidRunSyntheticGraceNs);
        }
        g_state.metadata.emitSignatures(rt);
        g_state.batches.flushAll(
            detail::MonitorBatchManager::FlushMode::Full);
    }
    if (g_state.adapter) g_state.adapter->drainProfilingData();
    while (RecordProcessor::processNext()) {}
    if (Runtime* rt = runtime(); rt && rt->hasSegmentContext()) {
        g_state.batches.flushAll(
            detail::MonitorBatchManager::FlushMode::Full);
    }
}

void Monitor::FlushSegmentDictionarySnapshot(
    SegmentDictionaryEmitter& emitter, Logger& logger,
    const std::string& session_id) {
    g_state.batches.flushDictionarySnapshot(emitter, logger, session_id);
}

void Monitor::EmitSegmentCaptureCapabilities() {
    if (g_state.adapter) {
        if (IMonitorBackend* backend = g_state.adapter->backend()) {
            backend->emitCapabilities();
        }
    }
}

void Monitor::EmitSegmentMetadata() {
    if (g_state.adapter) {
        if (IMonitorBackend* backend = g_state.adapter->backend()) {
            backend->emitSegmentMetadata();
        }
    }
    // Definitions enter the normal activity ring and are consumed by the
    // collector's next beat. Do not drain here: RecordProcessor has exactly
    // one consumer (CollectorLoop), and this helper is also called by startup
    // and SegmentRuntime threads.
}

bool Monitor::CommitSegmentBoundary(
    const std::function<bool(int64_t,
                             const std::vector<ScopeBatchRow>&,
                             const std::vector<ScopeBatchRow>&)>& commit) {
    if (!commit) return false;
    std::lock_guard boundary_lock(g_segmentScopeBoundaryMu);
    FlushForSegmentBoundary();
    const int64_t boundary_ns = detail::GetTimestampNs();
    auto [closes, opens] =
        g_state.batches.snapshotScopeContinuations(boundary_ns);
    return commit(boundary_ns, closes, opens);
}

void Monitor::WriteScopeRows(Logger& logger, const std::string& session_id,
                             const std::vector<ScopeBatchRow>& rows) {
    g_state.batches.writeScopeRows(logger, session_id, rows);
}

void SetSuppressOrphanSyntheticKernels(const bool suppress) { g_state.suppressOrphanSyntheticKernels = suppress; }
void SetDrainSyntheticKernelsMidRun(const bool enable) { g_state.drainSyntheticMidRun = enable; }
bool SuppressOrphanSyntheticKernelsForTesting() {
    return g_state.suppressOrphanSyntheticKernels.load(std::memory_order_acquire);
}

}  // namespace gpufl

#include "gpufl/core/monitor.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <stack>
#include <thread>
#include <unordered_map>
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
#include "gpufl/core/monitor_adapter.hpp"
#include "gpufl/core/ring_buffer.hpp"
#include "gpufl/core/runtime.hpp"
#include "gpufl/core/stack_registry.hpp"
#include "gpufl/core/stack_trace.hpp"

namespace gpufl {

RingBuffer<ActivityRecord, kMonitorBufferSize> g_monitorBuffer;

static std::unique_ptr<IMonitorAdapter> g_adapter;
static std::atomic g_initialized{false};
static std::thread g_collectorThread;
static std::atomic g_collectorRunning{false};
static thread_local std::stack<void*> g_rangeStack;

// Batch state — kernel/memcpy accessed only from CollectorLoop thread;
// scope/profile may be pushed from any thread (guarded by their own mutex)
static DictionaryManager      g_dictManager;
static BatchBuffer<KernelBatchRow>  g_kernelBatch;
static BatchBuffer<MemcpyBatchRow>  g_memcpyBatch;
static uint64_t g_kernelBatchId = 0;
static uint64_t g_memcpyBatchId = 0;

// Worker-local kernel-name demangle cache (Step 4a). Demangling moved off the
// CUPTI callback/activity threads to here. collectorProcessNext runs only on
// the collector thread (and on the main thread AFTER the collector is joined
// at shutdown), never concurrently — so no lock is needed (the old
// callback-side cache used demangle_mu_).
static std::unordered_map<std::string, std::string> g_kernelNameDemangleCache;
static const std::string& DemangleKernelNameCached(const char* raw) {
    static const std::string kFallback = "kernel_launch";
    if (!raw || raw[0] == '\0') return kFallback;
    auto it = g_kernelNameDemangleCache.find(raw);
    if (it != g_kernelNameDemangleCache.end()) return it->second;
    auto [ins, _] =
        g_kernelNameDemangleCache.emplace(raw, gpufl::core::DemangleName(raw));
    return ins->second;
}

// Collector-thread cache for demangling the "name@source" function keys that
// PC sampling interns. The PC_SAMPLE branch in collectorProcessNext runs only
// on the collector thread (and on the main thread after the collector joins at
// shutdown) — the same single-consumer contract as g_kernelNameDemangleCache,
// so no lock. CUPTI hands PC/SASS function names MANGLED; demangling here gives
// every engine ONE canonical kernel identity (matching Trace's already-
// demangled kernel_dict) that the backend multi-pass merge can join on. NOTE:
// SASS interns on the USER thread (PushProfileSamples) and uses its OWN burst-
// local cache so this map stays single-threaded — do not share it from there.
static std::unordered_map<std::string, std::string> g_functionKeyDemangleCache;
static const std::string& DemangleFunctionKeyCached(const std::string& key) {
    auto it = g_functionKeyDemangleCache.find(key);
    if (it != g_functionKeyDemangleCache.end()) return it->second;
    auto [ins, _] = g_functionKeyDemangleCache.emplace(
        key, gpufl::core::DemangleFunctionKey(key));
    return ins->second;
}
// Deferred kernel detail rows — written after the kernel batch so the
// backend's UPDATE (match by corr_id) always finds the INSERT first.
static std::vector<KernelDetailRow> g_pendingDetails;

// Tracks the most recently begun scope name ID.
// Updated by PushScopeRow (user thread) when a scope begin row is pushed.
static std::atomic<uint32_t> g_activeScopeNameId{0};

struct ScopeWindow {
    int64_t  start_ns = 0;
    int64_t  end_ns   = 0;
    uint64_t instance_id = 0;
    uint32_t name_id = 0;
    int      depth = 0;
};

struct OpenScopeWindow {
    int64_t  start_ns = 0;
    uint32_t name_id = 0;
    int      depth = 0;
};

static std::unordered_map<uint64_t, OpenScopeWindow> g_openScopeWindows;
static std::vector<ScopeWindow> g_completedScopeWindows;

static uint32_t ResolveScopeNameIdForTimestampLocked(const int64_t ts_ns) {
    uint32_t best_id = 0;
    int best_depth = -1;
    int64_t best_start = 0;

    for (auto it = g_completedScopeWindows.rbegin();
         it != g_completedScopeWindows.rend(); ++it) {
        if (ts_ns < it->start_ns || ts_ns > it->end_ns) continue;
        if (it->depth > best_depth ||
            (it->depth == best_depth && it->start_ns >= best_start)) {
            best_id = it->name_id;
            best_depth = it->depth;
            best_start = it->start_ns;
        }
    }
    return best_id;
}

static BatchBuffer<ScopeBatchRow>         g_scopeBatch;
static BatchBuffer<ProfileSampleBatchRow> g_profileBatch;
static BatchBuffer<PmSampleBatchRow>      g_pmSampleBatch;
static uint64_t g_scopeBatchId    = 0;
static uint64_t g_profileBatchId  = 0;
static uint64_t g_pmSampleBatchId = 0;
static std::mutex g_scopeBatchMu;
static std::atomic<uint64_t> g_nextScopeInstanceId{1};

// Per-event sync + memory-alloc emissions used to ship as full
// envelope JSON each time. Now batched + dictionary-encoded
// (sync's stack_trace interns into function_dict). Both buffers
// are written only from the CollectorLoop thread, same threading
// model as g_kernelBatch / g_memcpyBatch.
static BatchBuffer<SynchronizationEventBatchRow> g_syncBatch;
static BatchBuffer<MemoryAllocEventBatchRow>     g_memAllocBatch;
static uint64_t g_syncBatchId     = 0;
static uint64_t g_memAllocBatchId = 0;

static void flushBatches(Logger& logger, const std::string& session_id) {
    // Dictionary MUST be written before any batch that references its
    // name_id / kernel_id / function_id / metric_id entries, otherwise
    // readers see rows that reference undefined dictionary IDs.
    //
    // We call flushDictionary twice: once at the top for source content
    // and disassembly (which also reference dict IDs but are rare), and
    // again immediately before each batch emission. This closes a race
    // where the app thread could intern a new scope name AFTER the
    // initial flushDictionary call but BEFORE the batch flush — the
    // new name would be pushed into g_scopeBatch but remain dirty in
    // the dict, producing an ordering bug where scope_event_batch
    // references name_ids 2-5 before their dict_update is emitted.
    g_dictManager.flushDictionary(logger, session_id);
    g_dictManager.flushSourceContent(logger, session_id);
    g_dictManager.flushDisassembly(logger, session_id);
    if (!g_kernelBatch.empty()) {
        g_dictManager.flushDictionary(logger, session_id);
        logger.write(model::KernelEventBatchModel(
            g_kernelBatch, session_id, ++g_kernelBatchId));
        g_kernelBatch.clear();
        // Write deferred details AFTER their batch so the backend UPDATE
        // always finds the INSERT row that was just written.
        for (const auto& d : g_pendingDetails) {
            logger.write(model::KernelDetailModel(d));
        }
        g_pendingDetails.clear();
    }
    if (!g_memcpyBatch.empty()) {
        g_dictManager.flushDictionary(logger, session_id);
        logger.write(model::MemcpyEventBatchModel(
            g_memcpyBatch, session_id, ++g_memcpyBatchId));
        g_memcpyBatch.clear();
    }
    if (!g_syncBatch.empty()) {
        // Sync rows reference function_dict entries via function_id;
        // dict MUST flush first or backend resolution falls back to
        // "function#<id>" placeholders.
        g_dictManager.flushDictionary(logger, session_id);
        logger.write(model::SynchronizationEventBatchModel(
            g_syncBatch, session_id, ++g_syncBatchId));
        g_syncBatch.clear();
    }
    if (!g_memAllocBatch.empty()) {
        // Pure-numeric rows — no dictionary references.
        logger.write(model::MemoryAllocEventBatchModel(
            g_memAllocBatch, session_id, ++g_memAllocBatchId));
        g_memAllocBatch.clear();
    }
    {
        // Hold g_scopeBatchMu across BOTH the dict flush and the batch
        // write. This blocks app threads in PushScopeRow until we're
        // done. Since app threads always intern a name BEFORE pushing
        // the corresponding row (see ScopedMonitor ctor in gpufl.cpp),
        // any dirty dict entry at the moment we take this lock is
        // guaranteed to correspond either to a row already in
        // g_scopeBatch or to a row that won't be pushed until after we
        // release. Flushing dict here emits exactly the names the
        // outgoing batch references.
        std::lock_guard lk(g_scopeBatchMu);
        if (!g_scopeBatch.empty() || !g_profileBatch.empty() ||
            !g_pmSampleBatch.empty()) {
            g_dictManager.flushDictionary(logger, session_id);
        }
        if (!g_scopeBatch.empty()) {
            logger.write(model::ScopeEventBatchModel(
                g_scopeBatch, session_id, ++g_scopeBatchId));
            g_scopeBatch.clear();
        }
        if (!g_profileBatch.empty()) {
            logger.write(model::ProfileSampleBatchModel(
                g_profileBatch, session_id, ++g_profileBatchId));
            g_profileBatch.clear();
        }
        if (!g_pmSampleBatch.empty()) {
            logger.write(model::PmSampleBatchModel(
                g_pmSampleBatch, session_id, ++g_pmSampleBatchId));
            g_pmSampleBatch.clear();
        }
    }
}

// Worker-local launch-meta join map (Step 4b-2). The corr->meta join that ran
// under CuptiBackend::meta_mu_ on the CUPTI callback + BufferCompleted threads
// now happens here, on the single collector thread, so NO lock is needed (same
// threading model as g_kernelNameDemangleCache: written only by the collector,
// and by the main thread AFTER the collector is joined at shutdown — never
// concurrently). Keyed by CUPTI correlationId. Populated by KERNEL_LAUNCH_META
// records (API_ENTER), api_exit_ns patched by KERNEL_API_EXIT records
// (API_EXIT), consumed + erased when the matching KERNEL / MEMCPY / MEMSET
// activity record is processed. Entries still present at shutdown are launches
// CUPTI never delivered an activity record for; drainSyntheticKernels() flushes
// them as synthetic kernels (replacing CuptiBackend::FlushPendingKernels).
// Stores the whole KERNEL_LAUNCH_META ActivityRecord — it already carries every
// field the join and the synthetic-kernel path need, so no parallel struct.
static std::unordered_map<uint64_t, ActivityRecord> g_launchMetaByCorr;

// Worker-local sync-stack join map (Step 4c). The corr->stack join that ran
// under CuptiBackend::sync_meta_mu_ in BufferCompleted now happens here on the
// single collector thread, lock-free. Populated by SYNC_META records (sync
// API_ENTER), consumed + erased when the matching SYNCHRONIZATION activity
// record is processed. Only the stack_id is carried/joined — the old SyncMeta
// also held an api_enter_ns that nothing downstream ever read, so it's dropped.
// No synthetic drain: sync activity records are 1:1 with the API call, and
// orphans (e.g. sub-100ns syncs filtered in BufferCompleted) are simply dropped,
// same as before; the map is reset per session in Monitor::Initialize.
static std::unordered_map<uint64_t, size_t> g_syncStackByCorr;

// Join launch-callback metadata (scope path, stack id, API timestamps, const-
// bank size) from the worker-local meta map onto an activity record, then erase
// the entry. The activity record supplies everything else (kernel timing / dims
// / occupancy from CUpti_ActivityKernel11; bytes / kind for mem ops). Best-
// effort: if no meta arrived for this corr the record is left untouched — same
// as the old meta_mu_ join missing (e.g. the API_ENTER push lost to ring
// backpressure, or an activity record that arrived before its launch callback).
static void joinLaunchMeta(ActivityRecord& rec) {
    auto it = g_launchMetaByCorr.find(rec.corr_id);
    if (it == g_launchMetaByCorr.end()) return;
    const ActivityRecord& m = it->second;
    rec.scope_depth  = m.scope_depth;
    rec.stack_id     = m.stack_id;
    std::memcpy(rec.user_scope, m.user_scope, sizeof(rec.user_scope));
    rec.api_start_ns = m.api_start_ns;
    rec.api_exit_ns  = m.api_exit_ns;
    rec.const_bytes  = m.const_bytes;
    g_launchMetaByCorr.erase(it);
}

// Emit one fully-joined KERNEL record: intern the (demangled) name, push the
// kernel batch row, and — when detailed — the deferred detail row; flush if the
// batch filled. Extracted from collectorProcessNext's KERNEL branch (Step 4b-2)
// so drainSyntheticKernels can reuse the exact same emission path. `rec` must
// already carry the joined scope/stack metadata and resolved `stack_trace`.
static void emitKernelRecord(const ActivityRecord& rec,
                             const std::string& stack_trace, Runtime* rt) {
    // Demangle on the collector thread (Step 4a) — rec.name holds the RAW
    // mangled name pushed by the CUPTI callback/activity path.
    const uint32_t kernel_id =
        g_dictManager.internKernel(DemangleKernelNameCached(rec.name).c_str());

    KernelBatchRow row;
    row.start_ns    = rec.cpu_start_ns;
    row.kernel_id   = kernel_id;
    row.stream_id   = static_cast<uint32_t>(rec.stream);
    row.duration_ns = rec.duration_ns;
    row.corr_id     = rec.corr_id;
    row.dyn_shared  = rec.dyn_shared;
    row.num_regs    = rec.num_regs;
    row.has_details = rec.has_details ? 1 : 0;
    // Pre-stamped by KernelLaunchHandler; both fields are 0 when no framework
    // was tracking this launch.
    row.external_kind = rec.external_kind;
    row.external_id   = rec.external_id;
    g_kernelBatch.push(row);

    if (rec.has_details) {
        KernelDetailRow detail;
        detail.corr_id    = rec.corr_id;
        detail.session_id = rt->session_id;
        detail.pid        = detail::GetPid();
        detail.app        = rt->app_name;
        detail.grid_x = rec.grid_x; detail.grid_y = rec.grid_y;
        detail.grid_z = rec.grid_z;
        detail.block_x = rec.block_x; detail.block_y = rec.block_y;
        detail.block_z = rec.block_z;
        detail.static_shared         = rec.static_shared;
        detail.local_bytes           = rec.local_bytes;
        detail.const_bytes           = rec.const_bytes;
        detail.occupancy             = rec.occupancy;
        detail.reg_occupancy         = rec.reg_occupancy;
        detail.smem_occupancy        = rec.smem_occupancy;
        detail.warp_occupancy        = rec.warp_occupancy;
        detail.block_occupancy       = rec.block_occupancy;
        std::memcpy(detail.limiting_resource, rec.limiting_resource,
                    sizeof(detail.limiting_resource));
        detail.max_active_blocks      = rec.max_active_blocks;
        detail.local_mem_total        = rec.local_mem_total;
        detail.local_mem_per_thread   = rec.local_mem_per_thread;
        detail.cache_config_requested = rec.cache_config_requested;
        detail.cache_config_executed  = rec.cache_config_executed;
        detail.shared_mem_executed    = rec.shared_mem_executed;
        detail.user_scope  = rec.user_scope;
        detail.stack_trace = stack_trace;
        // Defer: written after the kernel batch by flushBatches().
        g_pendingDetails.push_back(std::move(detail));
    }

    if (g_kernelBatch.needsFlush()) {
        flushBatches(*rt->logger, rt->session_id);
    }
}

// Flush launch metas that never got a CUPTI activity record as synthetic
// kernels (Step 4b-2 — replaces CuptiBackend::FlushPendingKernels). Runs on the
// collector thread AFTER the ring is fully drained, so g_launchMetaByCorr holds
// exactly the orphaned launches (every kernel with an activity record has
// already joined + erased its entry). Common in PC Sampling / SASS safe modes,
// where CONCURRENT_KERNEL activity is off and launch callbacks are the only
// kernel source; empty (no-op) in normal Trace/Deep mode.
//
// CUPTI gave us no GPU timestamps for these, so duration is approximated as the
// gap to the next launch's dispatch (kernels run sequentially on the default
// stream of single-stream workloads) — or `flushNs` for the last one. The
// simplified occupancy was precomputed on the launch callback (it has the
// nvidia-only SM properties this core TU must not reach for); we just carry it.
// Idempotent: clears the map, so the second call from Monitor::Shutdown's
// post-join drain is a no-op.
static void drainSyntheticKernels(Runtime* rt) {
    if (g_launchMetaByCorr.empty()) return;
    if (!(rt && rt->logger)) return;
    const int64_t flushNs = detail::GetTimestampNs();

    std::vector<uint64_t> orderedCorr;
    orderedCorr.reserve(g_launchMetaByCorr.size());
    for (const auto& [corr, _] : g_launchMetaByCorr) orderedCorr.push_back(corr);
    std::sort(orderedCorr.begin(), orderedCorr.end(),
              [&](uint64_t a, uint64_t b) {
                  return g_launchMetaByCorr[a].api_start_ns <
                         g_launchMetaByCorr[b].api_start_ns;
              });

    GFL_LOG_DEBUG("[drainSyntheticKernels] draining ", orderedCorr.size(),
                  " synthetic kernel(s) at flushNs=", flushNs);

    for (size_t i = 0; i < orderedCorr.size(); ++i) {
        const uint64_t corr = orderedCorr[i];
        // Copy the stored meta: it already carries name, device_id, scope,
        // stack, has_details, grid/block/dyn_shared and the precomputed
        // simplified occupancy. We only override the synthetic-specific fields.
        ActivityRecord out = g_launchMetaByCorr[corr];
        out.type = TraceType::KERNEL;
        out.stream = 0;
        out.cpu_start_ns = out.api_start_ns;  // API_ENTER ns
        const int64_t nextEnterNs =
            (i + 1 < orderedCorr.size())
                ? g_launchMetaByCorr[orderedCorr[i + 1]].api_start_ns
                : flushNs;
        int64_t synthDur = nextEnterNs - out.api_start_ns;
        if (synthDur < 0) synthDur = 0;  // clock skew guard
        out.duration_ns = synthDur;
        out.corr_id = static_cast<unsigned>(corr);
        if (out.api_exit_ns <= 0) out.api_exit_ns = flushNs;

        const std::string stack_trace =
            (out.stack_id != 0) ? StackRegistry::instance().get(out.stack_id)
                                : "";
        GFL_LOG_DEBUG("[drainSyntheticKernels] synth corr=", corr,
                      " name=", out.name, " scope=", out.user_scope,
                      " dur=", out.duration_ns, "ns");
        emitKernelRecord(out, stack_trace, rt);
    }
    g_launchMetaByCorr.clear();
}

// Consume + route ONE record from the ring buffer; returns false when empty.
// File-scope (was a lambda inside CollectorLoop) so Monitor::Shutdown can
// re-drain on the MAIN thread after joining the collector. On Windows
// process-exit the collector thread can be torn down before it runs its own
// post-loop drain, stranding records (e.g. the synthetic kernels pushed by
// CuptiBackend::stop()) in the ring buffer. The post-join drain recovers them.
static bool collectorProcessNext() {
        ActivityRecord rec{};
        if (!g_monitorBuffer.Consume(rec)) return false;

        // Launch-meta control records (Step 4b-2): collector-thread-only join-
        // map maintenance, never emitted. Handled before the runtime/logger
        // check — they carry no output, only the join state that later KERNEL /
        // MEMCPY / MEMSET activity records read.
        if (rec.type == TraceType::KERNEL_LAUNCH_META) {
            // Merge into the worker-local join map. Keep-first-unless-upgrade: a
            // single logical launch fires BOTH runtime and driver callbacks with
            // the SAME corr; the first (runtime) wins unless it lacked grid/block
            // details and a later record supplies them. Mirrors the merge the old
            // KernelLaunchHandler::handle did under meta_mu_. (MemTransferHandler
            // metas always have has_details=false, so duplicate-corr mem callbacks
            // resolve to the first/runtime one — its user-facing scope label.)
            auto it = g_launchMetaByCorr.find(rec.corr_id);
            if (it == g_launchMetaByCorr.end()) {
                g_launchMetaByCorr.emplace(rec.corr_id, rec);
            } else if (!it->second.has_details && rec.has_details) {
                it->second = rec;
            }
            return true;
        }
        if (rec.type == TraceType::KERNEL_API_EXIT) {
            if (auto it = g_launchMetaByCorr.find(rec.corr_id);
                it != g_launchMetaByCorr.end()) {
                it->second.api_exit_ns = rec.api_exit_ns;
            }
            return true;
        }
        if (rec.type == TraceType::SYNC_META) {
            // Stash the sync call stack until its SYNCHRONIZATION activity
            // record arrives (Step 4c). Last-write-wins on corr collision (sync
            // corrs are unique per call, so collisions don't occur in practice).
            g_syncStackByCorr[rec.corr_id] = rec.stack_id;
            return true;
        }

        const int64_t duration_ns = rec.duration_ns;
        Runtime* rt = runtime();
        if (!(rt && rt->logger)) {
            GFL_LOG_DEBUG("[CollectorLoop] DROP rec type=", (int)rec.type,
                          " — runtime/logger null");
            return true;
        }
        if (rec.type == TraceType::KERNEL || rec.type == TraceType::MEMCPY ||
            rec.type == TraceType::MEMSET) {
            // Join launch-callback metadata recorded at API_ENTER (Step 4b-2).
            // Before the cutover the producing handler did this under meta_mu_;
            // now the raw activity record arrives with empty scope/stack and we
            // fill it here from the worker-local map (lock-free — single
            // consumer). No-op while the map is empty (scaffolding / cache miss).
            joinLaunchMeta(rec);
            const std::string stack_trace =
                (rec.stack_id != 0) ? StackRegistry::instance().get(rec.stack_id)
                                    : "";
            const char* platform =
                g_adapter ? g_adapter->platformName() : "unknown";

            if (rec.type == TraceType::KERNEL) {
                emitKernelRecord(rec, stack_trace, rt);

            } else if (rec.type == TraceType::MEMCPY) {
                MemcpyBatchRow row;
                row.start_ns    = rec.cpu_start_ns;
                row.stream_id   = static_cast<uint32_t>(rec.stream);
                row.duration_ns = duration_ns;
                row.bytes       = rec.bytes;
                row.copy_kind   = rec.copy_kind;
                row.corr_id     = rec.corr_id;
                g_memcpyBatch.push(row);

                if (g_memcpyBatch.needsFlush()) {
                    flushBatches(*rt->logger, rt->session_id);
                }

            } else {
                // MEMSET — infrequent, keep as immediate verbose write
                MemsetEvent be;
                be.platform    = platform;
                be.device_id   = rec.device_id;
                be.stream_id   = static_cast<uint32_t>(rec.stream);
                be.session_id  = rt->session_id;
                be.pid         = detail::GetPid();
                be.app         = rt->app_name;
                be.name        = rec.name;
                be.start_ns    = rec.cpu_start_ns;
                be.end_ns      = rec.cpu_start_ns + duration_ns;
                be.api_start_ns = rec.api_start_ns;
                be.api_exit_ns  = rec.api_exit_ns;
                be.user_scope  = rec.user_scope;
                be.scope_depth = rec.scope_depth;
                be.corr_id     = rec.corr_id;
                be.stack_trace = stack_trace;
                be.bytes       = rec.bytes;
                rt->logger->write(model::MemsetEventModel(be));
            }
        } else if (rec.type == TraceType::RANGE) {
            const uint32_t name_id =
                g_dictManager.internScopeName(rec.name);
            const uint64_t instance_id =
                g_nextScopeInstanceId.fetch_add(1, std::memory_order_relaxed);

            ScopeBatchRow begin_row;
            begin_row.ts_ns             = rec.cpu_start_ns;
            begin_row.scope_instance_id = instance_id;
            begin_row.name_id           = name_id;
            begin_row.event_type        = 0;  // begin
            begin_row.depth             = rec.scope_depth;

            ScopeBatchRow end_row;
            end_row.ts_ns             = rec.cpu_start_ns + duration_ns;
            end_row.scope_instance_id = instance_id;
            end_row.name_id           = name_id;
            end_row.event_type        = 1;  // end
            end_row.depth             = rec.scope_depth;

            {
                std::lock_guard lk(g_scopeBatchMu);
                g_scopeBatch.push(begin_row);
                g_scopeBatch.push(end_row);
            }
        } else if (rec.type == TraceType::PC_SAMPLE) {
            // Determine sample kind: 0=pc_sampling, 1=sass_metric
            uint8_t kind = 0;
            if (rec.metric_name[0] != '\0') {
                kind = 1;
            } else if (rec.sample_kind[0] != '\0' &&
                       rec.sample_kind[0] == 's') {
                kind = 1;  // "sass_metric"
            }

            // Demangle the name part so PC sampling's function identity matches
            // Trace's already-demangled kernel_dict for the cross-pass merge.
            const std::string func_key =
                std::string(rec.function_name) + "@" + rec.source_file;
            const uint32_t function_id =
                g_dictManager.internFunction(DemangleFunctionKeyCached(func_key));
            // For PC sampling rows metric_name is empty; use reason_name so the
            // stall reason string is interned into metric_dict and reachable via
            // metric_id on the backend.  For SASS rows metric_name is always set.
            const std::string metric_key = (rec.metric_name[0] != '\0')
                ? std::string(rec.metric_name)
                : rec.reason_name;
            const uint32_t metric_id =
                g_dictManager.internMetric(metric_key);
            // Use the most recently begun scope.  PC samples are pushed to the
            // ring buffer by EndPerfScope() (inside ScopedMonitor dtor), which
            // runs after the scope begin row is pushed via PushScopeRow().
            // g_activeScopeNameId is therefore already set to the correct ID
            // by the time the collector loop processes these records.
            const uint32_t scope_name_id =
                g_activeScopeNameId.load(std::memory_order_relaxed);

            const uint32_t source_file_id =
                g_dictManager.internSourceFile(rec.source_file);

            ProfileSampleBatchRow row;
            row.ts_ns           = rec.cpu_start_ns;
            row.corr_id         = rec.corr_id;
            row.device_id       = rec.device_id;
            row.function_id     = function_id;
            row.pc_offset       = rec.pc_offset;
            row.metric_id       = metric_id;
            row.metric_value    = (kind == 1) ? rec.metric_value
                                              : rec.samples_count;
            row.stall_reason    = rec.stall_reason;
            row.sample_kind     = kind;
            row.scope_name_id   = scope_name_id;
            row.source_file_id  = source_file_id;
            row.source_line     = rec.source_line;

            bool needs_flush = false;
            {
                std::lock_guard lk(g_scopeBatchMu);
                g_profileBatch.push(row);
                needs_flush = g_profileBatch.needsFlush();
            }
            if (needs_flush) {
                flushBatches(*rt->logger, rt->session_id);
            }
        } else if (rec.type == TraceType::NVTX_MARKER) {
            // NVTX range captured via CUPTI_ACTIVITY_KIND_MARKER. Emitted
            // directly as a single event (not batched) — NVTX traffic at
            // scale is primarily from PyTorch and framework internals,
            // which we're comfortable serializing per-event for now.
            // Consider batching if volume becomes a problem.
            NvtxMarkerEvent ev;
            ev.pid         = detail::GetPid();
            ev.app         = rt->app_name;
            ev.session_id  = rt->session_id;
            ev.name        = rec.name;
            ev.domain      = rec.user_scope;   // stashed here by cupti_backend
            ev.start_ns    = rec.cpu_start_ns;
            ev.end_ns      = rec.cpu_start_ns + duration_ns;
            ev.duration_ns = duration_ns;
            ev.marker_id   = rec.corr_id;
            rt->logger->write(model::NvtxMarkerModel(ev));
        } else if (rec.type == TraceType::GRAPH_LAUNCH) {
            // cudaGraphLaunch with aggregate timing. Emit per-event;
            // volume is so low (tens to low hundreds per session) that
            // batching would be pointless overhead.
            GraphLaunchEvent ev;
            ev.pid          = detail::GetPid();
            ev.app          = rt->app_name;
            ev.session_id   = rt->session_id;
            ev.start_ns     = rec.cpu_start_ns;
            ev.end_ns       = rec.cpu_start_ns + duration_ns;
            ev.duration_ns  = duration_ns;
            ev.graph_id     = rec.graph_id;
            ev.device_id    = rec.device_id;
            ev.stream_id    = static_cast<uint32_t>(rec.stream);
            ev.corr_id      = rec.corr_id;
            rt->logger->write(model::GraphLaunchEventModel(ev));
        } else if (rec.type == TraceType::MEMORY_ALLOC) {
            // cudaMalloc / cudaFree / cudaMallocAsync / etc.
            // Batched into memory_alloc_event_batch — per-event JSON
            // dropped because the envelope (type/pid/app/session_id)
            // amortizes far better across a 512-row batch. Pure-
            // numeric row → no dictionary lookup needed.
            MemoryAllocEventBatchRow row;
            row.start_ns    = rec.cpu_start_ns;
            row.duration_ns = duration_ns;          // 0 in v1
            row.memory_op   = rec.memory_op;
            row.memory_kind = rec.memory_kind;
            row.address     = rec.address;
            row.bytes       = rec.bytes;
            row.device_id   = rec.device_id;
            row.stream_id   = static_cast<uint32_t>(rec.stream);
            row.corr_id     = rec.corr_id;
            g_memAllocBatch.push(row);
        } else if (rec.type == TraceType::SYNCHRONIZATION) {
            // cudaStreamSynchronize / cudaDeviceSynchronize / etc.
            // Batched into synchronization_event_batch. The user call
            // stack captured by SynchronizationHandler on API_ENTER
            // (PR-B) gets interned via the existing function_dict so
            // hot loops with identical stacks ship the string exactly
            // once via dictionary_update — ~14× wire compression on
            // the canonical "sync inside a loop" workload.
            //
            // Join the API_ENTER call stack here (Step 4c) — moved off the
            // BufferCompleted thread / sync_meta_mu_ onto this single consumer.
            // 1:1 with the API call, so erase on join. Best-effort: a missing
            // SYNC_META (dropped to ring backpressure) leaves stack_id 0.
            if (auto it = g_syncStackByCorr.find(rec.corr_id);
                it != g_syncStackByCorr.end()) {
                rec.stack_id = it->second;
                g_syncStackByCorr.erase(it);
            }
            SynchronizationEventBatchRow row;
            row.start_ns    = rec.cpu_start_ns;
            row.duration_ns = duration_ns;
            row.sync_type   = rec.sync_type;
            row.stream_id   = static_cast<uint32_t>(rec.stream);
            row.event_id    = rec.sync_event_id;
            row.context_id  = rec.context_id;
            row.corr_id     = rec.corr_id;
            if (rec.stack_id != 0) {
                row.function_id = g_dictManager.internFunction(
                        StackRegistry::instance().get(rec.stack_id));
            } else {
                row.function_id = 0;
            }
            g_syncBatch.push(row);
        }

        return true;
}

void CollectorLoop() {
    GFL_LOG_DEBUG("[CollectorLoop] START");
    while (g_collectorRunning.load()) {
        if (!collectorProcessNext()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        static auto lastFlush = std::chrono::steady_clock::now();
        if (std::chrono::steady_clock::now() - lastFlush >
            std::chrono::milliseconds(250)) {
            if (const Runtime* rt = runtime(); rt && rt->logger) {
                flushBatches(*rt->logger, rt->session_id);
            }
            // Periodically drain PC sampling hardware buffer so data from
            // multiple kernels is collected (not just the first kernel).
            if (g_adapter) g_adapter->drainProfilingData();
            lastFlush = std::chrono::steady_clock::now();
        }
    }

    GFL_LOG_DEBUG("[CollectorLoop] while EXIT, running=",
                  g_collectorRunning.load());
    // Drain remaining ring buffer entries
    int drained = 0;
    while (collectorProcessNext()) { ++drained; }
    GFL_LOG_DEBUG("[CollectorLoop] drain-remaining consumed=", drained);

    // Final flush of any partially-filled batches. drainSyntheticKernels runs
    // FIRST (ring now empty → the meta map holds only orphaned launches) so the
    // synthetic kernel rows it emits are included in this last flush.
    if (Runtime* rt = runtime(); rt && rt->logger) {
        drainSyntheticKernels(rt);
        flushBatches(*rt->logger, rt->session_id);
    }
    GFL_LOG_DEBUG("[CollectorLoop] END");
}

void Monitor::Initialize(const MonitorOptions& opts) {
    if (g_initialized.exchange(true)) return;

    // Reset batch state for this session
    g_dictManager.reset();
    g_dictManager.enable_source_collection = opts.enable_source_collection;
    g_kernelBatch.clear();
    g_memcpyBatch.clear();
    // Worker-local launch-meta join map (Step 4b-2): drop any entries a prior
    // session left behind so corr ids can't collide across init/shutdown cycles.
    g_launchMetaByCorr.clear();
    g_syncStackByCorr.clear();  // Step 4c: same cross-session hygiene for syncs.
    g_kernelBatchId  = 0;
    g_memcpyBatchId  = 0;
    g_scopeBatch.clear();
    g_profileBatch.clear();
    g_pmSampleBatch.clear();
    g_scopeBatchId   = 0;
    g_profileBatchId = 0;
    g_pmSampleBatchId = 0;
    g_nextScopeInstanceId.store(1);
    g_activeScopeNameId.store(0);
    {
        std::lock_guard lk(g_scopeBatchMu);
        g_openScopeWindows.clear();
        g_completedScopeWindows.clear();
    }

    DebugLogger::setEnabled(opts.enable_debug_output);
    g_adapter = CreateMonitorAdapter(opts);
    if (g_adapter) g_adapter->initialize(opts);

    g_collectorRunning.store(true);
    g_collectorThread = std::thread(CollectorLoop);
}

void Monitor::Shutdown() {
    if (!g_initialized.exchange(false)) return;

    // Stop and tear down the backend BEFORE joining the collector thread.
    //
    // CuptiBackend::stop() calls cuptiActivityFlushAll(1), which fires
    // BufferCompleted -> pushes activity records to the ring buffer. Some
    // engines also emit final profiling rows from shutdown(); SassMetrics
    // deliberately defers cuptiSassMetricsFlushData until shutdown to avoid
    // mid-run PyTorch deadlocks. If we join the collector first and only then
    // call backend shutdown, those final PC_SAMPLE rows arrive after the
    // collector has drained and never become profile_sample_batch records.
    //
    // Keeping the collector alive until after backend shutdown ensures every
    // final activity/profiling record lands in the ring buffer and is consumed.
    GFL_LOG_DEBUG("Monitor::Shutdown: adapter->stop()");
    if (g_adapter) g_adapter->stop();
    GFL_LOG_DEBUG("Monitor::Shutdown: adapter->shutdown()");
    if (g_adapter) g_adapter->shutdown();

    GFL_LOG_DEBUG("Monitor::Shutdown: join collector thread");
    g_collectorRunning.store(false);
    if (g_collectorThread.joinable()) g_collectorThread.join();

    // Belt-and-suspenders post-join drain. On Windows process-exit the collector
    // thread can be torn down before running its own post-loop drain, stranding
    // late records — notably the synthetic kernels CuptiBackend::stop() pushes
    // during adapter->stop() above — in the ring buffer. Re-drain here on the
    // MAIN thread: it's guaranteed alive, the collector is joined (so we are the
    // sole consumer), and g_adapter + the logger are still valid (reset below).
    // No-op on Linux, where the collector already drained everything itself.
    GFL_LOG_DEBUG("Monitor::Shutdown: collector joined -> post-join drain");
    {
        int drained = 0;
        while (collectorProcessNext()) { ++drained; }
        if (Runtime* rt = runtime(); rt && rt->logger) {
            // Emit any launch metas the collector's own drain didn't reach
            // (Windows process-exit can tear the collector down early). No-op
            // when CollectorLoop already drained them — the map is cleared.
            drainSyntheticKernels(rt);
            flushBatches(*rt->logger, rt->session_id);
        }
        GFL_LOG_DEBUG("Monitor::Shutdown: post-join drained=", drained);
    }

    GFL_LOG_DEBUG("Monitor::Shutdown: adapter.reset()");
    g_adapter.reset();
    GFL_LOG_DEBUG("Monitor::Shutdown: done");
}

void Monitor::Start() {
    if (g_adapter) g_adapter->start();
}

void Monitor::Stop() {
    if (g_adapter) g_adapter->stop();
}

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

void Monitor::RecordStart(const char* name, const StreamHandle stream,
                          const TraceType type, void** outHandle) {
    auto* rec = new ActivityRecord();
    strncpy(rec->name, name, 127);
    rec->type       = type;
    rec->stream     = stream;
    rec->cpu_start_ns = detail::GetTimestampNs();
    rec->duration_ns  = 0;
    rec->has_details  = false;
    *outHandle = rec;
}

void Monitor::RecordStop(void* handle, StreamHandle) {
    auto* rec = static_cast<ActivityRecord*>(handle);
    rec->duration_ns = detail::GetTimestampNs() - rec->cpu_start_ns;
    g_monitorBuffer.Push(*rec);
    delete rec;
}

void Monitor::BeginProfilerScope(const char* name) {
    if (auto* b = GetBackend()) b->OnScopeStart(name);
}

void Monitor::EndProfilerScope(const char* name) {
    if (auto* b = GetBackend()) b->OnScopeStop(name);
}

void Monitor::BeginPerfScope(const char* name) {
    if (auto* b = GetBackend()) b->OnPerfScopeStart(name);
}

void Monitor::EndPerfScope(const char* name) {
    if (auto* b = GetBackend()) b->OnPerfScopeStop(name);
}

IMonitorBackend* Monitor::GetBackend() {
    return g_adapter ? g_adapter->backend() : nullptr;
}

uint32_t Monitor::InternScopeName(const std::string& name) {
    return g_dictManager.internScopeName(name);
}

void Monitor::EnqueueCubinForDisassembly(uint64_t crc, const uint8_t* data,
                                         size_t size) {
    g_dictManager.enqueueDisassembly(crc, data, size);
}

void Monitor::PushActivityRecord(const ActivityRecord& rec) {
    g_monitorBuffer.Push(rec);
}

void Monitor::PushScopeRow(const ScopeBatchRow& row) {
    if (row.event_type == 0) {  // begin: update active scope for PC sample association
        g_activeScopeNameId.store(row.name_id, std::memory_order_relaxed);
    }

    std::lock_guard lk(g_scopeBatchMu);
    if (row.event_type == 0) {
        g_openScopeWindows[row.scope_instance_id] = OpenScopeWindow{
            row.ts_ns, row.name_id, row.depth};
    } else {
        if (const auto it = g_openScopeWindows.find(row.scope_instance_id);
            it != g_openScopeWindows.end()) {
            g_completedScopeWindows.push_back(ScopeWindow{
                it->second.start_ns, row.ts_ns, row.scope_instance_id,
                row.name_id, it->second.depth});
            g_openScopeWindows.erase(it);
        }
    }
    g_scopeBatch.push(row);
}

void Monitor::PushProfileSamples(
    const std::vector<ProfileSampleInput>& samples) {
    if (samples.empty()) return;
    // PC samples (and SASS samples) are scope-attributed via the most
    // recently begun scope.  This thread (the user app thread inside
    // onScopeStop) is the SAME thread that pushed the scope-begin row
    // earlier in ScopedMonitor's ctor, so g_activeScopeNameId is already
    // set to the correct ID for these samples.
    const uint32_t scope_name_id =
        g_activeScopeNameId.load(std::memory_order_relaxed);
    // Single lock acquisition for the entire burst — replaces what was
    // previously thousands of g_monitorBuffer.Push() calls per scope drain.
    // DO NOT call flushBatches() from inside this lock: flushBatches itself
    // takes g_scopeBatchMu (via lock_guard, not recursive), so re-entry
    // would deadlock.  CollectorLoop's 250 ms periodic drain handles flush.
    std::lock_guard lk(g_scopeBatchMu);
    // Burst-local demangle cache (this runs on the USER thread). CUPTI hands
    // SASS function names MANGLED; demangling makes SASS's function identity
    // match Trace's already-demangled kernel_dict for the cross-pass merge.
    // Local (not the collector-thread g_functionKeyDemangleCache) so the maps
    // stay single-threaded and race-free; a burst shares few unique keys across
    // its per-(pc_offset,metric) rows, so one map per drain dedups cheaply.
    std::unordered_map<std::string, std::string> demangle_cache;
    auto demangledKey =
        [&demangle_cache](const std::string& key) -> const std::string& {
        auto it = demangle_cache.find(key);
        if (it != demangle_cache.end()) return it->second;
        return demangle_cache
            .emplace(key, gpufl::core::DemangleFunctionKey(key))
            .first->second;
    };
    for (const auto& s : samples) {
        ProfileSampleBatchRow row;
        row.ts_ns          = s.ts_ns;
        row.corr_id        = s.corr_id;
        row.device_id      = s.device_id;
        row.function_id    = g_dictManager.internFunction(demangledKey(s.function_key));
        row.pc_offset      = s.pc_offset;
        row.metric_id      = g_dictManager.internMetric(s.metric_name);
        row.metric_value   = s.metric_value;
        row.stall_reason   = s.stall_reason;
        row.sample_kind    = s.sample_kind;
        row.scope_name_id  = scope_name_id;
        row.source_file_id = g_dictManager.internSourceFile(s.source_file);
        row.source_line    = s.source_line;
        g_profileBatch.push(row);
    }
}

void Monitor::PushPmSamples(const std::vector<PmSampleInput>& samples) {
    if (samples.empty()) return;
    const uint32_t fallback_scope_name_id =
        g_activeScopeNameId.load(std::memory_order_relaxed);
    std::lock_guard lk(g_scopeBatchMu);
    for (const auto& s : samples) {
        if (s.metric_name.empty()) continue;
        PmSampleBatchRow row;
        row.sample_index = s.sample_index;
        row.ts_ns = s.ts_ns;
        row.device_id = s.device_id;
        row.metric_id = g_dictManager.internMetric(s.metric_name);
        row.value = s.value;
        row.scope_name_id = ResolveScopeNameIdForTimestampLocked(s.ts_ns);
        if (row.scope_name_id == 0) row.scope_name_id = fallback_scope_name_id;
        g_pmSampleBatch.push(row);
    }
}

void Monitor::EmitPmSamplingConfig(uint32_t device_id,
                                   uint32_t interval_us,
                                   uint32_t max_samples,
                                   const std::string& preset,
                                   const std::vector<std::string>& metrics) {
    Runtime* rt = runtime();
    if (!(rt && rt->logger)) return;
    PmSamplingConfigEvent ev;
    ev.session_id = rt->session_id;
    ev.ts_ns = detail::GetTimestampNs();
    ev.device_id = device_id;
    ev.interval_us = interval_us;
    ev.max_samples = max_samples;
    ev.preset = preset;
    ev.metrics = metrics;
    rt->logger->write(model::PmSamplingConfigModel(ev));
}

}  // namespace gpufl

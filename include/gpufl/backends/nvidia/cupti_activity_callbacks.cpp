#include "gpufl/backends/nvidia/cupti_backend.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "gpufl/backends/nvidia/cupti_activity_state.hpp"
#include "gpufl/backends/nvidia/cupti_runtime_support.hpp"
#include "gpufl/core/activity_record.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/monitor.hpp"
#include "gpufl/core/trace_type.hpp"

namespace gpufl {

void CUPTIAPI CuptiBackend::BufferRequested(uint8_t** buffer, size_t* size,
                                            size_t* maxNumRecords) {
    *size = 64 * 1024;
    *buffer = static_cast<uint8_t*>(malloc(*size));
    *maxNumRecords = 0;
}

void CUPTIAPI CuptiBackend::BufferCompleted(CUcontext context,
                                            uint32_t streamId, uint8_t* buffer,
                                            size_t size,
                                            const size_t validSize) {
    auto* backend = GetActiveCuptiBackend();
    if (!backend) {
        DebugLogger::error("[CUPTI] ",
                           "BufferCompleted: No active backend!");
        if (buffer) free(buffer);
        return;
    }

    // Per-session clock anchor, captured in start(). Defensive lazy-init in
    // case an activity record somehow arrives before start() set it.
    if (backend->base_cupti_ts_ == 0) {
        backend->base_cpu_ns_ = detail::GetTimestampNs();
        cuptiGetTimestamp(&backend->base_cupti_ts_);
    }
    const int64_t baseCpuNs = backend->base_cpu_ns_;
    const uint64_t baseCuptiTs = backend->base_cupti_ts_;

    // handlers_ is immutable after initialize() (see its declaration) - iterate
    // it directly: no handler_mu_, no per-buffer vector copy.
    const auto& handlers = backend->handlers_;

    if (validSize > 0) {
        // ----------------------------------------------------------------
        // Two-pass dispatch.
        //
        // Within a single CUPTI buffer flush, KERNEL records and
        // EXTERNAL_CORRELATION records arrive interleaved. The handler
        // chain stamps a kernel's external_kind/external_id by looking
        // up its correlationId in g_extCorrMap - but if the matching
        // EXTERNAL_CORRELATION record is later in the same buffer and
        // hasn't been processed yet, the lookup misses and the kernel
        // ships with no framework attribution.
        //
        // Fix: walk the buffer twice. The first pass touches ONLY
        // EXTERNAL_CORRELATION records, populating g_extCorrMap so
        // every entry from this buffer is in the map before any
        // kernel is dispatched. The second pass runs the full handler
        // chain plus the fall-through cases (skipping EXTERNAL_CORRELATION
        // since it's already processed).
        //
        // cuptiActivityGetNextRecord uses the `record` pointer as
        // iteration state - passing nullptr starts a fresh walk from
        // the beginning of the buffer, so calling it twice with a
        // reset pointer is the correct CUPTI idiom.
        // ----------------------------------------------------------------

        // ---- Pass 1: collect EXTERNAL_CORRELATION into g_extCorrMap ----
        {
            CUpti_Activity* record = nullptr;
            while (true) {
                const CUptiResult st =
                    cuptiActivityGetNextRecord(buffer, validSize, &record);
                if (st == CUPTI_ERROR_MAX_LIMIT_REACHED) break;
                if (st != CUPTI_SUCCESS) break;
                if (record->kind !=
                    CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION) {
                    continue;
                }
                auto* ec = reinterpret_cast<
                    const CUpti_ActivityExternalCorrelation*>(record);
                StoreExternalCorrelation(
                    ec->correlationId,
                    static_cast<uint8_t>(ec->externalKind),
                    ec->externalId);
                backend->external_correlation_seen_.fetch_add(
                    1, std::memory_order_relaxed);
                static std::atomic g_ec_count{0};
                const int n = g_ec_count.fetch_add(
                    1, std::memory_order_relaxed) + 1;
                if (n <= 5 || n % 100 == 0) {
                    GFL_LOG_DEBUG(
                        "[CUPTI] EXTERNAL_CORRELATION #", n,
                        " corr_id=", ec->correlationId,
                        " kind=", static_cast<int>(ec->externalKind),
                        " ext_id=", ec->externalId);
                }
            }
        }

        // ---- Pass 2: full handler + fall-through dispatch ----
        CUpti_Activity* record = nullptr;
        while (true) {
            const CUptiResult st =
                cuptiActivityGetNextRecord(buffer, validSize, &record);
            if (st == CUPTI_SUCCESS) {
                // Skip EXTERNAL_CORRELATION - already stored by pass 1.
                if (record->kind ==
                    CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION) {
                    continue;
                }
                bool handled = false;
                for (const auto& h : handlers) {
                    if (h->handleActivityRecord(record, baseCpuNs,
                                                baseCuptiTs)) {
                        handled = true;
                        break;
                    }
                }
                if (!handled) {
                    if (record->kind ==
                        CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR) {
                        auto* sl = reinterpret_cast<
                            const CUpti_ActivitySourceLocator*>(record);
                        if (sl->fileName) {
                            StoreSourceLocator(sl->id, sl->fileName,
                                               sl->lineNumber);
                            backend->source_locator_seen_.fetch_add(
                                1, std::memory_order_relaxed);
                        }
                    } else if (record->kind == CUPTI_ACTIVITY_KIND_FUNCTION) {
                        auto* fn = reinterpret_cast<
                            const CUpti_ActivityFunction*>(record);
                        if (fn->name) {
                            StoreFunctionName(fn->id, fn->name);
                            backend->function_record_seen_.fetch_add(
                                1, std::memory_order_relaxed);
                        }
                    } else if (record->kind ==
                               CUPTI_ACTIVITY_KIND_PC_SAMPLING) {
                        auto* pc = reinterpret_cast<
                            CUpti_ActivityPCSampling3*>(record);
                        ActivityRecord out{};
                        out.type = TraceType::PC_SAMPLE;
                        out.corr_id = pc->correlationId;
                        out.pc_offset =
                            static_cast<uint32_t>(pc->pcOffset);
                        std::snprintf(out.sample_kind,
                                      sizeof(out.sample_kind), "%s",
                                      "pc_sampling");
                        out.samples_count = pc->samples;
                        out.stall_reason = pc->stallReason;
                        out.device_id = backend->device_id_;
                        {
                            std::string sourceFile;
                            uint32_t sourceLine = 0;
                            if (LookupSourceLocator(pc->sourceLocatorId,
                                                    &sourceFile, &sourceLine)) {
                                std::snprintf(
                                    out.source_file,
                                    sizeof(out.source_file), "%s",
                                    sourceFile.c_str());
                                out.source_line = sourceLine;
                            }
                            std::string functionName;
                            if (LookupFunctionName(pc->functionId,
                                                   &functionName)) {
                                std::snprintf(
                                    out.function_name,
                                    sizeof(out.function_name), "%s",
                                    functionName.c_str());
                            }
                        }
                        g_monitorBuffer.Push(out);
                    } else if (record->kind == CUPTI_ACTIVITY_KIND_MARKER) {
                        // NVTX markers arrive as paired START/END records.
                        // Pair them by id to emit one ActivityRecord per
                        // completed range (TraceType::NVTX_MARKER, consumed
                        // by CollectorLoop -> NvtxMarkerModel JSON).
                        auto* m = reinterpret_cast<
                            const CUpti_ActivityMarker2*>(record);
                        const bool isStart =
                            (m->flags & CUPTI_ACTIVITY_FLAG_MARKER_START) != 0;
                        const bool isEnd =
                            (m->flags & CUPTI_ACTIVITY_FLAG_MARKER_END) != 0;

                        if (isStart) {
                            StoreNvtxMarkerStart(
                                m->id,
                                m->name ? m->name : "",
                                m->domain ? m->domain : "",
                                m->timestamp);
                        } else if (isEnd) {
                            NvtxOpenRange entry;
                            if (PopNvtxMarker(m->id, &entry)) {
                                ActivityRecord out{};
                                out.type = TraceType::NVTX_MARKER;
                                std::snprintf(out.name, sizeof(out.name),
                                              "%s", entry.name.c_str());
                                // Convert CUPTI timestamp (ns, monotonic
                                // but different epoch) to wall-clock ns
                                // using the same base delta other records
                                // use elsewhere in this callback.
                                const int64_t start_wall =
                                    static_cast<int64_t>(entry.start_ts) -
                                    static_cast<int64_t>(baseCuptiTs) +
                                    baseCpuNs;
                                const int64_t end_wall =
                                    static_cast<int64_t>(m->timestamp) -
                                    static_cast<int64_t>(baseCuptiTs) +
                                    baseCpuNs;
                                out.cpu_start_ns = start_wall;
                                out.duration_ns = end_wall - start_wall;
                                out.corr_id = m->id;
                                // Domain stored in user_scope slot for now
                                // (CollectorLoop passes it to the event).
                                std::snprintf(out.user_scope,
                                              sizeof(out.user_scope), "%s",
                                              entry.domain.c_str());
                                g_monitorBuffer.Push(out);
                                backend->nvtx_marker_emitted_.fetch_add(
                                    1, std::memory_order_relaxed);
                            }
                        }
                        // Other flag values (e.g. SYNC-only points) are
                        // ignored in v1; can be added as instantaneous
                        // events later if needed.
                    } else if (record->kind ==
                               CUPTI_ACTIVITY_KIND_GRAPH_TRACE) {
                        // F4: cudaGraphLaunch with aggregate timing.
                        // CUPTI gives one record per launch. start/end
                        // are in CUPTI's clock domain - convert to
                        // wall using the same baseCpuNs/baseCuptiTs
                        // delta the rest of BufferCompleted uses.
                        // start == end == 0 is a valid CUPTI signal
                        // for "couldn't collect timing"; we honor it
                        // by emitting duration=0 rather than dropping
                        // the row (the graph_id is still useful
                        // attribution).
                        auto* g = reinterpret_cast<
                            const CUpti_ActivityGraphTrace2*>(record);
                        int64_t start_wall = 0;
                        int64_t dur = 0;
                        if (g->start != 0 || g->end != 0) {
                            start_wall = static_cast<int64_t>(g->start) -
                                         static_cast<int64_t>(baseCuptiTs) +
                                         baseCpuNs;
                            const int64_t end_wall =
                                static_cast<int64_t>(g->end) -
                                static_cast<int64_t>(baseCuptiTs) + baseCpuNs;
                            dur = end_wall - start_wall;
                            if (dur < 0) dur = 0;  // clock-skew guard
                        }

                        ActivityRecord out{};
                        out.type = TraceType::GRAPH_LAUNCH;
                        out.cpu_start_ns = start_wall;
                        out.duration_ns = dur;
                        out.device_id = g->deviceId;
                        out.stream = g->streamId;
                        out.corr_id = g->correlationId;
                        out.graph_id = g->graphId;
                        g_monitorBuffer.Push(out);
                        backend->graph_activity_emitted_.fetch_add(
                            1, std::memory_order_relaxed);
                    } else if (record->kind ==
                               CUPTI_ACTIVITY_KIND_MEMORY2) {
                        // F3: cudaMalloc / cudaFree / cudaMallocAsync /
                        // cudaMallocManaged / cudaMallocHost. CUPTI's
                        // CUpti_ActivityMemory4 carries one timestamp
                        // (the host call ts) but no end timestamp -
                        // duration_ns is left at 0 in v1; if users
                        // need host-side cost we'd correlate against
                        // the matching cuptiActivity API record (DEFER).
                        auto* m = reinterpret_cast<
                            const CUpti_ActivityMemory4*>(record);
                        const int64_t ts_wall =
                            static_cast<int64_t>(m->timestamp) -
                            static_cast<int64_t>(baseCuptiTs) + baseCpuNs;

                        ActivityRecord out{};
                        out.type = TraceType::MEMORY_ALLOC;
                        out.cpu_start_ns = ts_wall;
                        out.duration_ns = 0;
                        out.bytes = m->bytes;
                        out.address = m->address;
                        out.memory_op = static_cast<uint8_t>(m->memoryOperationType);
                        out.memory_kind = static_cast<uint8_t>(m->memoryKind);
                        out.device_id = m->deviceId;
                        out.stream = m->streamId;
                        out.corr_id = m->correlationId;
                        g_monitorBuffer.Push(out);
                        backend->memory_activity_emitted_.fetch_add(
                            1, std::memory_order_relaxed);
                    } else if (record->kind ==
                               CUPTI_ACTIVITY_KIND_SYNCHRONIZATION) {
                        // F2: cudaStreamSynchronize / cudaDeviceSynchronize
                        // / cudaEventSynchronize / cuStreamWaitEvent timing.
                        // CUPTI delivers exactly one record per call, with
                        // wall-clock start/end already converted to ns.
                        // We push directly to the monitor ring buffer -
                        // CollectorLoop translates the ActivityRecord
                        // into a SynchronizationEvent and emits the JSON.
                        auto* s = reinterpret_cast<
                            const CUpti_ActivitySynchronization*>(record);

                        // Filter: drop sub-100ns syncs. CUPTI sometimes
                        // emits zero-duration spurious records on the
                        // CUDA driver's internal paths (idle wait, fast-
                        // path early-return). They're noise in the data
                        // and would dominate counts on pathological
                        // workloads. Documented threshold from the F2
                        // plan; expose as an init option later if needed.
                        const int64_t start_wall =
                            static_cast<int64_t>(s->start) -
                            static_cast<int64_t>(baseCuptiTs) + baseCpuNs;
                        const int64_t end_wall =
                            static_cast<int64_t>(s->end) -
                            static_cast<int64_t>(baseCuptiTs) + baseCpuNs;
                        const int64_t dur = end_wall - start_wall;
                        if (dur < 100) {
                            continue;
                        }

                        ActivityRecord out{};
                        out.type = TraceType::SYNCHRONIZATION;
                        out.cpu_start_ns = start_wall;
                        out.duration_ns = dur;
                        out.corr_id = s->correlationId;
                        out.stream = s->streamId;
                        out.sync_type = static_cast<uint8_t>(s->type);
                        out.sync_event_id = s->cudaEventId;
                        out.context_id = s->contextId;
                        // The user call stack captured by SynchronizationHandler
                        // at API_ENTER is joined on the collector thread now
                        // (g_syncStackByCorr in monitor.cpp, keyed by corr_id) -
                        // no sync_meta_mu_ here. out.stack_id stays 0; the
                        // collector fills it. (Step 4c.)
                        g_monitorBuffer.Push(out);
                        backend->sync_activity_emitted_.fetch_add(
                            1, std::memory_order_relaxed);
                    }
                    // (EXTERNAL_CORRELATION handled in pass 1 above -
                    //  the early `continue` at the top of this loop
                    //  ensures we never reach the fall-through chain
                    //  for that kind.)
                }
            } else if (st == CUPTI_ERROR_MAX_LIMIT_REACHED) {
                break;
            } else {
                ::gpufl::DebugLogger::error("[CUPTI] ",
                                            "Error parsing buffer: ", st);
                break;
            }
        }
    }

    free(buffer);
}

void CuptiBackend::GflCallback(void* userdata, CUpti_CallbackDomain domain,
                               CUpti_CallbackId cbid, const void* cbdata) {
    if (!cbdata) return;

    auto* backend = static_cast<CuptiBackend*>(userdata);
    if (!backend) return;

    // handlers_ is immutable after initialize() (see its declaration) - iterate
    // it directly on this per-callback hot path: no handler_mu_, no copy/alloc.
    const auto& handlers = backend->handlers_;

    bool apiHandled = false;

    for (const auto& handler : handlers) {
        if (handler->shouldHandle(domain, cbid)) {
            if (domain == CUPTI_CB_DOMAIN_RUNTIME_API ||
                domain == CUPTI_CB_DOMAIN_DRIVER_API) {
                if (apiHandled) continue;
                apiHandled = true;
            }
            handler->handle(domain, cbid, cbdata);
        }
    }
}

}  // namespace gpufl

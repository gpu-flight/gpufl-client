#include "gpufl/backends/nvidia/kernel_launch_handler.hpp"

#include <algorithm>  // std::min(initializer_list) — see occupancy calc below
#include <cstdio>
#include <cstdlib>
#include <cstring>    // strnlen — bounded read in cachedDemangle
#include <iterator>   // std::begin / std::end on the user_scope C-array
#include <set>
#include <string>
#if defined(__linux__)
#include <sys/uio.h>
#include <unistd.h>
#endif

#include "gpufl/backends/nvidia/cuda_feature_guards.hpp"
#include "gpufl/backends/nvidia/cupti_utils.hpp"
#include "gpufl/core/activity_record.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/monitor.hpp"
#include "gpufl/core/ring_buffer.hpp"
#include "gpufl/core/scope_registry.hpp"
#include "gpufl/core/stack_registry.hpp"
#include "gpufl/core/stack_trace.hpp"

using gpufl::core::DemangleName;

namespace gpufl {

namespace {

bool CopyReadableCString(const char* src, std::string* out) {
    if (!src || !out) return false;

#if defined(__linux__)
    constexpr size_t kMaxNameLen = 4096;
    char buf[kMaxNameLen] = {};

    iovec local{};
    local.iov_base = buf;
    local.iov_len = sizeof(buf);

    iovec remote{};
    remote.iov_base = const_cast<char*>(src);
    remote.iov_len = sizeof(buf);

    const ssize_t nread = process_vm_readv(getpid(), &local, 1, &remote, 1, 0);
    if (nread <= 0) return false;

    const size_t n = static_cast<size_t>(nread);
    const void* nul = std::memchr(buf, '\0', n);
    if (!nul) return false;

    const size_t len = static_cast<const char*>(nul) - buf;
    if (len == 0) return false;

    out->assign(buf, len);
    return true;
#else
    (void)src;
    (void)out;
    return false;
#endif
}

}  // namespace

KernelLaunchHandler::KernelLaunchHandler(CuptiBackend* backend)
    : backend_(backend) {}

const std::string& KernelLaunchHandler::cachedDemangle(const char* mangled) {
    static const std::string kFallback = "kernel_launch";
    if (!mangled) return kFallback;

    // Defensive bounded read. On CUDA 13.1 + PyTorch autograd's backward
    // worker threads, a SEGV was observed deep inside cachedDemangle on
    // an RTX 3090 — no DemangleName frame on the stack, which means the
    // implicit `std::string(mangled)` construction in find()/emplace()
    // crashed. That can happen if `cbInfo->symbolName` (or the activity
    // record's `name` field) points to a buffer that isn't reliably
    // null-terminated under some CUPTI codepath. strnlen short-circuits
    // at kMaxNameLen so a non-null-terminated pointer reads at most one
    // page of bogus memory instead of running off the end of mapping —
    // and if it never finds a terminator we fall back to a known-safe
    // string rather than letting std::string crash. PyTorch's longest
    // mangled cuBLAS / cutlass names top out around 350 chars in
    // practice; 4 KiB gives generous headroom.
    constexpr size_t kMaxNameLen = 4096;
    if (const size_t len = strnlen(mangled, kMaxNameLen);
        len == 0 || len == kMaxNameLen) return kFallback;

    std::lock_guard lk(demangle_mu_);
    if (const auto it = demangle_cache_.find(mangled); it != demangle_cache_.end()) return it->second;
    // Try-catch around the insert as belt-and-suspenders. If DemangleName
    // itself throws (it shouldn't — abi::__cxa_demangle returns a status
    // code and we check it — but defensive on platforms with unusual
    // allocator behavior), we return the fallback rather than letting
    // the exception unwind across a CUPTI callback boundary, which is
    // undefined behavior in CUPTI's callback contract.
    try {
        auto [inserted, _] = demangle_cache_.emplace(mangled, DemangleName(mangled));
        return inserted->second;
    } catch (...) {
        return kFallback;
    }
}

std::vector<std::pair<CUpti_CallbackDomain, CUpti_CallbackId>>
KernelLaunchHandler::requiredCallbacks() const {
    if (backend_ && backend_->SassMetricsOnlyMode()) {
        GFL_LOG_DEBUG(
            "[KernelLaunchHandler] launch API callbacks disabled in SASS "
            "metrics-only mode. Set GPUFL_SASS_ALLOW_ACTIVITY_WITH_METRICS=1 "
            "to test CUPTI activity/callback coexistence.");
        return {};
    }
    if (backend_ && backend_->IsSassProfilerMode()) {
        GFL_LOG_DEBUG(
            "[KernelLaunchHandler] launch API callbacks enabled in SASS "
            "profiler mode for synthetic kernel rows; CUPTI kernel activity "
            "remains controlled by the SASS activity policy.");
    }

    // The set of launch APIs that produce a kernel record we want
    // api_enter_ns / api_exit_ns for. Anything missing here means the
    // backend stores 0 for the API timestamps, the frontend can't
    // compute cpu_overhead / queue_latency, and the kernel detail
    // page falls back to "—" for those metrics. Add new launch CBIDs
    // here as CUPTI exposes them — shouldHandle() derives its filter from
    // this list, so there's only one place to update.
    //
    // The GPUFL_HAS_* feature gates (see cuda_feature_guards.hpp) keep the
    // agent buildable against older CUDA toolkits — the CBID enumerator
    // isn't declared if CUPTI predates the API.
    std::vector<std::pair<CUpti_CallbackDomain, CUpti_CallbackId>> cbs = {
        // ── Runtime API ───────────────────────────────────────────
        {CUPTI_CB_DOMAIN_RUNTIME_API,
         CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020},
        {CUPTI_CB_DOMAIN_RUNTIME_API,
         CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000},
        // Per-thread default stream variants (CUDA 7.0+) — used when
        // the program is compiled with --default-stream per-thread or
        // CUDA_API_PER_THREAD_DEFAULT_STREAM is defined.
        {CUPTI_CB_DOMAIN_RUNTIME_API,
         CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_ptsz_v7000},
        {CUPTI_CB_DOMAIN_RUNTIME_API,
         CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000},
        // ── Driver API ────────────────────────────────────────────
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunch},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchGrid},
        {CUPTI_CB_DOMAIN_DRIVER_API,
         CUPTI_DRIVER_TRACE_CBID_cuLaunchGridAsync},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel},
        {CUPTI_CB_DOMAIN_DRIVER_API,
         CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz},
    };

    // Cooperative kernel launches (CUDA 9.0+) — used by NCCL
    // collectives and any kernel that needs grid-wide
    // synchronization via cooperative_groups.
#if GPUFL_HAS_COOPERATIVE_LAUNCH
    cbs.push_back({CUPTI_CB_DOMAIN_RUNTIME_API,
                   CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000});
    cbs.push_back({CUPTI_CB_DOMAIN_RUNTIME_API,
                   CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_ptsz_v9000});
    cbs.push_back({CUPTI_CB_DOMAIN_RUNTIME_API,
                   CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernelMultiDevice_v9000});
    cbs.push_back({CUPTI_CB_DOMAIN_DRIVER_API,
                   CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel});
    cbs.push_back({CUPTI_CB_DOMAIN_DRIVER_API,
                   CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz});
    cbs.push_back({CUPTI_CB_DOMAIN_DRIVER_API,
                   CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice});
#endif

    // Modern extensible launch (CUDA 11.6+) — what PyTorch ≥ 1.13,
    // CUTLASS ≥ 2.10, and most new CUDA samples emit. Was the
    // primary cause of api_start_ns / api_exit_ns showing up as 0
    // for users on recent toolkits before this CBID was wired up.
#if GPUFL_HAS_EXTENSIBLE_LAUNCH
    cbs.push_back({CUPTI_CB_DOMAIN_RUNTIME_API,
                   CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060});
    cbs.push_back({CUPTI_CB_DOMAIN_RUNTIME_API,
                   CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_ptsz_v11060});
    cbs.push_back({CUPTI_CB_DOMAIN_DRIVER_API,
                   CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx});
    cbs.push_back({CUPTI_CB_DOMAIN_DRIVER_API,
                   CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz});
#endif

    return cbs;
}

std::vector<CUpti_ActivityKind> KernelLaunchHandler::requiredActivityKinds()
    const {
    if (backend_ && backend_->GetOptions().profiling_engine ==
                        ProfilingEngine::PcSampling) {
        GFL_LOG_DEBUG(
            "[KernelLaunchHandler] CUPTI kernel activity disabled in PC "
            "Sampling mode; launch callbacks will provide synthetic kernel "
            "rows.");
        return {};
    }

    if (backend_ && !backend_->AllowSassKernelActivity()) {
        GFL_LOG_DEBUG(
            "[KernelLaunchHandler] CUPTI kernel activity disabled in SASS "
            "profiler safe mode; launch callbacks will provide synthetic "
            "kernel rows. Set GPUFL_SASS_ALLOW_KERNEL_ACTIVITY=1 to test "
            "CONCURRENT_KERNEL explicitly.");
        return {};
    }

    // SASS metrics on sm_86/CUDA 13.2 produce all-zero counters when only
    // CONCURRENT_KERNEL is enabled. Main's validated coexistence path enables
    // both serialized and concurrent kernel activity; keep that combination so
    // kernel rows and non-zero SASS counters are collected in one run.
    return {CUPTI_ACTIVITY_KIND_KERNEL, CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL};
}

bool KernelLaunchHandler::shouldHandle(const CUpti_CallbackDomain domain,
                                       const CUpti_CallbackId cbid) const {
    // Single source of truth: a callback is handled iff we registered it in
    // requiredCallbacks(). Deriving the filter from that list (rather than
    // re-typing every CBID and its version guard here) means the two can
    // never drift — a CBID added to one but forgotten in the other used to
    // silently drop that launch API's telemetry. The set is identical for
    // every instance, so build it once on first call (thread-safe since
    // C++11) and reuse it; this stays on the per-callback hot path.
    static const std::set<std::pair<CUpti_CallbackDomain, CUpti_CallbackId>>
        kHandled = [this] {
            const auto cbs = requiredCallbacks();
            return std::set<std::pair<CUpti_CallbackDomain, CUpti_CallbackId>>(
                cbs.begin(), cbs.end());
        }();
    return kHandled.count({domain, cbid}) != 0;
}

void KernelLaunchHandler::handle(CUpti_CallbackDomain domain,
                                 CUpti_CallbackId cbid, const void* cbdata) {
    if (!backend_->IsActive()) return;

    auto* cbInfo = static_cast<const CUpti_CallbackData*>(cbdata);
    if (!cbInfo) {
        GFL_LOG_ERROR("[KernelLaunchHandler] cbInfo is null");
        return;
    }

    if (cbInfo->callbackSite == CUPTI_API_ENTER) {
        backend_->NoteKernelLaunchForCleanupFlush();
        LaunchMeta meta{};
        meta.api_enter_ns = detail::GetTimestampNs();

        // Prefer cbInfo->symbolName so callback-only SASS safe mode can still
        // group kernels by real symbol. CUDA 13.1 sometimes puts invalid tagged
        // values in symbolName, so never dereference it directly; copy through
        // process_vm_readv first and fall back to the stable API function name
        // (cudaLaunchKernel / cuLaunchKernelEx) if the probe fails.
        std::string copiedSymbol;
        if (CopyReadableCString(cbInfo->symbolName, &copiedSymbol)) {
            const std::string demangledName = DemangleName(copiedSymbol.c_str());
            std::snprintf(meta.name, sizeof(meta.name), "%s",
                          demangledName.c_str());
        } else {
            const char* nm = cbInfo->functionName;
            const std::string& demangledName = cachedDemangle(nm);
            std::snprintf(meta.name, sizeof(meta.name), "%s",
                          demangledName.c_str());
        }

        if (backend_->GetOptions().enable_stack_trace) {
            const std::string trace = core::GetCallStack(2);
            const std::string cleanTrace = detail::SanitizeStackTrace(trace);
            meta.stack_id =
                StackRegistry::instance().getOrRegister(cleanTrace);
        } else {
            meta.stack_id = 0;
        }

        auto& stack = getThreadScopeStack();
        if (!stack.empty()) {
            std::string fullPath;
            for (size_t i = 0; i < stack.size(); ++i) {
                if (i > 0) fullPath += "|";
                fullPath += stack[i];
            }
            std::snprintf(meta.user_scope, sizeof(meta.user_scope), "%s",
                          fullPath.c_str());
            meta.scope_depth = stack.size();
        } else if (const char* injectedApp = std::getenv("GPUFL_APP_NAME")) {
            if (std::getenv("GPUFL_INJECT") && injectedApp[0] != '\0') {
                std::string processScope = "process:";
                processScope += injectedApp;
                std::snprintf(meta.user_scope, sizeof(meta.user_scope), "%s",
                              processScope.c_str());
            } else {
                std::snprintf(meta.user_scope, sizeof(meta.user_scope), "%s",
                              "global");
            }
            meta.scope_depth = 0;
        } else {
            std::snprintf(meta.user_scope, sizeof(meta.user_scope), "%s",
                          "global");
            meta.scope_depth = 0;
        }

        auto setLaunchDims = [&meta](const unsigned int gx,
                                     const unsigned int gy,
                    const unsigned int gz, const unsigned int bx,
                    const unsigned int by, const unsigned int bz,
                                     size_t dyn_smem) {
            meta.has_details = true;
            meta.grid_x = static_cast<int>(gx);
            meta.grid_y = static_cast<int>(gy);
            meta.grid_z = static_cast<int>(gz);
            meta.block_x = static_cast<int>(bx);
            meta.block_y = static_cast<int>(by);
            meta.block_z = static_cast<int>(bz);
            meta.dyn_shared = static_cast<int>(dyn_smem);
        };

        if (cbInfo->functionParams != nullptr) {
            if (domain == CUPTI_CB_DOMAIN_RUNTIME_API &&
                cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) {
                const auto* p =
                    (cudaLaunchKernel_v7000_params*)(cbInfo->functionParams);
                setLaunchDims(p->gridDim.x, p->gridDim.y, p->gridDim.z,
                              p->blockDim.x, p->blockDim.y, p->blockDim.z,
                              p->sharedMem);
            } else if (domain == CUPTI_CB_DOMAIN_DRIVER_API &&
                       cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel) {
                const auto* p = (cuLaunchKernel_params*)cbInfo->functionParams;
                setLaunchDims(p->gridDimX, p->gridDimY, p->gridDimZ,
                              p->blockDimX, p->blockDimY, p->blockDimZ,
                              p->sharedMemBytes);
            }
#if GPUFL_HAS_EXTENSIBLE_LAUNCH
            // Extensible launch (CUDA 11.6+). cuda.core's launch() routes
            // through cuLaunchKernelEx — NOT the plain cuLaunchKernel handled
            // above — because LaunchConfig carries launch attributes (thread-
            // block clusters, cooperative, etc.). numba-cuda's modern
            // dispatcher (when cuda.core >= 1.0 is present) launches every
            // kernel this way, as do CUTLASS and recent CUDA samples. Without
            // these branches such kernels arrive with timing but has_details
            // stays false, so grid/block/occupancy are dropped from the report
            // ("kernel details missing" for Numba). The launch config is a
            // pointer captured at API_ENTER and is valid for the callback's
            // lifetime, so read it here while we're still inside the callback.
            else if (domain == CUPTI_CB_DOMAIN_DRIVER_API &&
                     (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx ||
                      cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz)) {
                const auto* p = (cuLaunchKernelEx_params*)cbInfo->functionParams;
                if (p->config != nullptr) {
                    const CUlaunchConfig* c = p->config;
                    setLaunchDims(c->gridDimX, c->gridDimY, c->gridDimZ,
                                  c->blockDimX, c->blockDimY, c->blockDimZ,
                                  c->sharedMemBytes);
                }
            } else if (domain == CUPTI_CB_DOMAIN_RUNTIME_API &&
                       (cbid ==
                            CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060 ||
                        cbid ==
                            CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_ptsz_v11060)) {
                const auto* p =
                    (cudaLaunchKernelExC_v11060_params*)cbInfo->functionParams;
                if (p->config != nullptr) {
                    const cudaLaunchConfig_t* c = p->config;
                    setLaunchDims(c->gridDim.x, c->gridDim.y, c->gridDim.z,
                                  c->blockDim.x, c->blockDim.y, c->blockDim.z,
                                  c->dynamicSmemBytes);
                }
            }
#endif
        }

        // Store metadata — emit later from scope stop (PC Sampling path)
        // or handleActivityRecord (normal path). Runtime and driver launch
        // callbacks can share one correlation id, so keep the richest metadata
        // and avoid duplicate diagnostic lines for the same logical launch.
        bool shouldLogLaunch = false;
        {
            std::lock_guard lk(backend_->meta_mu_);
            auto it = backend_->meta_by_corr_.find(cbInfo->correlationId);
            if (it == backend_->meta_by_corr_.end()) {
                backend_->meta_by_corr_.emplace(cbInfo->correlationId, meta);
                shouldLogLaunch = true;
            } else if (it->second.has_details && !meta.has_details) {
                GFL_LOG_DEBUG(
                    "[DEBUG-CALLBACK] Skipping overwrite of rich metadata "
                    "for CorrID ",
                    cbInfo->correlationId, " by Driver API.");
            } else if (!it->second.has_details && meta.has_details) {
                it->second = meta;
                shouldLogLaunch = true;
            }
        }
        // Diagnostic: confirm every logical kernel launch fires this callback.
        // If "GPU Time by Scope" shows only N kernels but this logs M>N,
        // the loss is downstream (activity records / FlushPendingKernels).
        if (shouldLogLaunch) {
            GFL_LOG_DEBUG("[KernelLaunchHandler] API_ENTER corr=",
                          cbInfo->correlationId, " name=", meta.name,
                          " scope=", meta.user_scope);
        }
    } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
        const int64_t exitNs = detail::GetTimestampNs();
        std::lock_guard lk(backend_->meta_mu_);
        auto it = backend_->meta_by_corr_.find(cbInfo->correlationId);
        if (it != backend_->meta_by_corr_.end()) {
            it->second.api_exit_ns = exitNs;
        }
    }
}

bool KernelLaunchHandler::handleActivityRecord(const CUpti_Activity* record,
                                               int64_t baseCpuNs,
                                               uint64_t baseCuptiTs) {
    if (!record) {
        GFL_LOG_ERROR("[KernelLaunchHandler] null activity record");
        return false;
    }
    if (record->kind != CUPTI_ACTIVITY_KIND_KERNEL &&
        record->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
        return false;
    }

    const auto* k = reinterpret_cast<const CUpti_ActivityKernel11*>(record);
    backend_->kernel_activity_seen_.fetch_add(1, std::memory_order_relaxed);

    // Diagnostic: log every real activity record as it arrives.
    GFL_LOG_DEBUG("[KernelLaunchHandler] ACTIVITY_RECORD corr=",
                  k->correlationId, " name=", k->name,
                  " start=", k->start, " end=", k->end,
                  " dur=", k->end - k->start);

    // NOTE (1.0.1): kernel_sample_rate_ms used to throttle activity-record
    // processing here — skipping records that arrived within a sampling
    // window. That was a serious bug: a throttled (skipped) record left its
    // launch metadata in meta_by_corr_, and FlushPendingKernels() then
    // resurrected the kernel as a SYNTHETIC record whose "duration" was the
    // host dispatch-gap (next launch − this launch), not GPU time. On a
    // host-bound workload that gap is data-loading wait (10–230 ms), so
    // Total GPU Time / GPU Busy were inflated ~60–70× (GPU Busy > 100% while
    // NVML showed the GPU idle). Kernel activity records carry the real GPU
    // end−start and are cheap, so we now ALWAYS process every one. The
    // kernel_sample_rate_ms option is retained for backward compatibility
    // (callers/config files won't break) but is intentionally ignored.

    ActivityRecord out{};
    out.device_id = k->deviceId;
    out.stream = static_cast<StreamHandle>(k->streamId);
    out.type = TraceType::KERNEL;
    const std::string& demangledKernelName = cachedDemangle(k->name);
    std::snprintf(out.name, sizeof(out.name), "%s", demangledKernelName.c_str());
    out.cpu_start_ns = baseCpuNs + static_cast<int64_t>(k->start - baseCuptiTs);
    out.duration_ns = static_cast<int64_t>(k->end - k->start);
    out.dyn_shared = k->dynamicSharedMemory;
    out.static_shared = k->staticSharedMemory;
    out.num_regs = k->registersPerThread;
    out.has_details = false;

    // Phase 1a: always-on fields from CUpti_ActivityKernel11
    out.local_mem_total = k->localMemoryTotal;
    out.local_mem_per_thread =
        k->localMemoryPerThread;  // 0 = no register spill
    out.cache_config_requested = k->cacheConfig.config.requested;
    out.cache_config_executed = k->cacheConfig.config.executed;
    out.shared_mem_executed = k->sharedMemoryExecuted;

    // stamp framework op id (PyTorch / TF / JAX / OPENACC) onto this
    // kernel if the framework was actively pushing external correlation
    // for this thread. Lookup is pop-on-read, so this also keeps the
    // map from accumulating stale entries across long sessions.
    //
    // Diagnostic: track hit / miss counters so we can tell whether the
    // F1 chain is broken at "no correlation records arrived" vs "they
    // arrived but with mismatched corr_ids" vs "matching but ordering
    // races (kernel processed before its correlation record)".
    {
        uint8_t  ext_kind = 0;
        uint64_t ext_id   = 0;
        const bool hit = LookupAndPopExternalCorrelation(
            k->correlationId, &ext_kind, &ext_id);
        if (hit) {
            out.external_kind = ext_kind;
            out.external_id   = ext_id;
            static std::atomic<int> g_ec_hit{0};
            const int n = g_ec_hit.fetch_add(1, std::memory_order_relaxed) + 1;
            if (n <= 5 || n % 100 == 0) {
                GFL_LOG_DEBUG(
                    "[KernelHandler] external lookup HIT #", n,
                    " corr=", k->correlationId, " ext_id=", ext_id);
            }
        } else {
            static std::atomic g_ec_miss{0};
            const int n = g_ec_miss.fetch_add(1, std::memory_order_relaxed) + 1;
            if (n <= 5 || n % 500 == 0) {
                GFL_LOG_DEBUG(
                    "[KernelHandler] external lookup MISS #", n,
                    " corr=", k->correlationId);
            }
        }
    }

    {
        const uint64_t corr = k->correlationId;
        out.corr_id = corr;
        std::lock_guard lk(backend_->meta_mu_);
        if (auto it = backend_->meta_by_corr_.find(corr);
            it != backend_->meta_by_corr_.end()) {
            const LaunchMeta& m = it->second;
            out.scope_depth = m.scope_depth;
            out.stack_id = m.stack_id;
            std::copy(std::begin(m.user_scope), std::end(m.user_scope),
                      std::begin(out.user_scope));
            out.api_start_ns = m.api_enter_ns;
            out.api_exit_ns = m.api_exit_ns;
            // Snapshot `has_details` before the erase below — we still need
            // it to decide whether to compute occupancy. Erasing first
            // would invalidate `m` (dangling reference).
            const bool hadDetails = m.has_details;
            if (hadDetails) {
                out.has_details = true;
                out.grid_x = m.grid_x;
                out.grid_y = m.grid_y;
                out.grid_z = m.grid_z;
                out.block_x = m.block_x;
                out.block_y = m.block_y;
                out.block_z = m.block_z;
                out.local_bytes = static_cast<int>(k->localMemoryPerThread);
                out.const_bytes = m.const_bytes;

                // Compute per-resource occupancy from activity record data
                // (registers, shared memory) and SM properties.
                SmProps props = GetSMProps(out.device_id);
                int threadsPerBlock = out.block_x * out.block_y * out.block_z;
                int warpsPerBlock =
                    (threadsPerBlock + props.warpSize - 1) / props.warpSize;
                int maxWarpsPerSM = props.maxThreadsPerSM / props.warpSize;

                // Warp limit
                int warpBlocks =
                    (warpsPerBlock > 0) ? (maxWarpsPerSM / warpsPerBlock) : 0;

                // Hardware block count limit
                int blockBlocks = props.maxBlocksPerSM;

                // Register limit — registers are allocated per-warp in
                // multiples of 256 on modern NVIDIA architectures (sm_6x+).
                constexpr int kRegAllocGranularity = 256;
                int regsPerWarp = (warpsPerBlock > 0 && out.num_regs > 0)
                                      ? (((out.num_regs * props.warpSize) +
                                          kRegAllocGranularity - 1) /
                                         kRegAllocGranularity) *
                                            kRegAllocGranularity
                                      : 0;
                int regsPerBlock = regsPerWarp * warpsPerBlock;
                int regBlocks = regsPerBlock > 0
                                    ? props.regsPerSM / regsPerBlock
                                    : warpBlocks;

                // Shared memory limit
                int smemPerBlock = out.static_shared + out.dyn_shared;
                int smemBlocks =
                    (smemPerBlock > 0) ? (props.sharedMemPerSM / smemPerBlock)
                                       : warpBlocks;

                out.max_active_blocks =
                    std::min({warpBlocks, regBlocks, blockBlocks, smemBlocks});

                auto toOcc = [&](const int blocks) -> float {
                    return maxWarpsPerSM > 0 && warpsPerBlock > 0
                               ? std::min(1.0f, static_cast<float>(
                                                    blocks * warpsPerBlock) /
                                                    maxWarpsPerSM)
                               : 0.0f;
                };
                out.warp_occupancy = toOcc(warpBlocks);
                out.reg_occupancy = toOcc(regBlocks);
                out.smem_occupancy = toOcc(smemBlocks);
                out.block_occupancy = toOcc(blockBlocks);
                out.occupancy = toOcc(out.max_active_blocks);

                struct {
                    float occ;
                    const char* name;
                } limiters[] = {
                    {out.warp_occupancy, "warps"},
                    {out.reg_occupancy, "registers"},
                    {out.smem_occupancy, "shared_mem"},
                    {out.block_occupancy, "blocks"},
                };
                auto limiting = "warps";
                float minOcc = out.warp_occupancy;
                for (auto& [occ, name] : limiters) {
                    if (occ < minOcc) {
                        minOcc = occ;
                        limiting = name;
                    }
                }
                std::snprintf(out.limiting_resource,
                              sizeof(out.limiting_resource), "%s", limiting);
            }
            // Always erase after processing — previously this was nested
            // inside `if (hadDetails)`, so launches without grid/block
            // metadata left stale entries in meta_by_corr_. At session
            // stop() FlushPendingKernels would then emit synthetic
            // duplicates for kernels that already had real activity
            // records emitted here. Move erase out of the conditional.
            backend_->meta_by_corr_.erase(it);
        }
    }

    g_monitorBuffer.Push(out);
    backend_->kernel_activity_emitted_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

}  // namespace gpufl

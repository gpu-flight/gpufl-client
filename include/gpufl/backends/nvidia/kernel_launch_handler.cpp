#include "gpufl/backends/nvidia/kernel_launch_handler.hpp"

#include "gpufl/core/env_vars.hpp"

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
#elif defined(_WIN32)
// ReadProcessMemory / GetCurrentProcess for the Windows symbolName probe below.
// Define NOMINMAX first — this TU uses std::min and windows.h's min/max macros
// would otherwise shadow it.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
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
#elif defined(_WIN32)
    constexpr size_t kMaxNameLen = 4096;
    char buf[kMaxNameLen] = {};
    SIZE_T nread = 0;
    // Windows analog of the Linux process_vm_readv probe: read our OWN process
    // memory so an invalid/tagged symbolName pointer (CUDA can hand these back)
    // fails cleanly instead of crashing on a direct dereference. A partial copy
    // at a page boundary returns FALSE/ERROR_PARTIAL_COPY but still fills `buf`
    // up to `nread`; kernel symbols are short, so the NUL is usually in there.
    ReadProcessMemory(GetCurrentProcess(), src, buf, sizeof(buf), &nread);
    if (nread == 0) return false;
    const void* nul = std::memchr(buf, '\0', static_cast<size_t>(nread));
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

// Precompute the SIMPLIFIED occupancy carried on a synthetic kernel's launch-
// meta record. Only grid/block + dynamic shared memory are known at launch time
// (per-thread registers and static shared memory live on the activity record,
// which by definition never arrives for a synthetic kernel), so occupancy is
// bounded by warps + blocks only — byte-for-byte the math the old
// CuptiBackend::FlushPendingKernels ran at stop(). Runs here on the launch
// callback (the nvidia layer, where GetSMProps lives) so the core collector TU
// can stay free of CUDA headers; drainSyntheticKernels in monitor.cpp just
// copies these fields. Only called in synthetic-kernel modes (see
// CuptiBackend::WillEmitSyntheticKernels), so normal/Deep launches pay nothing.
void ComputeSyntheticOccupancy(ActivityRecord& rec, uint32_t device_id) {
    SmProps props = GetSMProps(static_cast<int>(device_id));
    int threadsPerBlock = rec.block_x * rec.block_y * rec.block_z;
    int warpsPerBlock = (threadsPerBlock + props.warpSize - 1) / props.warpSize;
    int maxWarpsPerSM = props.maxThreadsPerSM / props.warpSize;
    int warpBlocks = (warpsPerBlock > 0) ? (maxWarpsPerSM / warpsPerBlock) : 0;
    int blockBlocks = props.maxBlocksPerSM;
    rec.max_active_blocks = std::min(warpBlocks, blockBlocks);
    auto toOcc = [&](int blocks) -> float {
        return (maxWarpsPerSM > 0 && warpsPerBlock > 0)
                   ? std::min(1.0f, static_cast<float>(blocks * warpsPerBlock) /
                                        maxWarpsPerSM)
                   : 0.0f;
    };
    rec.warp_occupancy = toOcc(warpBlocks);
    rec.block_occupancy = toOcc(blockBlocks);
    rec.occupancy = rec.warp_occupancy;
    std::snprintf(rec.limiting_resource, sizeof(rec.limiting_resource), "%s",
                  "warps");
}

}  // namespace

KernelLaunchHandler::KernelLaunchHandler(CuptiBackend* backend)
    : backend_(backend) {}

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
    // Single source of truth for whether CUPTI kernel ACTIVITY is collected:
    // CuptiBackend::collectsKernelEvents() (covers single engines AND combos).
    // When false (PC sampling, or SASS safe mode), launch callbacks provide
    // synthetic kernel rows instead.
    if (backend_ && !backend_->collectsKernelEvents()) {
        GFL_LOG_DEBUG(
            "[KernelLaunchHandler] CUPTI kernel activity disabled for this "
            "engine selection; launch callbacks will provide synthetic kernel "
            "rows.");
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
            return std::set(
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

        // Build a KERNEL_LAUNCH_META record and push it to the lock-free ring
        // (Step 4b-2). The corr->meta join that used to run here under meta_mu_
        // now happens on the single collector thread (joinLaunchMeta in
        // monitor.cpp) — this callback takes NO lock. Push is heap-free: the
        // ring slot is preallocated and Push copies the POD into it.
        ActivityRecord metaRec{};
        metaRec.type = TraceType::KERNEL_LAUNCH_META;
        metaRec.corr_id = cbInfo->correlationId;
        // device_id is consumed only by the synthetic-kernel path (real kernels
        // take k->deviceId from the activity record); matches the old
        // FlushPendingKernels `out.device_id = device_id_`.
        metaRec.device_id = backend_->device_id_;
        metaRec.api_start_ns = detail::GetTimestampNs();  // API_ENTER ns

        // Prefer cbInfo->symbolName so callback-only SASS safe mode can still
        // group kernels by real symbol. CUDA 13.1 sometimes puts invalid tagged
        // values in symbolName, so never dereference it directly; copy through
        // process_vm_readv first and fall back to the stable API function name
        // (cudaLaunchKernel / cuLaunchKernelEx) if the probe fails.
        // Store the RAW name; demangling is deferred to the collector thread
        // (DemangleKernelNameCached in monitor.cpp). %.*s bounds the source
        // read so a non-null-terminated CUPTI name pointer can't overrun.
        std::string copiedSymbol;
        if (CopyReadableCString(cbInfo->symbolName, &copiedSymbol)) {
            std::snprintf(metaRec.name, sizeof(metaRec.name), "%.*s",
                          static_cast<int>(sizeof(metaRec.name) - 1),
                          copiedSymbol.c_str());
        } else {
            const char* nm = cbInfo->functionName ? cbInfo->functionName : "";
            std::snprintf(metaRec.name, sizeof(metaRec.name), "%.*s",
                          static_cast<int>(sizeof(metaRec.name) - 1), nm);
        }

        if (backend_->GetOptions().enable_stack_trace) {
            // Capture raw return addresses only (cheap); symbol resolution is
            // deferred to the collector thread via StackRegistry::get(),
            // keeping dbghelp/SymFromAddr off this per-launch CUPTI callback.
            metaRec.stack_id = StackRegistry::instance().getOrRegister(
                core::CaptureCallStackRaw(2));
        } else {
            metaRec.stack_id = 0;
        }

        auto& stack = getThreadScopeStack();
        if (!stack.empty()) {
            std::string fullPath;
            for (size_t i = 0; i < stack.size(); ++i) {
                if (i > 0) fullPath += "|";
                fullPath += stack[i];
            }
            std::snprintf(metaRec.user_scope, sizeof(metaRec.user_scope), "%s",
                          fullPath.c_str());
            metaRec.scope_depth = stack.size();
        } else if (const char* injectedApp = std::getenv(gpufl::env::kAppName)) {
            if (std::getenv(gpufl::env::kInject) && injectedApp[0] != '\0') {
                std::string processScope = "process:";
                processScope += injectedApp;
                std::snprintf(metaRec.user_scope, sizeof(metaRec.user_scope),
                              "%s", processScope.c_str());
            } else {
                std::snprintf(metaRec.user_scope, sizeof(metaRec.user_scope),
                              "%s", "global");
            }
            metaRec.scope_depth = 0;
        } else {
            std::snprintf(metaRec.user_scope, sizeof(metaRec.user_scope), "%s",
                          "global");
            metaRec.scope_depth = 0;
        }

        auto setLaunchDims = [&metaRec](const unsigned int gx,
                                        const unsigned int gy,
                    const unsigned int gz, const unsigned int bx,
                    const unsigned int by, const unsigned int bz,
                                        size_t dyn_smem) {
            metaRec.has_details = true;
            metaRec.grid_x = static_cast<int>(gx);
            metaRec.grid_y = static_cast<int>(gy);
            metaRec.grid_z = static_cast<int>(gz);
            metaRec.block_x = static_cast<int>(bx);
            metaRec.block_y = static_cast<int>(by);
            metaRec.block_z = static_cast<int>(bz);
            metaRec.dyn_shared = static_cast<int>(dyn_smem);
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

        // Synthetic-kernel modes (PC Sampling / SASS safe): no CUPTI activity
        // record will arrive, so precompute the simplified occupancy now — the
        // only place with the nvidia SM properties — and carry it on the meta
        // record for drainSyntheticKernels to copy. Skipped in normal / Deep
        // mode (the activity record supplies full occupancy), keeping this
        // callback thin. NOTE: a rare leftover meta orphaned at stop() in normal
        // mode (its activity record dropped/in-flight) now ships occupancy 0
        // instead of the old simplified value — acceptable for a degraded
        // best-effort row whose duration is already a host-dispatch estimate.
        if (metaRec.has_details && backend_->WillEmitSyntheticKernels()) {
            ComputeSyntheticOccupancy(metaRec, metaRec.device_id);
        }

        g_monitorBuffer.Push(metaRec);

        // Diagnostic: confirm every logical kernel launch fires this callback.
        // Runtime+driver share a corr so this can log twice per launch; the
        // collector's keep-first merge dedups the actual join.
        GFL_LOG_DEBUG("[KernelLaunchHandler] API_ENTER corr=", metaRec.corr_id,
                      " name=", metaRec.name, " scope=", metaRec.user_scope);
    } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
        // Push the API-exit timestamp as its own record; the collector patches
        // it onto the matching launch meta. Same thread as API_ENTER, so it
        // enters the ring after the ENTER record (FIFO by push order).
        ActivityRecord exitRec{};
        exitRec.type = TraceType::KERNEL_API_EXIT;
        exitRec.corr_id = cbInfo->correlationId;
        exitRec.api_exit_ns = detail::GetTimestampNs();
        g_monitorBuffer.Push(exitRec);
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
    const char* kernelName = k->name ? k->name : "";
    const bool looksLikeReplayWait =
        std::strcmp(kernelName, "WaitNs") == 0 &&
        k->gridX == 1 && k->gridY == 1 && k->gridZ == 1 &&
        k->blockX == 1 && k->blockY == 1 && k->blockZ == 1;
    if (backend_->UsesRangeProfilerKernelReplay() && looksLikeReplayWait) {
        GFL_LOG_DEBUG("[KernelLaunchHandler] skipping RangeProfilerKernelReplay "
                      "internal wait kernel corr=", k->correlationId);
        return false;
    }

    // Diagnostic: log every real activity record as it arrives.
    GFL_LOG_DEBUG("[KernelLaunchHandler] ACTIVITY_RECORD corr=",
                  k->correlationId, " name=", kernelName,
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
    // Store the RAW (mangled) name; demangle is deferred to the collector
    // thread. %.*s bounds the source read against a non-terminated name.
    std::snprintf(out.name, sizeof(out.name), "%.*s",
                  static_cast<int>(sizeof(out.name) - 1),
                  kernelName);
    out.cpu_start_ns = baseCpuNs + static_cast<int64_t>(k->start - baseCuptiTs);
    out.duration_ns = static_cast<int64_t>(k->end - k->start);
    out.dyn_shared = k->dynamicSharedMemory;
    out.static_shared = k->staticSharedMemory;
    out.num_regs = k->registersPerThread;

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

    out.corr_id = k->correlationId;

    // Launch dims + occupancy come straight from the activity record:
    // CUpti_ActivityKernel11 carries the ACTUAL launched grid/block/regs/
    // shared-mem, so they're computed with NO lock and for every kernel —
    // independent of whether the launch-callback metadata is present. (Step 4b)
    out.has_details = true;
    out.grid_x = k->gridX;
    out.grid_y = k->gridY;
    out.grid_z = k->gridZ;
    out.block_x = k->blockX;
    out.block_y = k->blockY;
    out.block_z = k->blockZ;
    out.local_bytes = static_cast<int>(k->localMemoryPerThread);
    {
        // Per-resource occupancy from launch config + SM properties.
        SmProps props = GetSMProps(out.device_id);
        int threadsPerBlock = out.block_x * out.block_y * out.block_z;
        int warpsPerBlock =
            (threadsPerBlock + props.warpSize - 1) / props.warpSize;
        int maxWarpsPerSM = props.maxThreadsPerSM / props.warpSize;
        int warpBlocks =
            (warpsPerBlock > 0) ? (maxWarpsPerSM / warpsPerBlock) : 0;
        int blockBlocks = props.maxBlocksPerSM;
        // Registers are allocated per-warp in multiples of 256 on sm_6x+.
        constexpr int kRegAllocGranularity = 256;
        int regsPerWarp = (warpsPerBlock > 0 && out.num_regs > 0)
                              ? (((out.num_regs * props.warpSize) +
                                  kRegAllocGranularity - 1) /
                                 kRegAllocGranularity) *
                                    kRegAllocGranularity
                              : 0;
        int regsPerBlock = regsPerWarp * warpsPerBlock;
        int regBlocks =
            regsPerBlock > 0 ? props.regsPerSM / regsPerBlock : warpBlocks;
        int smemPerBlock = out.static_shared + out.dyn_shared;
        int smemBlocks = (smemPerBlock > 0)
                             ? (props.sharedMemPerSM / smemPerBlock)
                             : warpBlocks;
        out.max_active_blocks =
            std::min({warpBlocks, regBlocks, blockBlocks, smemBlocks});
        auto toOcc = [&](const int blocks) -> float {
            return maxWarpsPerSM > 0 && warpsPerBlock > 0
                       ? std::min(1.0f, static_cast<float>(blocks *
                                                           warpsPerBlock) /
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
        std::snprintf(out.limiting_resource, sizeof(out.limiting_resource),
                      "%s", limiting);
    }

    // Launch-callback metadata (scope path, stack id, API timestamps, const-
    // bank size — the only fields NOT in the activity record) is joined on the
    // collector thread now (joinLaunchMeta in monitor.cpp), keyed by corr_id.
    // This callback takes NO meta_mu_ — it pushes the raw activity record and
    // returns. (Step 4b-2: the join moved off the CUPTI threads to drop the
    // mutex; the matching KERNEL_LAUNCH_META was pushed by handle() at ENTER.)
    g_monitorBuffer.Push(out);
    backend_->kernel_activity_emitted_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

}  // namespace gpufl

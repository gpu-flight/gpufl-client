#include "gpufl/backends/nvidia/cupti_backend.hpp"
#include "gpufl/backends/nvidia/cuda_cleanup_handler.hpp"

#include <cupti_pcsampling.h>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

// Platform-specific includes for runtime NVTX injection path discovery.
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>  // EnumProcessModules / GetModuleFileNameA
#elif defined(__linux__)
#include <dlfcn.h>
#endif

#include "gpufl/backends/nvidia/cuda_collector.hpp"
#include "gpufl/backends/nvidia/cupti_utils.hpp"
#include "gpufl/backends/nvidia/engine/pc_sampling_engine.hpp"
#include "gpufl/backends/nvidia/engine/pc_sampling_with_sass_engine.hpp"
#include "gpufl/backends/nvidia/engine/pm_sampling_engine.hpp"
#include "gpufl/backends/nvidia/engine/range_profiler_engine.hpp"
#include "gpufl/backends/nvidia/engine/sass_metrics_engine.hpp"
#include "gpufl/backends/nvidia/kernel_launch_handler.hpp"
#include "gpufl/backends/nvidia/mem_transfer_handler.hpp"
#include "gpufl/backends/nvidia/resource_handler.hpp"
#include "gpufl/backends/nvidia/synchronization_handler.hpp"
#include "gpufl/core/activity_record.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/logger/logger.hpp"
#include "gpufl/core/model/lifecycle_model.hpp"
#include "gpufl/core/runtime.hpp"
#include "gpufl/core/ring_buffer.hpp"
#include "gpufl/core/stack_registry.hpp"
#include "gpufl/core/trace_type.hpp"

namespace gpufl {
std::atomic<CuptiBackend*> g_activeBackend{nullptr};

namespace {

void AddCapability(CaptureCapabilitiesEvent& evt, std::string feature,
                   bool requested, std::string status, std::string mode,
                   std::string reason, std::string message) {
    evt.capabilities.push_back(CaptureCapability{
        std::move(feature), requested, std::move(status), std::move(mode),
        std::move(reason), std::move(message)});
}

// Persistent maps for ActivityAPI PC sampling companion records.
// SOURCE_LOCATOR records map sourceLocatorId → (fileName, lineNumber).
// FUNCTION records map functionId → functionName.
// Both arrive in the same buffer as PC_SAMPLING records and must outlive
// individual BufferCompleted calls.
std::mutex g_sourceLocatorMu;
std::unordered_map<uint32_t, std::pair<std::string, uint32_t>> g_sourceLocatorMap;
std::unordered_map<uint32_t, std::string> g_functionNameMap;

// NVTX marker pairing. CUPTI delivers each NVTX range as two separate
// activity records: one with flags=START, one with flags=END, both
// sharing the same id. We pair them here in the buffer-completion
// callback to emit a single NvtxMarkerEvent with start, end, and
// duration. Map entry value: (name, start_timestamp, domain).
struct NvtxOpen {
    std::string name;
    std::string domain;
    uint64_t start_ts = 0;
};
std::mutex g_nvtxMu;
std::unordered_map<uint32_t, NvtxOpen> g_nvtxOpen;

// External-correlation map.
//
// CUPTI emits CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION records whenever a
// framework brackets a code region with cuptiActivityPushExternalCorrelationId
// and a kernel launches inside that bracket. The record carries:
//   - externalKind  : which framework (PyTorch / TF / JAX / OPENACC / ...)
//   - externalId    : the framework's per-op id
//   - correlationId : the CUPTI per-launch id, identical to the matching
//                     KERNEL activity record's correlationId
//
// In practice the EXTERNAL_CORRELATION record arrives *before* the matching
// KERNEL record within a single buffer (CUPTI emits the bracket events as
// the launch is enqueued, before the launch completes on the GPU). When
// the kernel record arrives we look up its corr_id in this map, stamp the
// (kind, id) onto the kernel's ActivityRecord, and erase from the map so
// it doesn't leak across sessions.
//
// If a kernel arrives BEFORE its external correlation record (rare; would
// require CUPTI to deliver records out of generation order), we miss the
// stamp for that one launch — emit external_id == 0, treated by the
// dashboard as "no framework attribution." Acceptable best-effort.
struct ExternalCorrInfo {
    uint8_t  kind = 0;
    uint64_t id   = 0;
};
std::mutex g_extCorrMu;
std::unordered_map<uint32_t, ExternalCorrInfo> g_extCorrMap;
}  // namespace

// Public helper for cross-TU access. KernelLaunchHandler (different .cpp)
// calls this from `handleActivityRecord` to stamp the kernel with its
// framework op id. Returns false (and leaves outputs untouched) when no
// matching external correlation has been seen yet for this corr_id.
//
// Pop-on-read: each correlation record matches exactly one kernel, and
// keeping stale entries would slowly grow the map across long sessions.
bool LookupAndPopExternalCorrelation(uint32_t corr_id,
                                     uint8_t* kind_out,
                                     uint64_t* id_out) {
    std::lock_guard<std::mutex> lk(g_extCorrMu);
    auto it = g_extCorrMap.find(corr_id);
    if (it == g_extCorrMap.end()) return false;
    if (kind_out) *kind_out = it->second.kind;
    if (id_out)   *id_out   = it->second.id;
    g_extCorrMap.erase(it);
    return true;
}

// F1 active push: thin wrappers over CUPTI's correlation stack. The
// caller (e.g. `gpufl.torch.attach()`) calls these around a code region
// — every kernel launched in between gets the (kind, id) emitted as an
// EXTERNAL_CORRELATION record, which our BufferCompleted path then
// stamps onto the matching kernel's row. This is what makes F1 useful
// without requiring a framework profiler to be running.
//
// Both operations are pure CUPTI library calls; they don't need a
// CuptiBackend instance to exist (the stack is per-thread inside CUPTI
// itself). Safe to call before init / after shutdown — CUPTI returns
// CUPTI_ERROR_NOT_INITIALIZED which we silently ignore.
//
// Diagnostic: count pushes + log the first few + log any error result.
// "Pushes happen with OK return but no EXTERNAL_CORRELATION records"
// is a distinct failure mode from "pushes never happen" — these logs
// distinguish them. Also log the OS thread id; if pushes happen on a
// different thread than the kernel launches, CUPTI's per-thread stack
// won't bracket the launch.
void pushExternalCorrelation(uint32_t kind, uint64_t id) {
    const CUptiResult res = cuptiActivityPushExternalCorrelationId(
        static_cast<CUpti_ExternalCorrelationKind>(kind), id);
    static std::atomic<int> g_push_count{0};
    const int n = g_push_count.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n <= 5 || res != CUPTI_SUCCESS) {
        const auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
        GFL_LOG_DEBUG("[ExternalCorr] push #", n,
                      " kind=", kind, " id=", id,
                      " result=", static_cast<int>(res),
                      " tid=", static_cast<uint64_t>(tid));
    }
}

void popExternalCorrelation(uint32_t kind) {
    uint64_t lastId = 0;
    const CUptiResult res = cuptiActivityPopExternalCorrelationId(
        static_cast<CUpti_ExternalCorrelationKind>(kind), &lastId);
    static std::atomic<int> g_pop_count{0};
    const int n = g_pop_count.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n <= 5 || res != CUPTI_SUCCESS) {
        GFL_LOG_DEBUG("[ExternalCorr] pop #", n,
                      " kind=", kind,
                      " lastId=", lastId,
                      " result=", static_cast<int>(res));
    }
}

namespace {
bool IsInsufficientPrivilege(CUptiResult res) {
    if (res == CUPTI_ERROR_INSUFFICIENT_PRIVILEGES) return true;
#ifdef CUPTI_ERROR_VIRTUALIZED_DEVICE_INSUFFICIENT_PRIVILEGES
    if (res == CUPTI_ERROR_VIRTUALIZED_DEVICE_INSUFFICIENT_PRIVILEGES)
        return true;
#endif
    return false;
}

void LogCuptiIfUnexpected(const char* scope, const char* op, CUptiResult res) {
    if (res == CUPTI_SUCCESS || res == CUPTI_ERROR_NOT_INITIALIZED ||
        IsInsufficientPrivilege(res)) {
        return;
    }
    LogCuptiErrorIfFailed(scope, op, res);
}

#if defined(__linux__)
// CUPTI loads PerfWorks by soname the first time a profiling feature
// runs — cuptiPCSamplingEnable (PC sampling) and cuptiProfilerInitialize
// (SASS / Range / Deep) both do. PerfWorks is NOT one library: the host
// API (libnvperf_host.so) pulls in companions — libnvperf_target.so (the
// driver-side counterpart that NVPW_CUDA_LoadDriver initializes) and, for
// PC sampling, libpcsamplingutil.so. ALL of them must come from the SAME
// CUDA install as the libcupti we're bound to. If the dynamic loader
// resolves ANY of them from a DIFFERENT install (classic case: a pip
// `nvidia-cu13` CUPTI inside a venv + the system /usr/local/cuda nvperf),
// NVPW_CUDA_LoadDriver SEGFAULTs on the version mismatch — the crash that
// killed BOTH Deep and PcSampling in PyTorch venvs.
//
// Putting the matching directory on LD_LIBRARY_PATH fixes it because that
// redirects the WHOLE set. An earlier version of this preloaded ONLY
// libnvperf_host.so, which was NOT enough: host resolved to the venv copy
// but its companion libnvperf_target.so still resolved to the mismatched
// system copy, so NVPW_CUDA_LoadDriver kept crashing. So we preload the
// ENTIRE PerfWorks set sitting next to our libcupti, RTLD_GLOBAL, in
// dependency order (target before host). The loader tracks shared objects
// by SONAME, so once these are resident CUPTI's later internal dlopen()s
// return THESE regardless of LD_LIBRARY_PATH ordering. Best-effort:
// anything missing is logged and skipped — CUPTI falls back to the
// loader's choice (prior behavior).
void PreloadMatchingPerfWorks() {
    Dl_info info{};
    // &cuptiSubscribe resolves into whichever libcupti this binary is
    // bound to; dladdr hands back that library's on-disk path.
    if (!dladdr(reinterpret_cast<void*>(&cuptiSubscribe), &info) ||
        !info.dli_fname || !info.dli_fname[0]) {
        GFL_LOG_DEBUG("[CuptiBackend] PerfWorks preload: couldn't locate our "
                      "libcupti via dladdr; skipping (CUPTI will use the "
                      "loader's PerfWorks libs).");
        return;
    }
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path dir = fs::path(info.dli_fname).parent_path();
    if (dir.empty() || !fs::is_directory(dir, ec)) {
        GFL_LOG_DEBUG("[CuptiBackend] PerfWorks preload: our libcupti's "
                      "directory is not accessible; skipping.");
        return;
    }

    auto tryLoad = [](const fs::path& p) {
        if (dlopen(p.string().c_str(), RTLD_LAZY | RTLD_GLOBAL)) {
            GFL_LOG_DEBUG("[CuptiBackend] Preloaded PerfWorks lib: ",
                          p.string());
        } else {
            const char* err = dlerror();
            GFL_LOG_DEBUG("[CuptiBackend] PerfWorks preload skipped ",
                          p.string(), " (", err ? err : "n/a", ")");
        }
    };

    // Bucket the companion libs in our CUPTI's directory. Load order
    // matters: dependencies before dependents. libnvperf_target.so (driver
    // side) and other helpers go first, libnvperf_host.so last — otherwise
    // host's DT_NEEDED on the target soname resolves to the system copy
    // BEFORE we make the matching one resident, and soname-dedup then locks
    // in the wrong target.
    std::vector<fs::path> targets, hosts, others;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (ec) break;
        const std::string name = entry.path().filename().string();
        if (name.find(".so") == std::string::npos) continue;
        if (name.rfind("libnvperf_target", 0) == 0) {
            targets.push_back(entry.path());
        } else if (name.rfind("libnvperf_host", 0) == 0) {
            hosts.push_back(entry.path());
        } else if (name.rfind("libnvperf", 0) == 0 ||
                   name.rfind("libpcsamplingutil", 0) == 0) {
            others.push_back(entry.path());
        }
    }

    if (targets.empty() && hosts.empty() && others.empty()) {
        GFL_LOG_DEBUG("[CuptiBackend] No PerfWorks libs found next to our "
                      "CUPTI in ", dir.string(), " — CUPTI will use the "
                      "loader's choice, which may mismatch and crash in "
                      "NVPW_CUDA_LoadDriver on split CUDA installs.");
        return;
    }

    for (const auto& p : targets) tryLoad(p);  // driver-side first
    for (const auto& p : others) tryLoad(p);   // pcsamplingutil, etc.
    for (const auto& p : hosts) tryLoad(p);     // host API last
}
#else
// Non-Linux: the same split-install hazard exists on Windows
// (cupti64_*.dll loading a mismatched nvperf_host.dll) but isn't wired
// up yet — Windows users typically run a single consistent CUDA toolkit.
inline void PreloadMatchingPerfWorks() {}
#endif
}  // namespace

void CuptiBackend::initialize(const MonitorOptions& opts) {
    opts_ = opts;

    DebugLogger::setEnabled(opts_.enable_debug_output);

    // Create the engine (no CUDA context needed yet)
    switch (opts_.profiling_engine) {
        case ProfilingEngine::PcSampling:
            engine_ = std::make_unique<PcSamplingEngine>();
            GFL_LOG_DEBUG("[CuptiBackend] Engine: PcSampling");
            break;
        case ProfilingEngine::SassMetrics:
            engine_ = std::make_unique<SassMetricsEngine>();
            GFL_LOG_DEBUG("[CuptiBackend] Engine: SassMetrics");
            break;
        case ProfilingEngine::PmSampling:
#if GPUFL_HAS_PERFWORKS
            engine_ = std::make_unique<PmSamplingEngine>();
            GFL_LOG_DEBUG("[CuptiBackend] Engine: PmSampling");
#else
            GFL_LOG_ERROR(
                "[CuptiBackend] PmSampling engine requires "
                "GPUFL_HAS_PERFWORKS; falling back to kernel-trace only");
#endif
            break;
        case ProfilingEngine::RangeProfiler:
#if GPUFL_HAS_PERFWORKS
            engine_ = std::make_unique<RangeProfilerEngine>();
            GFL_LOG_DEBUG("[CuptiBackend] Engine: RangeProfiler");
#else
            GFL_LOG_ERROR(
                "[CuptiBackend] RangeProfiler engine requires "
                "GPUFL_HAS_PERFWORKS; falling back to kernel-trace only");
#endif
            break;
        case ProfilingEngine::Deep:
            // Deep = the deepest analysis the GPU supports. SASS (Profiler
            // API) and PC sampling are mutually exclusive on current drivers,
            // so the composite collects one or the other: SASS metrics on
            // Blackwell+ (where lazy patching is safe), PC sampling on older
            // GPUs (SASS lazy patching deadlocks against concurrent kernel
            // launches there — observed on Ampere sm_86). The engine logs
            // which path it selected in initialize().
            engine_ = std::make_unique<PcSamplingWithSassEngine>();
            GFL_LOG_DEBUG("[CuptiBackend] Engine: Deep");
            break;
        case ProfilingEngine::Trace:
        default:
            // No sampling engine — activity records only (kernels, memcpy,
            // sync). ProfilingEngine::Monitor never reaches here:
            // CreateMonitorAdapter returns nullptr for it, so no
            // CuptiBackend is created at all.
            GFL_LOG_DEBUG("[CuptiBackend] Engine: none (activity trace only)");
            break;
    }

    // Any engine that touches PerfWorks (PcSampling via cuptiPCSamplingEnable;
    // SassMetrics / RangeProfiler / Deep via cuptiProfilerInitialize) must use
    // the libnvperf_host.so that MATCHES our libcupti, or NVPW_CUDA_LoadDriver
    // segfaults on a split CUDA install. Preload it now, before any of those
    // CUPTI calls run. `engine_` is null only for Trace, which needs no
    // PerfWorks, so we skip the preload there.
    if (engine_) {
        PreloadMatchingPerfWorks();
    }

    g_activeBackend.store(this, std::memory_order_release);

    // Internal handler registration
    RegisterHandler(std::make_shared<ResourceHandler>(this));
    RegisterHandler(std::make_shared<CudaCleanupHandler>(this));
    RegisterHandler(std::make_shared<KernelLaunchHandler>(this));
    RegisterHandler(std::make_shared<MemTransferHandler>(this));
    RegisterHandler(std::make_shared<SynchronizationHandler>(this));

    GFL_LOG_DEBUG("Subscribing to CUPTI...");
    CUPTI_CHECK_RETURN(
        cuptiSubscribe(&subscriber_,
                       reinterpret_cast<CUpti_CallbackFunc>(GflCallback), this),
        "[GPUFL Monitor] ERROR: Failed to subscribe to CUPTI\n"
        "[GPUFL Monitor] This may indicate:\n"
        "  - CUPTI library not found or incompatible\n"
        "  - Insufficient permissions\n"
        "  - CUDA driver issues");
    GFL_LOG_DEBUG("CUPTI subscription successful");

    std::set<CUpti_CallbackDomain> domains;
    std::set<std::pair<CUpti_CallbackDomain, CUpti_CallbackId>> callbacks;
    {
        std::lock_guard<std::mutex> lk(handler_mu_);
        for (const auto& h : handlers_) {
            for (auto d : h->requiredDomains()) domains.insert(d);
            for (auto cb : h->requiredCallbacks()) callbacks.insert(cb);
        }
    }
    for (auto d : domains) CUPTI_CHECK(cuptiEnableDomain(1, subscriber_, d));
    for (auto& [domain, cbid] : callbacks)
        CUPTI_CHECK(cuptiEnableCallback(1, subscriber_, domain, cbid));

    CUptiResult resCb =
        cuptiActivityRegisterCallbacks(BufferRequested, BufferCompleted);
    if (resCb != CUPTI_SUCCESS) {
        GFL_LOG_ERROR("FATAL: Failed to register activity callbacks.");
        LogCuptiErrorIfFailed("CUPTI", "cuptiActivityRegisterCallbacks", resCb);
        initialized_ = false;
        return;
    }

    initialized_ = true;
    GFL_LOG_DEBUG("Callbacks registered successfully.");
}

void CuptiBackend::shutdown() {
    if (!initialized_) return;

    if (active_.load(std::memory_order_relaxed)) {
        stop();
    }

    // Delegate engine teardown first
    if (engine_) {
        engine_->stop();
        engine_->shutdown();
    }

    // Belt-and-suspenders final drain. stop() already disabled all
    // activity kinds and flushed, but engine teardown above can emit a
    // last burst (e.g. PcSamplingEngine::shutdown's StopAndCollectPcSampling_
    // collection). Disabling SOURCE_LOCATOR + FUNCTION here matches what
    // PcSamplingEngine enables on the PC-sampling paths (no-op for engines
    // that never enabled them). The final flush
    // guarantees every BufferCompleted callback has returned before
    // we null g_activeBackend below — without this, late deliveries
    // raced the pointer-clear and surfaced as "No active backend!"
    // noise on benchmarks that init/shutdown gpufl repeatedly in a
    // single process (run_benchmark.py).
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_FUNCTION);
    cudaDeviceSynchronize();
    LogCuptiIfUnexpected("shutdown", "cuptiActivityFlushAll(final)",
                         cuptiActivityFlushAll(1));
    EmitCaptureCapabilities_();
    if (engine_) engine_.reset();

    cuptiUnsubscribe(subscriber_);
    g_activeBackend.store(nullptr, std::memory_order_release);
    initialized_ = false;
}

CUptiResult (*CuptiBackend::get_value())(CUpti_ActivityKind) {
    return cuptiActivityEnable;
}

void CuptiBackend::start() {
    if (!initialized_) return;
    kernel_activity_seen_.store(0, std::memory_order_relaxed);
    kernel_activity_emitted_.store(0, std::memory_order_relaxed);
    kernel_activity_throttled_.store(0, std::memory_order_relaxed);
    memory_activity_emitted_.store(0, std::memory_order_relaxed);
    external_correlation_seen_.store(0, std::memory_order_relaxed);
    source_locator_seen_.store(0, std::memory_order_relaxed);
    function_record_seen_.store(0, std::memory_order_relaxed);
    capture_capabilities_emitted_.store(false, std::memory_order_relaxed);

    // Resolve CUDA context/device before asking handlers for activity kinds.
    // SASS safe-mode policy is device dependent, and the policy log should
    // report the real SM version. Querying requiredActivityKinds() before this
    // point used device_id_=0 and could choose the wrong activity policy.
    const bool haveCudaContext = EnsureCudaContext(&ctx_);
    if (haveCudaContext) {
        cuptiGetDeviceId(ctx_, &device_id_);
        GetSMProps(device_id_);
        chip_name_ = getChipName(device_id_);
        cached_device_name_ = GetCurrentDeviceName();
    } else if (engine_) {
        GFL_LOG_ERROR(
            "[CuptiBackend] Failed to get CUDA context; "
            "engine will not start.");
    }

    if (IsSassProfilerMode()) {
        const ComputeCapability cc =
            GetComputeCapability(static_cast<int>(device_id_));
        const uint32_t cuptiVersion = GetCuptiVersion();
        GFL_LOG_DEBUG("[CuptiBackend] SASS activity policy: ",
                      UseSafeSassActivityDefaults() ? "safe" : "full",
                      " (sm=", cc.major, cc.minor,
                      ", cupti_version=", cuptiVersion, ")");
    }

    // SOURCE_LOCATOR + FUNCTION activity records feed only the Activity-API
    // PC-sampling source-correlation maps (g_sourceLocatorMap /
    // g_functionNameMap, read solely by the CUPTI_ACTIVITY_KIND_PC_SAMPLING
    // handler). PcSamplingEngine now enables them on the path that consumes
    // them (both on the ActivityAPI branch; SOURCE_LOCATOR also on the
    // SamplingAPI branch), so engines that don't PC-sample — SassMetrics,
    // RangeProfiler, Trace — no longer emit records nothing reads, and
    // PcSampling stops enabling them when it falls back to the new SamplingAPI
    // on CUDA 13.x.
    // MARKER records capture NVTX push/pop ranges. We enable this
    // unconditionally because:
    //   - GFL_SCOPE itself now emits NVTX ranges (see gpufl.cpp)
    //   - PyTorch / cuDNN / cuBLAS / NCCL emit NVTX automatically
    //   - The cost is ~zero when no NVTX traffic exists
    // Paired START/END records are merged into NvtxMarkerEvent below
    // in BufferCompleted.
    if (AllowSassMarkerActivity()) {
        CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER));
    } else {
        GFL_LOG_DEBUG(
            "[CuptiBackend] MARKER activity disabled in SASS profiler mode. "
            "Set GPUFL_SASS_ALLOW_MARKER_ACTIVITY=1 to test it.");
    }

    // SYNCHRONIZATION records capture every cudaStreamSynchronize /
    // cudaDeviceSynchronize / cudaEventSynchronize / cuStreamWaitEvent
    // call with start/end timestamps. Volume is mid-scale, no anchor
    // activity kind required (CUPTI emits these regardless of which
    // API kinds are enabled). Soft-fail on enable so a CUPTI build that
    // doesn't support the kind still lets the rest of collection work.
    if (opts_.enable_synchronization && AllowSassSyncActivity()) {
        const CUptiResult res_sync =
            cuptiActivityEnable(CUPTI_ACTIVITY_KIND_SYNCHRONIZATION);
        if (res_sync != CUPTI_SUCCESS) {
            LogCuptiIfUnexpected(
                "Synchronization",
                "cuptiActivityEnable(SYNCHRONIZATION)", res_sync);
        }
    }

    // MEMORY2 records capture cudaMalloc / cudaFree /
    // cudaMallocAsync / cudaFreeAsync / cudaMallocManaged with
    // address, bytes, and memoryKind. **Default-off** in v1 (see
    // InitOptions::enable_memory_tracking) because TF eager and
    // similar workloads can produce a high volume; opt-in until we
    // validate overhead in the field.
    //
    // Soft-fail on enable: older CUPTI versions (CUDA 11) shipped
    // without MEMORY2 — they had MEMORY (deprecated) which has a
    // different record shape. We don't try to fall back; if MEMORY2
    // isn't available we log and continue without F3 attribution.
    if (opts_.enable_memory_tracking && AllowSassMemory2Activity()) {
        const CUptiResult res_mem =
            cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMORY2);
        if (res_mem != CUPTI_SUCCESS) {
            LogCuptiIfUnexpected(
                "MemoryTracking",
                "cuptiActivityEnable(MEMORY2)", res_mem);
        }
    }

    // GRAPH_TRACE captures cudaGraphLaunch with aggregate timing
    // (one record per launch, not per node). **Default-off** in v1
    // because some Blackwell driver builds reset PC sampling on first
    // graph launch — the planning doc has the full risk note. Soft-
    // fail on enable so older CUPTI without the kind keeps working.
    if (opts_.enable_cuda_graphs_tracking && AllowSassGraphActivity()) {
        const CUptiResult res_g =
            cuptiActivityEnable(CUPTI_ACTIVITY_KIND_GRAPH_TRACE);
        if (res_g != CUPTI_SUCCESS) {
            LogCuptiIfUnexpected(
                "CudaGraphsTracking",
                "cuptiActivityEnable(GRAPH_TRACE)", res_g);
        }
    }

    // EXTERNAL_CORRELATION records appear when a framework brackets
    // its op with cuptiActivityPushExternalCorrelationId AND CUPTI has
    // a way to anchor each launch as an "event" worth correlating. The
    // anchoring uses CUPTI's activity-kind path (RUNTIME records),
    // NOT the cuptiSubscribe callback path we use for kernel-launch
    // metadata capture. So enabling EXTERNAL_CORRELATION alone is not
    // enough: CUPTI silently emits nothing because there's no
    // RUNTIME/DRIVER record to attach the external id to.
    //
    // We enable RUNTIME alongside EXTERNAL_CORRELATION as the smallest
    // sufficient anchor. RUNTIME captures cudaLaunchKernel / cudaMemcpy
    // / etc. as CUpti_ActivityAPI records — high volume in tight loops,
    // but we don't dispatch them anywhere; they fall through the
    // BufferCompleted handler chain and get freed with the buffer.
    // The cost is per-API-call activity record allocation in the CUPTI
    // buffer, not a per-call user callback.
    //
    // (DRIVER kind is RUNTIME's lower-level cousin. We choose RUNTIME
    // because PyTorch / TF / JAX call cudaLaunchKernel via the runtime
    // API, not the driver API directly. If a workload only uses cuLaunch
    // we may need DRIVER too — defer until we see a session that needs
    // it.)
    const bool enableExternalCorrelation =
        opts_.enable_external_correlation && AllowSassExternalCorrelation();
    if (opts_.enable_external_correlation && IsSassProfilerMode() &&
        !AllowSassExternalCorrelation()) {
        GFL_LOG_DEBUG(
            "[CuptiBackend] EXTERNAL_CORRELATION disabled in SASS profiler "
            "mode. Set GPUFL_SASS_ALLOW_EXTERNAL_CORRELATION=1 to test it.");
    }
    if (enableExternalCorrelation) {
        const CUptiResult res_ec =
            cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);
        if (res_ec != CUPTI_SUCCESS) {
            // Soft-fail: log and continue. If CUPTI doesn't know this kind
            // we still want kernel collection to work.
            LogCuptiIfUnexpected(
                "ExternalCorrelation",
                "cuptiActivityEnable(EXTERNAL_CORRELATION)", res_ec);
        }
        // RUNTIME + DRIVER are BOTH needed as anchors. CUPTI emits
        // EXTERNAL_CORRELATION records only for launches whose API
        // path was being tracked. PyTorch's `torch.randn` and most
        // memcpy ops go through the cudaLaunchKernel/cudaMemcpy
        // RUNTIME API, but optimized libraries like CUTLASS,
        // cuBLAS-Lt, and Triton's CUDA backend launch via the lower-
        // level cuLaunchKernel/cuLaunchKernelEx DRIVER API instead.
        //
        // Empirical evidence (Heavy_Stress_App, RTX 5060 + CUDA 13.1):
        // with only RUNTIME enabled, cutlass_sgemm kernels (51 out of
        // 53 in a typical session) get external_id = 0 because no
        // EXTERNAL_CORRELATION record is emitted for them. Adding
        // DRIVER fixes this without affecting RUNTIME-based ops.
        //
        // Cost: each cudaLaunchKernel produces TWO API records now
        // (one RUNTIME, one DRIVER) instead of one. They land in the
        // CUPTI buffer, fall through our handler chain unhandled,
        // and get freed with the buffer. Measured overhead in the
        // 50-iteration stress benchmark: <1% wall-clock difference.
        // Worth it for full F1 attribution coverage.
        const CUptiResult res_rt =
            cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);
        if (res_rt != CUPTI_SUCCESS) {
            LogCuptiIfUnexpected(
                "ExternalCorrelation",
                "cuptiActivityEnable(RUNTIME)", res_rt);
        }
        const CUptiResult res_drv =
            cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER);
        if (res_drv != CUPTI_SUCCESS) {
            LogCuptiIfUnexpected(
                "ExternalCorrelation",
                "cuptiActivityEnable(DRIVER)", res_drv);
        }
    }

    // NVTX v3 injection — register CUPTI as the NVTX provider.
    //
    // NVTX v3 (header-only since CUDA 11, mandatory in CUDA 13) works via
    // a lazy injection mechanism: on the *first* nvtxRangePushA/Pop call, it
    // reads the NVTX_INJECTION64_PATH environment variable, loads that DLL,
    // and calls its "InitializeInjectionNvtx2" export. CUPTI exports this
    // function, so pointing the env var at the already-loaded CUPTI DLL
    // makes every subsequent NVTX call route through CUPTI and produce
    // CUPTI_ACTIVITY_KIND_MARKER records.
    //
    // If NVTX_INJECTION64_PATH is not set, nvtxRangePushA is a silent no-op
    // (all calls go to the null provider) and no MARKER records are ever
    // produced, regardless of cuptiActivityEnable(MARKER) being called.
    //
    // We discover the CUPTI DLL path at runtime from the address of a CUPTI
    // function that is already loaded in this process — robust to non-standard
    // install locations and doesn't require CUDA_PATH to be set.
    // We only set the env var if it hasn't been set externally, so a user who
    // wants to use a custom injection library can still do so.
#if GPUFL_HAS_NVTX
    if (!getenv("NVTX_INJECTION64_PATH")) {
        // Discover the CUPTI DLL path by finding the loaded module that:
        //   (a) exports "InitializeInjectionNvtx2" (the symbol NVTX v3 calls)
        //   (b) is the same CUPTI we linked against — i.e., it lives under
        //       the CUDA toolkit (CUDA_PATH), NOT under a framework bundle.
        //
        // Context: PyTorch ships its own cupti64_*.dll under lib/. If we
        // blindly pick the first module that exports InitializeInjectionNvtx2
        // we may get PyTorch's CUPTI, which has a different subscriber context
        // than the CUDA toolkit CUPTI our backend already runs under.
        // Using the wrong CUPTI as the NVTX injection DLL causes CUPTI to
        // deliver NVTX marker records to the wrong subscriber (silent loss),
        // and in some cases causes a crash at the CUDA runtime boundary due
        // to two CUPTI instances both intercepting the same operations.
        //
        // Strategy: two-pass scan.
        //   Pass 0: prefer the module whose path begins with CUDA_PATH.
        //   Pass 1: accept any module that exports InitializeInjectionNvtx2
        //           (fallback for systems where CUDA_PATH is not set).
#if defined(_WIN32)
        const char* cudaPath = getenv("CUDA_PATH");
        HMODULE hMods[512];
        DWORD cbNeeded = 0;
        HANDLE hProcess = GetCurrentProcess();
        if (EnumProcessModules(hProcess, hMods, sizeof(hMods), &cbNeeded)) {
            DWORD count = cbNeeded / sizeof(HMODULE);
            if (count > 512) count = 512;

            char chosen[MAX_PATH] = {};
            for (DWORD pass = 0; pass < 2 && chosen[0] == '\0'; ++pass) {
                for (DWORD i = 0; i < count; ++i) {
                    if (!GetProcAddress(hMods[i], "InitializeInjectionNvtx2"))
                        continue;
                    char modPath[MAX_PATH] = {};
                    DWORD len = GetModuleFileNameA(hMods[i], modPath, MAX_PATH);
                    if (len == 0 || len >= MAX_PATH) continue;

                    if (pass == 0 && cudaPath) {
                        // Accept only if under the CUDA toolkit path
                        if (_strnicmp(modPath, cudaPath, strlen(cudaPath)) == 0) {
                            strncpy_s(chosen, modPath, MAX_PATH - 1);
                            break;
                        }
                    } else if (pass == 1) {
                        // Fallback: take the first we can find
                        strncpy_s(chosen, modPath, MAX_PATH - 1);
                        break;
                    }
                }
            }
            if (chosen[0]) {
                _putenv_s("NVTX_INJECTION64_PATH", chosen);
                GFL_LOG_DEBUG("[CuptiBackend] NVTX injection path: ", chosen);
            }
        }
#elif defined(__linux__)
        // On Linux, libcupti.so is loaded once. dlsym(RTLD_DEFAULT, ...) finds
        // whichever .so was loaded first; prefer CUDA_PATH if set.
        const char* cudaPath = getenv("CUDA_PATH");
        void* sym = nullptr;

        if (cudaPath) {
            // Build the expected libcupti.so path under CUDA_PATH
            std::string soPath = std::string(cudaPath) + "/extras/CUPTI/lib64/libcupti.so";
            void* hCupti = dlopen(soPath.c_str(), RTLD_NOLOAD | RTLD_LAZY);
            if (!hCupti) {
                soPath = std::string(cudaPath) + "/lib64/libcupti.so";
                hCupti = dlopen(soPath.c_str(), RTLD_NOLOAD | RTLD_LAZY);
            }
            if (hCupti) {
                sym = dlsym(hCupti, "InitializeInjectionNvtx2");
                if (sym) {
                    Dl_info info{};
                    if (dladdr(sym, &info) && info.dli_fname && info.dli_fname[0]) {
                        setenv("NVTX_INJECTION64_PATH", info.dli_fname, 0);
                        GFL_LOG_DEBUG("[CuptiBackend] NVTX injection path: ",
                                      info.dli_fname);
                    }
                }
                dlclose(hCupti);
            }
        }
        if (!sym) {
            // Fallback: find any loaded .so that exports the symbol
            sym = dlsym(RTLD_DEFAULT, "InitializeInjectionNvtx2");
            if (sym) {
                Dl_info info{};
                if (dladdr(sym, &info) && info.dli_fname && info.dli_fname[0]) {
                    setenv("NVTX_INJECTION64_PATH", info.dli_fname, 0);
                    GFL_LOG_DEBUG("[CuptiBackend] NVTX injection path: ",
                                  info.dli_fname);
                }
            }
        }
#endif
    }
#endif  // GPUFL_HAS_NVTX

    // Enable activity kinds required by registered handlers (always on)
    {
        std::set<CUpti_ActivityKind> kinds;
        {
            std::lock_guard<std::mutex> lk(handler_mu_);
            for (const auto& h : handlers_)
                for (auto k : h->requiredActivityKinds()) kinds.insert(k);
        }
        for (auto k : kinds) CUPTI_CHECK(cuptiActivityEnable(k));
    }

    // Initialize and start the engine (requires CUDA context)
    if (engine_ && haveCudaContext) {
        EngineContext ectx{ctx_, device_id_, chip_name_, &cubin_mu_,
                           &cubin_by_crc_};
        engine_->initialize(opts_, ectx);
        engine_->start();
    }

    // Re-enable activity kinds after engine start. Some engines call
    // cuptiProfilerInitialize() or cuptiSassMetricsEnable(), which on some
    // systems (e.g. insufficient profiler privileges) can internally reset or
    // disable previously-enabled activity kinds including
    // CUPTI_ACTIVITY_KIND_KERNEL.  Re-enabling here is idempotent and ensures
    // kernel activity records are produced regardless of engine type.
    {
        std::set<CUpti_ActivityKind> kinds;
        {
            std::lock_guard lk(handler_mu_);
            for (const auto& h : handlers_)
                for (auto k : h->requiredActivityKinds()) kinds.insert(k);
        }
        for (auto k : kinds) cuptiActivityEnable(k);
    }

    // also re-enable EXTERNAL_CORRELATION + RUNTIME after engine
    // start. The engines above (PcSampling, SassMetrics, RangeProfiler)
    // reset ALL activity-kind subscriptions, not just kernel-related
    // ones. Neither kind is tied to any handler, so they get dropped
    // from the re-enable set — this block restores them.
    //
    // RUNTIME is the anchor that makes EXTERNAL_CORRELATION actually
    // emit records (see the start-of-start() block for the rationale).
    if (enableExternalCorrelation) {
        const CUptiResult ec_res =
            cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);
        const CUptiResult rt_res =
            cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);
        const CUptiResult drv_res =
            cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER);
        GFL_LOG_DEBUG(
            "[CuptiBackend] re-enable EXTERNAL_CORRELATION post-engine: ",
            (ec_res == CUPTI_SUCCESS ? "OK" : "FAILED"),
            " (CUptiResult=", static_cast<int>(ec_res), ")",
            "; RUNTIME: ",
            (rt_res == CUPTI_SUCCESS ? "OK" : "FAILED"),
            " (CUptiResult=", static_cast<int>(rt_res), ")",
            "; DRIVER: ",
            (drv_res == CUPTI_SUCCESS ? "OK" : "FAILED"),
            " (CUptiResult=", static_cast<int>(drv_res), ")");
    } else {
        GFL_LOG_DEBUG(
            "[CuptiBackend] enable_external_correlation = false; "
            "no F1 attribution will be captured");
    }

    // F2: re-enable SYNCHRONIZATION post-engine for the same reason —
    // engines reset all activity-kind subscriptions during their
    // initialize() phase. Idempotent; CUPTI ignores the second enable
    // when the kind is already on. SYNCHRONIZATION isn't tied to any
    // handler so it would otherwise be silently dropped.
    if (opts_.enable_synchronization && AllowSassSyncActivity()) {
        const CUptiResult sync_res =
            cuptiActivityEnable(CUPTI_ACTIVITY_KIND_SYNCHRONIZATION);
        GFL_LOG_DEBUG(
            "[CuptiBackend] re-enable SYNCHRONIZATION post-engine: ",
            (sync_res == CUPTI_SUCCESS ? "OK" : "FAILED"),
            " (CUptiResult=", static_cast<int>(sync_res), ")");
    }

    // F3: matching post-engine re-enable for MEMORY2.
    if (opts_.enable_memory_tracking && AllowSassMemory2Activity()) {
        const CUptiResult mem_res =
            cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMORY2);
        GFL_LOG_DEBUG(
            "[CuptiBackend] re-enable MEMORY2 post-engine: ",
            (mem_res == CUPTI_SUCCESS ? "OK" : "FAILED"),
            " (CUptiResult=", static_cast<int>(mem_res), ")");
    }

    // F4: matching post-engine re-enable for GRAPH_TRACE.
    if (opts_.enable_cuda_graphs_tracking && AllowSassGraphActivity()) {
        const CUptiResult g_res =
            cuptiActivityEnable(CUPTI_ACTIVITY_KIND_GRAPH_TRACE);
        GFL_LOG_DEBUG(
            "[CuptiBackend] re-enable GRAPH_TRACE post-engine: ",
            (g_res == CUPTI_SUCCESS ? "OK" : "FAILED"),
            " (CUptiResult=", static_cast<int>(g_res), ")");
    }

    active_.store(true);
    GFL_LOG_DEBUG("Backend started.");
}


void CuptiBackend::EmitCaptureCapabilities_() const {
    const Runtime* rt = runtime();
    if (!(rt && rt->logger)) return;
    if (capture_capabilities_emitted_.exchange(true, std::memory_order_acq_rel)) {
        return;
    }

    const bool sassMode = IsSassProfilerMode();
    const bool pcSamplingMode =
        opts_.profiling_engine == ProfilingEngine::PcSampling;
    const bool kernelActivity =
        pcSamplingMode ? false : (!sassMode || AllowSassKernelActivity());
    const bool syntheticKernels = !kernelActivity && (sassMode || pcSamplingMode);
    const bool cubinCapture = NeedsCubinCapture();

    bool sassActive = false;
    bool pcActive = false;
    bool pmActive = false;

    // Capability emission happens after final engine shutdown so the report can
    // see late-flushed samples. Some engines drop their operational flag during
    // shutdown, so keep a path active if it was requested and produced data.
    if (opts_.profiling_engine == ProfilingEngine::SassMetrics && engine_) {
        sassActive = engine_->isOperational() || engine_->producedData();
    } else if (opts_.profiling_engine == ProfilingEngine::PcSampling && engine_) {
        pcActive = engine_->isOperational() || engine_->producedData();
    } else if (opts_.profiling_engine == ProfilingEngine::PmSampling && engine_) {
        pmActive = engine_->isOperational() || engine_->producedData();
    } else if (opts_.profiling_engine == ProfilingEngine::Deep) {
        if (const auto* deep =
                dynamic_cast<const PcSamplingWithSassEngine*>(engine_.get())) {
            sassActive = deep->sassActive() || deep->sassProducedData();
            pcActive = deep->pcSamplingActive() || deep->pcProducedData();
            pmActive = deep->pmSamplingActive() || deep->pmProducedData();
        }
    }

    // Did each path actually emit rows (vs merely arm)? Drives the
    // "enabled_no_data" status so a capability that was turned ON but produced
    // zero records is reported honestly instead of as "collected".
    const uint64_t kernelRows =
        kernel_activity_emitted_.load(std::memory_order_relaxed);
    const uint64_t memoryRows =
        memory_activity_emitted_.load(std::memory_order_relaxed);
    const uint64_t externalRows =
        external_correlation_seen_.load(std::memory_order_relaxed);
    const uint64_t sourceRows =
        source_locator_seen_.load(std::memory_order_relaxed);
    const uint64_t functionRows =
        function_record_seen_.load(std::memory_order_relaxed);
    const bool kernelHasData = kernelRows > 0;
    const bool memoryHasData = memoryRows > 0;
    const bool externalHasData = externalRows > 0;
    const bool sourceHasData = sourceRows > 0 || functionRows > 0;

    bool sassHasData = false;
    bool pcHasData = false;
    bool pmHasData = false;
    if (engine_) {
        if (const auto* deep =
                dynamic_cast<const PcSamplingWithSassEngine*>(engine_.get())) {
            sassHasData = deep->sassProducedData();
            pcHasData = deep->pcProducedData();
            pmHasData = deep->pmProducedData();
        } else if (opts_.profiling_engine == ProfilingEngine::SassMetrics) {
            sassHasData = engine_->producedData();
        } else if (opts_.profiling_engine == ProfilingEngine::PcSampling) {
            pcHasData = engine_->producedData();
        } else if (opts_.profiling_engine == ProfilingEngine::PmSampling) {
            pmHasData = engine_->producedData();
        }
    }

    std::string selected = ProfilingEngineWireName(opts_.profiling_engine);
    if (opts_.profiling_engine == ProfilingEngine::Deep) {
        if (sassActive) selected = "nvidia.sass_metrics";
        else if (pcActive) selected = "nvidia.pc_sampling";
        else selected = "nvidia.none";
    }

    CaptureCapabilitiesEvent evt;
    evt.session_id = rt->session_id;
    evt.ts_ns = detail::GetTimestampNs();
    evt.requested_engine = ProfilingEngineWireName(opts_.profiling_engine);
    evt.selected_engine = selected;

    const bool metricsOnly = SassMetricsOnlyMode();
    AddCapability(evt, "kernel_events", true,
                  kernelHasData
                      ? (syntheticKernels ? "fallback" : "collected")
                      : (metricsOnly ? "skipped" : "enabled_no_data"),
                  metricsOnly
                      ? "sass_metrics_only"
                      : (syntheticKernels ? "launch_callbacks_synthetic" : "cupti_activity"),
                  kernelHasData
                      ? (syntheticKernels
                            ? (pcSamplingMode ? "cupti_kernel_activity_conflicts_with_pc_sampling"
                                              : "cupti_kernel_activity_deadlock_risk")
                            : "")
                      : (metricsOnly ? "disabled_to_preserve_sass_counters"
                                     : "enabled_but_no_records"),
                  kernelHasData
                      ? (syntheticKernels
                            ? (pcSamplingMode
                                  ? "Kernel rows were collected from launch callbacks; durations are estimated because CUPTI kernel activity is disabled while PC Sampling API is active."
                                  : "Kernel rows were collected from launch callbacks; durations are estimated because CUPTI kernel activity is disabled in SASS safe mode.")
                            : "Kernel rows were collected from CUPTI kernel activity records.")
                      : (metricsOnly
                            ? "Kernel activity was intentionally disabled because CUPTI SASS Metrics requires metrics-only mode on this GPU/driver to produce non-zero counters."
                            : "Kernel tracing was enabled but emitted no kernel rows this session."));
    AddCapability(evt, "kernel_names", true,
                  kernelHasData
                      ? (syntheticKernels ? "partial" : "collected")
                      : (metricsOnly ? "skipped" : "enabled_no_data"),
                  metricsOnly
                      ? "sass_metrics_only"
                      : (syntheticKernels ? "callback_symbol_probe" : "cupti_activity_name"),
                  kernelHasData
                      ? (syntheticKernels ? "symbol_name_may_be_unavailable" : "")
                      : (metricsOnly ? "disabled_to_preserve_sass_counters"
                                     : "enabled_but_no_records"),
                  kernelHasData
                      ? (syntheticKernels
                            ? "Kernel names use CUPTI callback symbolName when safely readable, otherwise the CUDA launch API name."
                            : "Kernel names came from CUPTI activity records.")
                      : (metricsOnly
                            ? "Kernel name tracing was intentionally disabled to keep SASS metric counters valid."
                            : "Kernel name capture was enabled but no kernel rows were emitted."));
    AddCapability(evt, "kernel_details", true,
                  kernelHasData
                      ? (syntheticKernels ? "partial" : "collected")
                      : (metricsOnly ? "skipped" : "enabled_no_data"),
                  metricsOnly
                      ? "sass_metrics_only"
                      : (syntheticKernels ? "launch_callback_params" : "cupti_activity_details"),
                  kernelHasData
                      ? (syntheticKernels ? "activity_details_unavailable" : "")
                      : (metricsOnly ? "disabled_to_preserve_sass_counters"
                                     : "enabled_but_no_records"),
                  kernelHasData
                      ? (syntheticKernels
                            ? "Grid/block parameters are captured from launch callbacks; register and occupancy details may be unavailable."
                            : "Kernel details came from CUPTI activity records and launch metadata.")
                      : (metricsOnly
                            ? "Kernel detail tracing was intentionally disabled to keep SASS metric counters valid."
                            : "Kernel detail capture was enabled but no kernel rows were emitted."));
    AddCapability(evt, "cubin_disassembly", cubinCapture,
                  cubinCapture ? "collected" : "not_requested",
                  cubinCapture ? "module_resource_callbacks" : "disabled",
                  "",
                  cubinCapture
                      ? "CUBINs were captured for offline SASS disassembly."
                      : "This profiling engine does not request CUBIN capture.");
    AddCapability(evt, "sass_metrics",
                  opts_.profiling_engine == ProfilingEngine::SassMetrics ||
                      opts_.profiling_engine == ProfilingEngine::Deep,
                  sassActive ? (sassHasData ? "collected" : "enabled_no_data") :
                      ((opts_.profiling_engine == ProfilingEngine::SassMetrics ||
                        opts_.profiling_engine == ProfilingEngine::Deep) ? "skipped" : "not_requested"),
                  sassActive ? "cupti_sass_metrics" : "disabled",
                  sassActive ? (sassHasData ? "" : "enabled_but_no_samples")
                             : "not_selected_or_not_operational",
                  sassActive
                      ? (sassHasData
                            ? "SASS metrics were collected for this session."
                            : "SASS metrics were enabled but produced no instruction-level samples this session (e.g. kernels too short, or CUPTI replay returned no data).")
                      : "SASS metrics were not collected for this session.");
    AddCapability(evt, "pc_sampling",
                  opts_.profiling_engine == ProfilingEngine::PcSampling ||
                      opts_.profiling_engine == ProfilingEngine::Deep,
                  pcActive ? (pcHasData ? "collected" : "enabled_no_data") :
                      (opts_.profiling_engine == ProfilingEngine::Deep && sassActive ? "skipped" :
                       (opts_.profiling_engine == ProfilingEngine::PcSampling ? "skipped" : "not_requested")),
                  pcActive ? "cupti_pc_sampling" : "disabled",
                  opts_.profiling_engine == ProfilingEngine::Deep && sassActive
                      ? "mutually_exclusive_with_sass_metrics" :
                    (pcActive ? (pcHasData ? "" : "enabled_but_no_samples")
                              : "not_selected_or_not_operational"),
                  opts_.profiling_engine == ProfilingEngine::Deep && sassActive
                      ? "Deep selected SASS metrics; PC sampling was skipped because SASS metrics and PC sampling are mutually exclusive in one run."
                      : (pcActive ? (pcHasData
                            ? "PC sampling was collected for this session."
                            : "PC sampling was enabled but produced no stall samples this session (e.g. kernels too short for the sampling period).")
                                  : "PC sampling was not collected for this session."));
    AddCapability(evt, "pm_sampling",
                  opts_.profiling_engine == ProfilingEngine::PmSampling ||
                      opts_.profiling_engine == ProfilingEngine::Deep,
                  pmActive ? (pmHasData ? "collected" : "enabled_no_data")
                           : ((opts_.profiling_engine == ProfilingEngine::PmSampling ||
                               opts_.profiling_engine == ProfilingEngine::Deep) ? "skipped" : "not_requested"),
                  pmActive ? "cupti_pm_sampling" : "disabled",
                  pmActive ? (pmHasData ? "" : "enabled_but_no_samples")
                           : "not_selected_or_not_operational",
                  pmActive
                      ? (pmHasData
                            ? "PM sampling hardware metric samples were collected for this session."
                            : "PM sampling was enabled but produced no hardware samples this session.")
                      : "PM sampling was not collected for this session.");
    AddCapability(evt, "source_correlation", pcActive,
                  pcActive ? (sourceHasData ? "collected" : "enabled_no_data")
                           : (sassActive ? "skipped" : "not_requested"),
                  pcActive ? "pc_sampling_source_locator" : "disabled",
                  pcActive ? (sourceHasData ? "" : "enabled_but_no_records")
                           : (sassActive ? "sass_metrics_have_no_source_lines" : "not_requested"),
                  pcActive
                      ? (sourceHasData
                            ? "PC sampling source locator/function records were collected for CUDA source correlation."
                            : "PC sampling source correlation was enabled but emitted no source locator/function records.")
                      : "CUDA source-line correlation was not collected in this session.");
    const bool memoryRequestedAndAllowed =
        opts_.enable_memory_tracking && AllowSassMemory2Activity();
    AddCapability(evt, "memory_activity", opts_.enable_memory_tracking,
                  memoryRequestedAndAllowed
                      ? (memoryHasData ? "collected" : "enabled_no_data")
                      : (opts_.enable_memory_tracking ? "skipped" : "not_requested"),
                  memoryRequestedAndAllowed ? "cupti_memory" : "disabled",
                  memoryRequestedAndAllowed
                      ? (memoryHasData ? "" : "enabled_but_no_records")
                      : (opts_.enable_memory_tracking && !AllowSassMemory2Activity()
                            ? "sass_safe_mode_memory_activity_disabled" : ""),
                  memoryRequestedAndAllowed
                      ? (memoryHasData
                            ? "CUPTI memory activity records were collected."
                            : "CUPTI memory activity was enabled but emitted no memory rows this session.")
                      : "CUPTI memory activity was not collected.");
    const bool externalRequestedAndAllowed =
        opts_.enable_external_correlation && AllowSassExternalCorrelation();
    AddCapability(evt, "external_correlation", opts_.enable_external_correlation,
                  externalRequestedAndAllowed
                      ? (externalHasData ? "collected" : "enabled_no_data")
                      : (opts_.enable_external_correlation ? "skipped" : "not_requested"),
                  externalRequestedAndAllowed ? "cupti_external_correlation" : "disabled",
                  externalRequestedAndAllowed
                      ? (externalHasData ? "" : "enabled_but_no_records")
                      : (opts_.enable_external_correlation && !AllowSassExternalCorrelation()
                            ? "sass_safe_mode_external_correlation_disabled" : ""),
                  externalRequestedAndAllowed
                      ? (externalHasData
                            ? "Framework external correlation records were collected."
                            : "Framework external correlation was enabled but emitted no records this session.")
                      : "Framework external correlation was not collected.");

    rt->logger->write(model::CaptureCapabilitiesModel(evt));
}

void CuptiBackend::FlushProfilingDataBeforeCudaTeardown(const char* reason) {
    if (!initialized_ || !active_.load(std::memory_order_relaxed) || !engine_) {
        return;
    }
    if (!IsSassProfilerMode()) return;

    const int64_t now = detail::GetTimestampNs();
    int64_t expected = 0;
    if (!last_cleanup_flush_ns_.compare_exchange_strong(
            expected, now, std::memory_order_acq_rel, std::memory_order_relaxed)) {
        return;
    }

    engine_->flushBeforeCudaTeardown(reason);
}

void CuptiBackend::stop() {
    if (!initialized_) return;
    active_.store(false);

    // Stop the engine BEFORE flushing activity records.  PcSamplingEngine::stop()
    // disables the SamplingAPI session — while it's armed, cuptiActivityFlushAll
    // returns zero kernel records on driver 590+.
    if (engine_) engine_->stop();

    // Disable all activity kinds FIRST, before the flush. The previous
    // order (sync → flush → disable) left activity tracking enabled
    // during the flush, so new records could be queued by the GPU
    // while we were draining. Those new records then arrived AFTER
    // shutdown() had cleared g_activeBackend, firing the noisy
    // "[CUPTI] BufferCompleted: No active backend!" log and (worse)
    // leaking activity into the next session's measurement on
    // benchmarks that init/shutdown gpufl repeatedly in one process —
    // run_benchmark.py's GEMM→PyTorch transition is the canonical
    // case where this surfaced (RTX 3090 + Linux). Disabling first
    // closes the queue so the subsequent flush truly drains
    // everything pending.
    //
    // Already-queued records are NOT dropped by cuptiActivityDisable;
    // they still come back through BufferCompleted during the flush
    // below. So we don't lose any data by disabling first.
    {
        std::set<CUpti_ActivityKind> kinds;
        {
            std::lock_guard lk(handler_mu_);
            for (const auto& h : handlers_)
                for (auto k : h->requiredActivityKinds()) kinds.insert(k);
        }
        for (auto k : kinds) cuptiActivityDisable(k);
    }

    // Matching tear-down of the same anchor / supplementary kinds
    // start() enables. Same disable-before-flush rationale — keeps
    // the activity queue fully closed before drain.
    if (opts_.enable_external_correlation) {
        cuptiActivityDisable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);
        cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME);
        cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER);
    }
    if (opts_.enable_synchronization) {
        cuptiActivityDisable(CUPTI_ACTIVITY_KIND_SYNCHRONIZATION);
    }
    if (opts_.enable_memory_tracking) {
        cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMORY2);
    }
    if (opts_.enable_cuda_graphs_tracking) {
        cuptiActivityDisable(CUPTI_ACTIVITY_KIND_GRAPH_TRACE);
    }

    // Now drain. cuda sync first so any in-flight kernel finishes and
    // emits its end record; flush then blocks until every BufferCompleted
    // callback has returned. With kinds disabled above, no new records
    // can sneak into the queue during this window.
    cudaDeviceSynchronize();
    LogCuptiIfUnexpected("Perfworks", "cuptiActivityFlushAll",
                         cuptiActivityFlushAll(1));
    FlushPendingKernels();
    {
        std::lock_guard lk(g_extCorrMu);
        g_extCorrMap.clear();
    }

    const uint64_t seen = kernel_activity_seen_.load(std::memory_order_relaxed);
    const uint64_t emitted =
        kernel_activity_emitted_.load(std::memory_order_relaxed);
    const uint64_t throttled =
        kernel_activity_throttled_.load(std::memory_order_relaxed);
    GFL_LOG_DEBUG("[KernelLaunchHandler] activity summary seen=", seen,
                  " emitted=", emitted, " throttled=", throttled);
}

void CuptiBackend::RegisterHandler(
    const std::shared_ptr<ICuptiHandler>& handler) {
    if (!handler) return;
    std::lock_guard lk(handler_mu_);
    handlers_.push_back(handler);
}

void CuptiBackend::FlushPendingKernels() {
    const int64_t flushNs = detail::GetTimestampNs();
    std::unordered_map<uint64_t, LaunchMeta> pending;
    {
        std::lock_guard lk(meta_mu_);
        pending = std::move(meta_by_corr_);
    }
    GFL_LOG_DEBUG("[FlushPendingKernels] draining ", pending.size(),
                  " synthetic kernel(s) at flushNs=", flushNs);

    // Build an order-by-api-enter index so we can approximate per-kernel
    // GPU time as the interval between this kernel's dispatch and the
    // next kernel's dispatch (or the scope flush time for the last one).
    //
    // CUPTI did NOT deliver activity records for these kernels (common on
    // Blackwell with PC Sampling SamplingAPI: PC Sampling captures the
    // kernel stream, so KERNEL activity kinds stay dormant). Without GPU
    // timestamps we fall back to host-side dispatch intervals — an
    // imperfect proxy that nonetheless reflects the actual sequential
    // execution pattern for typical single-stream workloads and is
    // strictly better than reporting 0.
    std::vector<uint64_t> orderedCorr;
    orderedCorr.reserve(pending.size());
    for (const auto& [corr, _] : pending) orderedCorr.push_back(corr);
    std::sort(orderedCorr.begin(), orderedCorr.end(),
              [&](uint64_t a, uint64_t b) {
                  return pending[a].api_enter_ns < pending[b].api_enter_ns;
              });

    for (size_t i = 0; i < orderedCorr.size(); ++i) {
        const uint64_t corr = orderedCorr[i];
        auto& m = pending[corr];
        ActivityRecord out{};
        out.device_id = device_id_;
        out.stream = 0;
        out.type = TraceType::KERNEL;
        std::snprintf(out.name, sizeof(out.name), "%s", m.name);
        out.cpu_start_ns = m.api_enter_ns;

        // Synthetic duration: interval until the next kernel's dispatch
        // (they run sequentially on the default stream of single-stream
        // workloads, so one's completion roughly aligns with the next
        // one's dispatch return) — or flushNs for the last kernel
        // (it completed between its dispatch and our post-sync flush).
        const int64_t nextEnterNs = (i + 1 < orderedCorr.size())
            ? pending[orderedCorr[i + 1]].api_enter_ns
            : flushNs;
        int64_t synthDur = nextEnterNs - m.api_enter_ns;
        if (synthDur < 0) synthDur = 0;  // clock skew guard
        out.duration_ns = synthDur;
        out.corr_id = static_cast<unsigned>(corr);
        out.api_start_ns = m.api_enter_ns;
        out.api_exit_ns = m.api_exit_ns > 0 ? m.api_exit_ns : flushNs;
        out.scope_depth = m.scope_depth;
        out.stack_id = m.stack_id;
        std::copy(std::begin(m.user_scope), std::end(m.user_scope),
                  std::begin(out.user_scope));
        if (m.has_details) {
            out.has_details = true;
            out.grid_x = m.grid_x;
            out.grid_y = m.grid_y;
            out.grid_z = m.grid_z;
            out.block_x = m.block_x;
            out.block_y = m.block_y;
            out.block_z = m.block_z;
            out.dyn_shared = m.dyn_shared;

            SmProps props = GetSMProps(out.device_id);
            int threadsPerBlock =
                out.block_x * out.block_y * out.block_z;
            int warpsPerBlock =
                (threadsPerBlock + props.warpSize - 1) / props.warpSize;
            int maxWarpsPerSM = props.maxThreadsPerSM / props.warpSize;
            int warpBlocks = (warpsPerBlock > 0)
                                 ? (maxWarpsPerSM / warpsPerBlock) : 0;
            int blockBlocks = props.maxBlocksPerSM;
            out.max_active_blocks = std::min(warpBlocks, blockBlocks);
            auto toOcc = [&](int blocks) -> float {
                return (maxWarpsPerSM > 0 && warpsPerBlock > 0)
                           ? std::min(1.0f,
                                      static_cast<float>(
                                          blocks * warpsPerBlock) /
                                          maxWarpsPerSM)
                           : 0.0f;
            };
            out.warp_occupancy = toOcc(warpBlocks);
            out.block_occupancy = toOcc(blockBlocks);
            out.occupancy = out.warp_occupancy;
            std::snprintf(out.limiting_resource,
                          sizeof(out.limiting_resource), "%s", "warps");
        }
        GFL_LOG_DEBUG("[FlushPendingKernels] synth corr=", corr,
                      " name=", out.name,
                      " scope=", out.user_scope,
                      " dur=", out.duration_ns, "ns");
        g_monitorBuffer.Push(out);
        kernel_activity_seen_.fetch_add(1, std::memory_order_relaxed);
        kernel_activity_emitted_.fetch_add(1, std::memory_order_relaxed);
    }
}

// ---- Static callbacks ------------------------------------------------------

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
    auto* backend = g_activeBackend.load(std::memory_order_acquire);
    if (!backend) {
        DebugLogger::error("[CUPTI] ",
                                    "BufferCompleted: No active backend!");
        if (buffer) free(buffer);
        return;
    }

    static int64_t baseCpuNs = detail::GetTimestampNs();
    static uint64_t baseCuptiTs = 0;
    if (baseCuptiTs == 0) cuptiGetTimestamp(&baseCuptiTs);

    std::vector<std::shared_ptr<ICuptiHandler>> handlers;
    {
        std::lock_guard lk(backend->handler_mu_);
        handlers = backend->handlers_;
    }

    if (validSize > 0) {
        // ----------------------------------------------------------------
        // Two-pass dispatch.
        //
        // Within a single CUPTI buffer flush, KERNEL records and
        // EXTERNAL_CORRELATION records arrive interleaved. The handler
        // chain stamps a kernel's external_kind/external_id by looking
        // up its correlationId in g_extCorrMap — but if the matching
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
        // iteration state — passing nullptr starts a fresh walk from
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
                {
                    std::lock_guard lk(g_extCorrMu);
                    g_extCorrMap[ec->correlationId] = ExternalCorrInfo{
                        static_cast<uint8_t>(ec->externalKind),
                        ec->externalId,
                    };
                }
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
                // Skip EXTERNAL_CORRELATION — already stored by pass 1.
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
                            std::lock_guard lk(g_sourceLocatorMu);
                            g_sourceLocatorMap[sl->id] = {sl->fileName,
                                                          sl->lineNumber};
                            backend->source_locator_seen_.fetch_add(
                                1, std::memory_order_relaxed);
                        }
                    } else if (record->kind == CUPTI_ACTIVITY_KIND_FUNCTION) {
                        auto* fn = reinterpret_cast<
                            const CUpti_ActivityFunction*>(record);
                        if (fn->name) {
                            std::lock_guard lk(g_sourceLocatorMu);
                            g_functionNameMap[fn->id] = fn->name;
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
                            std::lock_guard lk(g_sourceLocatorMu);
                            auto slIt = g_sourceLocatorMap.find(
                                pc->sourceLocatorId);
                            if (slIt != g_sourceLocatorMap.end()) {
                                std::snprintf(
                                    out.source_file,
                                    sizeof(out.source_file), "%s",
                                    slIt->second.first.c_str());
                                out.source_line = slIt->second.second;
                            }
                            auto fnIt =
                                g_functionNameMap.find(pc->functionId);
                            if (fnIt != g_functionNameMap.end()) {
                                std::snprintf(
                                    out.function_name,
                                    sizeof(out.function_name), "%s",
                                    fnIt->second.c_str());
                            }
                        }
                        g_monitorBuffer.Push(out);
                    } else if (record->kind == CUPTI_ACTIVITY_KIND_MARKER) {
                        // NVTX markers arrive as paired START/END records.
                        // Pair them by id to emit one ActivityRecord per
                        // completed range (TraceType::NVTX_MARKER, consumed
                        // by CollectorLoop → NvtxMarkerModel JSON).
                        auto* m = reinterpret_cast<
                            const CUpti_ActivityMarker2*>(record);
                        const bool isStart =
                            (m->flags & CUPTI_ACTIVITY_FLAG_MARKER_START) != 0;
                        const bool isEnd =
                            (m->flags & CUPTI_ACTIVITY_FLAG_MARKER_END)   != 0;

                        if (isStart) {
                            NvtxOpen entry;
                            entry.name     = m->name   ? m->name   : "";
                            entry.domain   = m->domain ? m->domain : "";
                            entry.start_ts = m->timestamp;
                            std::lock_guard<std::mutex> lk(g_nvtxMu);
                            g_nvtxOpen[m->id] = std::move(entry);
                        } else if (isEnd) {
                            NvtxOpen entry;
                            bool found = false;
                            {
                                std::lock_guard<std::mutex> lk(g_nvtxMu);
                                auto it = g_nvtxOpen.find(m->id);
                                if (it != g_nvtxOpen.end()) {
                                    entry = std::move(it->second);
                                    g_nvtxOpen.erase(it);
                                    found = true;
                                }
                            }
                            if (found) {
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
                                out.duration_ns  = end_wall - start_wall;
                                out.corr_id      = m->id;
                                // Domain stored in user_scope slot for now
                                // (CollectorLoop passes it to the event).
                                std::snprintf(out.user_scope,
                                              sizeof(out.user_scope), "%s",
                                              entry.domain.c_str());
                                g_monitorBuffer.Push(out);
                            }
                        }
                        // Other flag values (e.g. SYNC-only points) are
                        // ignored in v1; can be added as instantaneous
                        // events later if needed.
                    } else if (record->kind ==
                               CUPTI_ACTIVITY_KIND_GRAPH_TRACE) {
                        // F4: cudaGraphLaunch with aggregate timing.
                        // CUPTI gives one record per launch. start/end
                        // are in CUPTI's clock domain — convert to
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
                        out.type         = TraceType::GRAPH_LAUNCH;
                        out.cpu_start_ns = start_wall;
                        out.duration_ns  = dur;
                        out.device_id    = g->deviceId;
                        out.stream       = g->streamId;
                        out.corr_id      = g->correlationId;
                        out.graph_id     = g->graphId;
                        g_monitorBuffer.Push(out);
                    } else if (record->kind ==
                               CUPTI_ACTIVITY_KIND_MEMORY2) {
                        // F3: cudaMalloc / cudaFree / cudaMallocAsync /
                        // cudaMallocManaged / cudaMallocHost. CUPTI's
                        // CUpti_ActivityMemory4 carries one timestamp
                        // (the host call ts) but no end timestamp —
                        // duration_ns is left at 0 in v1; if users
                        // need host-side cost we'd correlate against
                        // the matching cuptiActivity API record (DEFER).
                        auto* m = reinterpret_cast<
                            const CUpti_ActivityMemory4*>(record);
                        const int64_t ts_wall =
                            static_cast<int64_t>(m->timestamp) -
                            static_cast<int64_t>(baseCuptiTs) + baseCpuNs;

                        ActivityRecord out{};
                        out.type         = TraceType::MEMORY_ALLOC;
                        out.cpu_start_ns = ts_wall;
                        out.duration_ns  = 0;
                        out.bytes        = m->bytes;
                        out.address      = m->address;
                        out.memory_op    = static_cast<uint8_t>(m->memoryOperationType);
                        out.memory_kind  = static_cast<uint8_t>(m->memoryKind);
                        out.device_id    = m->deviceId;
                        out.stream       = m->streamId;
                        out.corr_id      = m->correlationId;
                        g_monitorBuffer.Push(out);
                        backend->memory_activity_emitted_.fetch_add(
                            1, std::memory_order_relaxed);
                    } else if (record->kind ==
                               CUPTI_ACTIVITY_KIND_SYNCHRONIZATION) {
                        // F2: cudaStreamSynchronize / cudaDeviceSynchronize
                        // / cudaEventSynchronize / cuStreamWaitEvent timing.
                        // CUPTI delivers exactly one record per call, with
                        // wall-clock start/end already converted to ns.
                        // We push directly to the monitor ring buffer —
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
                        out.type          = TraceType::SYNCHRONIZATION;
                        out.cpu_start_ns  = start_wall;
                        out.duration_ns   = dur;
                        out.corr_id       = s->correlationId;
                        out.stream        = s->streamId;
                        out.sync_type     = static_cast<uint8_t>(s->type);
                        out.sync_event_id = s->cudaEventId;
                        out.context_id    = s->contextId;
                        // Join the user call stack captured by
                        // SynchronizationHandler on API_ENTER. Erasing
                        // after lookup keeps sync_meta_by_corr_ bounded
                        // — sync activity records are 1:1 with the
                        // API call, unlike kernel correlations that
                        // can span multiple activity records.
                        // (BufferCompleted is a static method, so we
                        // access non-static state through the backend
                        // pointer captured at line 851.)
                        {
                            std::lock_guard lk(backend->sync_meta_mu_);
                            auto smIt = backend->sync_meta_by_corr_.find(s->correlationId);
                            if (smIt != backend->sync_meta_by_corr_.end()) {
                                out.stack_id = smIt->second.stack_id;
                                backend->sync_meta_by_corr_.erase(smIt);
                            }
                        }
                        g_monitorBuffer.Push(out);
                    }
                    // (EXTERNAL_CORRELATION handled in pass 1 above —
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

    std::vector<std::shared_ptr<ICuptiHandler>> handlers;
    {
        std::lock_guard<std::mutex> lk(backend->handler_mu_);
        handlers = backend->handlers_;
    }

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

#include "gpufl/backends/nvidia/cupti_backend.hpp"

#include <cupti_pcsampling.h>
#include <cupti_profiler_target.h>
#include <cupti_sass_metrics.h>
#include <cupti_target.h>

#if GPUFL_HAS_PERFWORKS
#include <cupti_range_profiler.h>
#endif

#include <cstring>
#include <exception>
#include <set>

#include "gpufl/backends/nvidia/cuda_collector.hpp"
#include "gpufl/backends/nvidia/cupti_utils.hpp"
#include "gpufl/backends/nvidia/engine/pc_sampling_engine.hpp"
#include "gpufl/backends/nvidia/engine/pc_sampling_with_sass_engine.hpp"
#include "gpufl/backends/nvidia/engine/range_profiler_engine.hpp"
#include "gpufl/backends/nvidia/engine/sass_metrics_engine.hpp"
#include "gpufl/backends/nvidia/kernel_launch_handler.hpp"
#include "gpufl/backends/nvidia/mem_transfer_handler.hpp"
#include "gpufl/backends/nvidia/resource_handler.hpp"
#include "gpufl/core/activity_record.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/ring_buffer.hpp"
#include "gpufl/core/scope_registry.hpp"
#include "gpufl/core/stack_registry.hpp"
#include "gpufl/core/stack_trace.hpp"
#include "gpufl/core/trace_type.hpp"

namespace gpufl {
std::atomic<gpufl::CuptiBackend*> g_activeBackend{nullptr};

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
        case ProfilingEngine::RangeProfiler:
#if GPUFL_HAS_PERFWORKS
            engine_ = std::make_unique<RangeProfilerEngine>();
            GFL_LOG_DEBUG("[CuptiBackend] Engine: RangeProfiler");
#else
            GFL_LOG_ERROR(
                "[CuptiBackend] RangeProfiler engine requires "
                "GPUFL_HAS_PERFWORKS; falling back to None");
#endif
            break;
        case ProfilingEngine::PcSamplingWithSass:
            engine_ = std::make_unique<PcSamplingWithSassEngine>();
            GFL_LOG_DEBUG("[CuptiBackend] Engine: PcSamplingWithSass");
            break;
        case ProfilingEngine::None:
        default:
            GFL_LOG_DEBUG("[CuptiBackend] Engine: None (monitoring only)");
            break;
    }

    g_activeBackend.store(this, std::memory_order_release);

    // Internal handler registration
    RegisterHandler(std::make_shared<ResourceHandler>(this));
    RegisterHandler(std::make_shared<KernelLaunchHandler>(this));
    RegisterHandler(std::make_shared<MemTransferHandler>(this));

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
        engine_.reset();
    }

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

    CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR));

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
    if (engine_) {
        if (EnsureCudaContext(&ctx_)) {
            cuptiGetDeviceId(ctx_, &device_id_);
            GetSMProps(device_id_);
            chip_name_ = getChipName(device_id_);
            cached_device_name_ = GetCurrentDeviceName();

            EngineContext ectx{ctx_, device_id_, chip_name_, &cubin_mu_,
                               &cubin_by_crc_};
            engine_->initialize(opts_, ectx);
            engine_->start();
        } else {
            GFL_LOG_ERROR(
                "[CuptiBackend] Failed to get CUDA context; "
                "engine will not start.");
        }
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
            std::lock_guard<std::mutex> lk(handler_mu_);
            for (const auto& h : handlers_)
                for (auto k : h->requiredActivityKinds()) kinds.insert(k);
        }
        for (auto k : kinds) cuptiActivityEnable(k);
    }

    active_.store(true);
    GFL_LOG_DEBUG("Backend started.");
}

void CuptiBackend::stop() {
    if (!initialized_) return;
    active_.store(false);

    LogCuptiIfUnexpected("Perfworks", "cuptiActivityFlushAll",
                         cuptiActivityFlushAll(1));

    {
        std::set<CUpti_ActivityKind> kinds;
        {
            std::lock_guard<std::mutex> lk(handler_mu_);
            for (const auto& h : handlers_)
                for (auto k : h->requiredActivityKinds()) kinds.insert(k);
        }
        for (auto k : kinds) cuptiActivityDisable(k);
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
    for (auto& [corr, m] : pending) {
        ActivityRecord out{};
        out.device_id = device_id_;
        out.stream = 0;
        out.type = TraceType::KERNEL;
        std::snprintf(out.name, sizeof(out.name), "%s", m.name);
        out.cpu_start_ns = m.api_enter_ns;
        out.duration_ns = flushNs - m.api_enter_ns;
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
        CUpti_Activity* record = nullptr;
        while (true) {
            const CUptiResult st =
                cuptiActivityGetNextRecord(buffer, validSize, &record);
            if (st == CUPTI_SUCCESS) {
                bool handled = false;
                for (const auto& h : handlers) {
                    if (h->handleActivityRecord(record, baseCpuNs,
                                                baseCuptiTs)) {
                        handled = true;
                        break;
                    }
                }
                if (!handled &&
                    record->kind == CUPTI_ACTIVITY_KIND_PC_SAMPLING) {
                    auto* pc =
                        reinterpret_cast<CUpti_ActivityPCSampling3*>(record);
                    ActivityRecord out{};
                    out.type = TraceType::PC_SAMPLE;
                    out.corr_id = pc->correlationId;
                    std::snprintf(out.sample_kind, sizeof(out.sample_kind),
                                  "%s", "pc_sampling");
                    out.samples_count = pc->samples;
                    out.stall_reason = pc->stallReason;
                    out.device_id =
                        reinterpret_cast<const CUpti_ActivityKernel11*>(record)
                            ->deviceId;
                    g_monitorBuffer.Push(out);
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

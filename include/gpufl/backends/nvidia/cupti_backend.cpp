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

#include "gpufl/backends/nvidia/cupti_utils.hpp"
#include "gpufl/backends/nvidia/engine/pc_sampling_engine.hpp"
#include "gpufl/backends/nvidia/engine/range_profiler_engine.hpp"
#include "gpufl/backends/nvidia/engine/sass_metrics_engine.hpp"
#include "gpufl/backends/nvidia/kernel_launch_handler.hpp"
#include "gpufl/backends/nvidia/mem_transfer_handler.hpp"
#include "gpufl/backends/nvidia/resource_handler.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/ring_buffer.hpp"
#include "gpufl/core/trace_type.hpp"

#include "gpufl/backends/nvidia/cuda_collector.hpp"
#include "gpufl/core/scope_registry.hpp"
#include "gpufl/core/stack_registry.hpp"
#include "gpufl/core/stack_trace.hpp"

namespace gpufl {
std::atomic<gpufl::CuptiBackend*> g_activeBackend{nullptr};

extern RingBuffer<ActivityRecord, 1024> g_monitorBuffer;

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
            GFL_LOG_ERROR("[CuptiBackend] RangeProfiler engine requires "
                          "GPUFL_HAS_PERFWORKS; falling back to None");
#endif
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

    // Delegate engine teardown first
    if (engine_) {
        engine_->stop();
        engine_->shutdown();
        engine_.reset();
    }

    LogCuptiErrorIfFailed("Perfworks", "cuptiActivityFlushAll",
                          cuptiActivityFlushAll(1));

    {
        std::lock_guard<std::mutex> lk(handler_mu_);
        std::set<CUpti_CallbackDomain> domains;
        for (const auto& h : handlers_)
            for (auto d : h->requiredDomains()) domains.insert(d);
        for (auto d : domains) cuptiEnableDomain(0, subscriber_, d);
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
            chip_name_           = getChipName(device_id_);
            cached_device_name_  = GetCurrentDeviceName();

            EngineContext ectx{ctx_, device_id_, chip_name_,
                               &cubin_mu_, &cubin_by_crc_};
            engine_->initialize(opts_, ectx);
            engine_->start();
        } else {
            GFL_LOG_ERROR("[CuptiBackend] Failed to get CUDA context; "
                          "engine will not start.");
        }
    }

    active_.store(true);
    GFL_LOG_DEBUG("Backend started.");
}

void CuptiBackend::stop() {
    if (!initialized_) return;
    active_.store(false);

    LogCuptiErrorIfFailed("Perfworks", "cuptiActivityFlushAll",
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
    std::lock_guard<std::mutex> lk(handler_mu_);
    handlers_.push_back(handler);
}

// ---- Static callbacks ------------------------------------------------------

void CUPTIAPI CuptiBackend::BufferRequested(uint8_t** buffer, size_t* size,
                                            size_t* maxNumRecords) {
    *size          = 64 * 1024;
    *buffer        = static_cast<uint8_t*>(malloc(*size));
    *maxNumRecords = 0;
}

void CUPTIAPI CuptiBackend::BufferCompleted(CUcontext context,
                                            uint32_t streamId, uint8_t* buffer,
                                            size_t size,
                                            const size_t validSize) {
    auto* backend = g_activeBackend.load(std::memory_order_acquire);
    if (!backend) {
        ::gpufl::DebugLogger::error("[CUPTI] ",
                                    "BufferCompleted: No active backend!");
        if (buffer) free(buffer);
        return;
    }

    static int64_t  baseCpuNs   = detail::GetTimestampNs();
    static uint64_t baseCuptiTs = 0;
    if (baseCuptiTs == 0) cuptiGetTimestamp(&baseCuptiTs);

    std::vector<std::shared_ptr<ICuptiHandler>> handlers;
    {
        std::lock_guard<std::mutex> lk(backend->handler_mu_);
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
                    out.type         = TraceType::PC_SAMPLE;
                    out.corr_id      = pc->correlationId;
                    std::snprintf(out.sample_kind, sizeof(out.sample_kind),
                                  "%s", "pc_sampling");
                    out.samples_count = pc->samples;
                    out.stall_reason = pc->stallReason;
                    out.device_id    =
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

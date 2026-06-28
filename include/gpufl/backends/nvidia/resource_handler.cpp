#include "gpufl/backends/nvidia/resource_handler.hpp"

#include <cupti_pcsampling.h>
#include <zlib.h>

#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/monitor.hpp"

namespace gpufl {

ResourceHandler::ResourceHandler(CuptiBackend *backend) : backend_(backend) {
    worker_ = std::thread(&ResourceHandler::workerLoop, this);
}

ResourceHandler::~ResourceHandler() {
    {
        std::lock_guard lk(pending_mu_);
        stop_worker_ = true;
    }
    pending_cv_.notify_all();
    if (worker_.joinable()) worker_.join();
}

bool ResourceHandler::shouldHandle(CUpti_CallbackDomain domain,
                                   CUpti_CallbackId cbid) const {
    return domain == CUPTI_CB_DOMAIN_RESOURCE;
}

std::vector<CUpti_CallbackDomain> ResourceHandler::requiredDomains() const {
    return {CUPTI_CB_DOMAIN_RESOURCE};
}

std::vector<std::pair<CUpti_CallbackDomain, CUpti_CallbackId>>
ResourceHandler::requiredCallbacks() const {
    return {
        {CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_MODULE_LOADED},
        {CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_MODULE_PROFILED},
        // Completes a deferred engine start: Windows injection initializes
        // before the target has a CUDA context, so context-bound engines
        // (PC sampling, SASS, …) wait for the first context here.
        {CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_CONTEXT_CREATED},
        // Drain activity buffers while the context is still alive, just before
        // the driver tears it down - for contexts destroyed mid-process (an
        // explicit cudaDeviceReset/cuCtxDestroy, or multi-context apps), where
        // the at-exit flush is skipped to avoid a driver deadlock (see
        // CuptiBackend::FlushOnContextDestroy). This does NOT fire on Windows
        // process exit (cudart leaves context teardown to driver DLL-detach);
        // those records are recovered by Monitor::Shutdown's post-join drain.
        // NOTE: cuptiEnableDomain(RESOURCE) already enables every RESOURCE
        // callback, so this entry is belt-and-suspenders / documentation.
        {CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING},
    };
}

void ResourceHandler::handle(CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
                             const void *cbdata) {
    auto resourceData = static_cast<const CUpti_ResourceData *>(cbdata);

    // The target created its first CUDA context - complete a deferred
    // engine start (no-op when the engine started normally). The heavy
    // work runs on the backend's own thread, NOT here: this callback
    // fires from inside the driver's context-creation path.
    if (cbid == CUPTI_CBID_RESOURCE_CONTEXT_CREATED) {
        if (backend_) backend_->RequestDeferredEngineStart(resourceData->context);
        return;
    }

    // Context is about to be destroyed - flush activity NOW, synchronously,
    // while it's still valid. Returning from this callback lets the driver
    // proceed with teardown, so the flush must complete before we return.
    if (cbid == CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING) {
        if (backend_) backend_->FlushOnContextDestroy();
        return;
    }

    if (cbid == CUPTI_CBID_RESOURCE_MODULE_LOADED ||
        cbid == CUPTI_CBID_RESOURCE_MODULE_PROFILED) {
        // Only capture cubins when the active engine actually consumes
        // them. For profiling_engine=None (monitoring only) and
        // RangeProfiler (scope-level HW counters) there is no per-PC data
        // to overlay disassembly on, so capturing + disassembling +
        // uploading cubins would be pure waste. Gating here keeps
        // "monitoring only" truly monitoring-only and stops the
        // cubin_disassembly bloat seen with PyTorch under None.
        if (!backend_->NeedsCubinCapture()) return;

        const auto *modData = static_cast<CUpti_ModuleResourceData *>(
            resourceData->resourceDescriptor);
        if (modData && modData->pCubin && modData->cubinSize > 0) {
            const void *cubinPtr = modData->pCubin;
            const size_t cubinSize = modData->cubinSize;

            // CUPTI_CBID_RESOURCE_MODULE_PROFILED fires on every kernel
            // launch when PC sampling is active, including for
            // SASS-patched cubin variants.  Use the raw pointer as an
            // O(1) sentinel - cubin memory is stable for the process
            // lifetime.  We must NOT skip MODULE_PROFILED entirely:
            // its cubin may have a different CRC than the original
            // MODULE_LOADED cubin, and CUPTI's PC sampling data
            // references that CRC for source correlation.
            {
                std::lock_guard lk(backend_->cubin_mu_);
                if (backend_->seen_cubin_ptrs_.count(cubinPtr)) return;
                backend_->seen_cubin_ptrs_.insert(cubinPtr);
            }

            // Copy the cubin bytes here (safe - no CUPTI calls) and hand off
            // to the worker thread. cuptiGetCubinCrc() must NOT run from this
            // callback: SASS holds CUPTI-internal locks during cubin patching
            // (modules loaded after cuptiSassMetricsEnable() are patched lazily
            // even with enableLazyPatching=0), so calling it here deadlocks.
            // The worker computes the CRC off the callback.
            GFL_LOG_DEBUG("Queuing Cubin for CRC at ", cubinPtr,
                          " Size: ", cubinSize);
            std::vector bytes(
                static_cast<const uint8_t *>(cubinPtr),
                static_cast<const uint8_t *>(cubinPtr) + cubinSize);
            {
                std::lock_guard lk(pending_mu_);
                pending_.push(std::move(bytes));
            }
            pending_cv_.notify_one();
        }
    }
}

void ResourceHandler::workerLoop() {
    while (true) {
        std::vector<uint8_t> data;
        {
            std::unique_lock lk(pending_mu_);
            pending_cv_.wait(lk,
                             [this] { return !pending_.empty() || stop_worker_; });
            if (stop_worker_ && pending_.empty()) return;
            data = std::move(pending_.front());
            pending_.pop();
        }
        processCubin(std::move(data));
    }
}

void ResourceHandler::processCubin(std::vector<uint8_t> data) {
    // Windows-injection PC sampling can't call cuptiGetCubinCrc at all: it takes
    // CUPTI's internal global lock, which disengages the armed HW sampler (zero
    // samples). It uses zlib crc32 instead - harmless because PC samples join
    // disassembly by FUNCTION NAME, not by cubin CRC (the CRC is only the
    // cubin_disassembly message's group key). Every other path keeps
    // cuptiGetCubinCrc, the exact CRC CUPTI puts in its records.
    const bool inject = backend_->IsWindowsInjectedPcSampling();
    GFL_LOG_DEBUG("Computing CRC for cubin copy, size=", data.size());
    uint64_t cubinCrc = 0;
    if (inject) {
        cubinCrc = crc32(0, data.data(), static_cast<uInt>(data.size()));
    } else {
        CUpti_GetCubinCrcParams crcParams = {CUpti_GetCubinCrcParamsSize};
        crcParams.cubin = data.data();
        crcParams.cubinSize = data.size();
        if (cuptiGetCubinCrc(&crcParams) == CUPTI_SUCCESS) {
            cubinCrc = crcParams.cubinCrc;
        } else {
            GFL_LOG_DEBUG("cuptiGetCubinCrc failed, falling back to zlib crc32");
            cubinCrc = crc32(0, data.data(), static_cast<uInt>(data.size()));
        }
    }

    bool isNew = false;
    const uint8_t *enqueuePtr = nullptr;
    size_t enqueueSize = 0;
    {
        std::lock_guard lk(backend_->cubin_mu_);
        if (backend_->cubin_by_crc_.find(cubinCrc) ==
            backend_->cubin_by_crc_.end()) {
            auto &[map_data, map_crc] = backend_->cubin_by_crc_[cubinCrc];

            map_crc = cubinCrc;
            map_data = std::move(data);

            enqueuePtr = map_data.data();
            enqueueSize = map_data.size();
            isNew = true;
        }
    }
    if (isNew) {
        Monitor::EnqueueCubinForDisassembly(cubinCrc, enqueuePtr, enqueueSize);
        GFL_LOG_DEBUG("Finished EnqueueCubinForDisassembly!");
        if (inject) {
            // Disassemble + emit NOW, on this worker thread during the run -
            // NOT at shutdown. nvdisasm during the Windows-injection process-
            // exit teardown intermittently hangs (proven); here the GPU is
            // live and stable. nvdisasm is a subprocess (no CUPTI), so the
            // armed sampler is unaffected.
            Monitor::FlushDisassemblyNow();
        }
    }
}

}  // namespace gpufl

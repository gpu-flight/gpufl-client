#include "gpufl/backends/nvidia/resource_handler.hpp"

#include <cupti_pcsampling.h>

#include "gpufl/backends/nvidia/cupti_utils.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/monitor.hpp"

namespace gpufl {

ResourceHandler::ResourceHandler(CuptiBackend *backend) : backend_(backend) {
    worker_ = std::thread(&ResourceHandler::workerLoop, this);
}

ResourceHandler::~ResourceHandler() {
    {
        std::lock_guard<std::mutex> lk(pending_mu_);
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
    };
}

void ResourceHandler::handle(CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
                             const void *cbdata) {
    auto resourceData = static_cast<const CUpti_ResourceData *>(cbdata);

    if (cbid == CUPTI_CBID_RESOURCE_MODULE_LOADED ||
        cbid == CUPTI_CBID_RESOURCE_MODULE_PROFILED) {
        auto *modData = static_cast<CUpti_ModuleResourceData *>(
            resourceData->resourceDescriptor);
        if (modData && modData->pCubin && modData->cubinSize > 0) {
            const void *cubinPtr = modData->pCubin;
            const size_t cubinSize = modData->cubinSize;

            {
                std::lock_guard<std::mutex> lk(backend_->cubin_mu_);
                if (backend_->seen_cubin_ptrs_.count(cubinPtr)) return;
                // CUPTI_CBID_RESOURCE_MODULE_PROFILED fires on every kernel
                // launch when PC sampling is active, including for
                // SASS-patched cubin variants that have a different pointer
                // than the original. Mark the pointer seen and bail out —
                // the original cubin was already processed by MODULE_LOADED.
                if (cbid == CUPTI_CBID_RESOURCE_MODULE_PROFILED) {
                    backend_->seen_cubin_ptrs_.insert(cubinPtr);
                    return;
                }
                backend_->seen_cubin_ptrs_.insert(cubinPtr);
            }

            // Copy the cubin bytes here (safe — no CUPTI calls).
            // cuptiGetCubinCrc() must NOT be called from this callback:
            // SASS holds CUPTI-internal locks during cubin patching (even
            // with enableLazyPatching=0, modules loaded after
            // cuptiSassMetricsEnable() are still patched lazily). Calling
            // cuptiGetCubinCrc() here deadlocks. Defer to the worker thread.
            GFL_LOG_DEBUG("Queuing Cubin for CRC at ", cubinPtr,
                          " Size: ", cubinSize);
            std::vector<uint8_t> bytes(
                static_cast<const uint8_t *>(cubinPtr),
                static_cast<const uint8_t *>(cubinPtr) + cubinSize);
            {
                std::lock_guard<std::mutex> lk(pending_mu_);
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
            std::unique_lock<std::mutex> lk(pending_mu_);
            pending_cv_.wait(lk,
                             [this] { return !pending_.empty() || stop_worker_; });
            if (stop_worker_ && pending_.empty()) return;
            data = std::move(pending_.front());
            pending_.pop();
        }

        // Now outside the CUPTI callback — SASS locks are released.
        // Safe to call cuptiGetCubinCrc() on the copied bytes.
        CUpti_GetCubinCrcParams params = {CUpti_GetCubinCrcParamsSize};
        params.cubinSize = data.size();
        params.cubin = data.data();
        GFL_LOG_DEBUG("Computing CRC for cubin copy, size=", data.size());
        if (cuptiGetCubinCrc(&params) != CUPTI_SUCCESS) {
            GFL_LOG_ERROR("[ResourceHandler] Failed to compute CRC for cubin");
            continue;
        }

        bool isNew = false;
        const uint8_t *enqueuePtr = nullptr;
        size_t enqueueSize = 0;
        {
            std::lock_guard<std::mutex> lk(backend_->cubin_mu_);
            if (backend_->cubin_by_crc_.find(params.cubinCrc) ==
                backend_->cubin_by_crc_.end()) {
                auto &info = backend_->cubin_by_crc_[params.cubinCrc];
                info.crc = params.cubinCrc;
                info.data = std::move(data);
                enqueuePtr = info.data.data();
                enqueueSize = info.data.size();
                isNew = true;
            }
        }
        if (isNew) {
            Monitor::EnqueueCubinForDisassembly(params.cubinCrc, enqueuePtr,
                                                enqueueSize);
        }
    }
}

}  // namespace gpufl

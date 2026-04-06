#include "gpufl/backends/nvidia/resource_handler.hpp"

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
            std::unique_lock lk(pending_mu_);
            pending_cv_.wait(lk,
                             [this] { return !pending_.empty() || stop_worker_; });
            if (stop_worker_ && pending_.empty()) return;
            data = std::move(pending_.front());
            pending_.pop();
        }

        // Compute the cubin CRC using zlib crc32 — this matches the algorithm
        // CUPTI uses internally for the cubinCrc field in activity records
        // (CUpti_ActivityKernel11, CUpti_ActivityPCSampling3, etc.).
        //
        // We previously called cuptiGetCubinCrc() here, but that function
        // acquires CUPTI's internal global lock, which is also held by
        // cuptiSassMetricsEnable() while it patches loaded modules.  Even
        // from this worker thread the call can therefore deadlock when
        // cuptiSassMetricsEnable() is still running on another thread.
        // Using zlib directly avoids all CUPTI locks.
        GFL_LOG_DEBUG("Computing CRC for cubin copy, size=", data.size());
        const uint64_t cubinCrc = crc32(0, data.data(), static_cast<uInt>(data.size()));

        bool isNew = false;
        const uint8_t *enqueuePtr = nullptr;
        size_t enqueueSize = 0;
        {
            GFL_LOG_DEBUG("Lock here");
            std::lock_guard lk(backend_->cubin_mu_);
            GFL_LOG_DEBUG("Lock acquired");
            if (backend_->cubin_by_crc_.find(cubinCrc) ==
                backend_->cubin_by_crc_.end()) {
                auto &[map_data, map_crc] = backend_->cubin_by_crc_[cubinCrc];

                map_crc = cubinCrc;
                map_data = std::move(data); // Now successfully moves the outer bytes into the map

                enqueuePtr = map_data.data();
                enqueueSize = map_data.size(); // Will now be > 0
                isNew = true;
            }
        }
        if (isNew) {
            Monitor::EnqueueCubinForDisassembly(cubinCrc, enqueuePtr,
                                                enqueueSize);
            GFL_LOG_DEBUG("Finished EnqueueCubinForDisassembly!");
        }
    }
}

}  // namespace gpufl

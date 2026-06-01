#pragma once

#include "gpufl/backends/nvidia/cupti_backend.hpp"
#include "gpufl/backends/nvidia/cupti_common.hpp"

namespace gpufl {

class CudaCleanupHandler : public ICuptiHandler {
   public:
    explicit CudaCleanupHandler(CuptiBackend* backend) : backend_(backend) {}

    const char* getName() const override { return "CudaCleanupHandler"; }
    bool shouldHandle(CUpti_CallbackDomain domain,
                      CUpti_CallbackId cbid) const override;
    void handle(CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
                const void* cbdata) override;
    std::vector<std::pair<CUpti_CallbackDomain, CUpti_CallbackId>>
    requiredCallbacks() const override;

   private:
    static const char* CleanupReason(CUpti_CallbackDomain domain,
                                     CUpti_CallbackId cbid);

    CuptiBackend* backend_ = nullptr;
};

}  // namespace gpufl

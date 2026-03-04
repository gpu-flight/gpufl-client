#pragma once

#include "gpufl/backends/nvidia/cupti_backend.hpp"
#include "gpufl/backends/nvidia/cupti_common.hpp"

namespace gpufl {

class KernelLaunchHandler : public ICuptiHandler {
   public:
    explicit KernelLaunchHandler(CuptiBackend* backend);

    const char* getName() const override { return "KernelLaunchHandler"; }
    bool shouldHandle(CUpti_CallbackDomain domain,
                      CUpti_CallbackId cbid) const override;
    void handle(CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
                const void* cbdata) override;
    std::vector<CUpti_CallbackDomain> requiredDomains() const override;
    std::vector<CUpti_ActivityKind> requiredActivityKinds() const override;
    bool handleActivityRecord(const CUpti_Activity* record, int64_t baseCpuNs,
                              uint64_t baseCuptiTs) override;

   private:
    CuptiBackend* backend_;
};

}  // namespace gpufl

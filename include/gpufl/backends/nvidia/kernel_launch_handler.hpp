#pragma once

#include <mutex>
#include <string>
#include <unordered_map>

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
    std::vector<std::pair<CUpti_CallbackDomain, CUpti_CallbackId>>
    requiredCallbacks() const override;
    std::vector<CUpti_ActivityKind> requiredActivityKinds() const override;
    bool handleActivityRecord(const CUpti_Activity* record, int64_t baseCpuNs,
                              uint64_t baseCuptiTs) override;

   private:
    CuptiBackend* backend_;
    // Cache for demangled kernel names — avoids re-demangling on every launch
    std::mutex demangle_mu_;
    std::unordered_map<std::string, std::string> demangle_cache_;
    const std::string& cachedDemangle(const char* mangled);
};

}  // namespace gpufl

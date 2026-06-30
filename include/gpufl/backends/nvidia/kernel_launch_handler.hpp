#pragma once

#include <mutex>
#include <set>
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
    // shouldHandle()'s CBID filter, built once from requiredCallbacks() here in
    // the constructor (NOT a function-local static - see shouldHandle()).
    const std::set<std::pair<CUpti_CallbackDomain, CUpti_CallbackId>> kHandled_;
    // NOTE: kernel-name demangling moved OFF the callback path (Step 4a). The
    // collector thread demangles raw names (DemangleKernelNameCached in
    // monitor.cpp); there is no per-callback demangle mutex anymore.
};

}  // namespace gpufl

#pragma once

#include "gpufl/backends/nvidia/cupti_backend.hpp"
#include "gpufl/backends/nvidia/cupti_common.hpp"

namespace gpufl {

// Keeps the PTX of modules loaded at run time. cuobjdump finds PTX only in an
// executable's fatbin; code that hands PTX text to the driver itself (cuda-oxide
// Rust kernels, NVRTC, a JIT linker) is seen only here, at the load call.
class ModuleLoadPtxHandler : public ICuptiHandler {
   public:
    explicit ModuleLoadPtxHandler(CuptiBackend* backend) : backend_(backend) {}

    const char* getName() const override { return "ModuleLoadPtxHandler"; }
    bool shouldHandle(CUpti_CallbackDomain domain,
                      CUpti_CallbackId cbid) const override;
    void handle(CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
                const void* cbdata) override;
    std::vector<std::pair<CUpti_CallbackDomain, CUpti_CallbackId>>
    requiredCallbacks() const override;

   private:
    CuptiBackend* backend_ = nullptr;
};

}  // namespace gpufl

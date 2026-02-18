#pragma once

#include "gpufl/backends/nvidia/cupti_common.hpp"
#include "gpufl/backends/nvidia/cupti_backend.hpp"

namespace gpufl {

    class KernelLaunchHandler : public ICuptiHandler {
    public:
        explicit KernelLaunchHandler(CuptiBackend* backend);
        
        const char* getName() const override { return "KernelLaunchHandler"; }
        bool shouldHandle(CUpti_CallbackDomain domain, CUpti_CallbackId cbid) const override;
        void handle(CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void* cbdata) override;
    private:
        CuptiBackend* backend_;
    };

} // namespace gpufl

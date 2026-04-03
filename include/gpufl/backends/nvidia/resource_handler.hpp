#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "gpufl/backends/nvidia/cupti_backend.hpp"
#include "gpufl/backends/nvidia/cupti_common.hpp"

namespace gpufl {

class ResourceHandler : public ICuptiHandler {
   public:
    explicit ResourceHandler(CuptiBackend* backend);
    ~ResourceHandler();

    const char* getName() const override { return "ResourceHandler"; }
    bool shouldHandle(CUpti_CallbackDomain domain,
                      CUpti_CallbackId cbid) const override;
    void handle(CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
                const void* cbdata) override;
    std::vector<CUpti_CallbackDomain> requiredDomains() const override;
    std::vector<std::pair<CUpti_CallbackDomain, CUpti_CallbackId>>
    requiredCallbacks() const override;

   private:
    void workerLoop();

    CuptiBackend* backend_;

    std::queue<std::vector<uint8_t>> pending_;
    std::mutex pending_mu_;
    std::condition_variable pending_cv_;
    bool stop_worker_{false};
    std::thread worker_;
};

}  // namespace gpufl

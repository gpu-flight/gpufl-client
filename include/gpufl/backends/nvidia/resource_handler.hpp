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
    // Compute the cubin CRC (cuptiGetCubinCrc normally; zlib crc32 under
    // Windows-injection PC sampling, where the CUPTI call would disengage the
    // sampler), store it in the backend's cubin map, and enqueue it for
    // disassembly. Under injection it also disassembles immediately so nvdisasm
    // runs during the live run, not during the fragile process-exit teardown.
    void processCubin(std::vector<uint8_t> data);

    CuptiBackend* backend_;

    std::queue<std::vector<uint8_t>> pending_;
    std::mutex pending_mu_;
    std::condition_variable pending_cv_;
    bool stop_worker_{false};
    std::thread worker_;
};

}  // namespace gpufl

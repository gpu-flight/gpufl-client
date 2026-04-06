#pragma once

#include <memory>
#include <string>

#include <cupti.h>

#include "gpufl/backends/nvidia/cupti_backend.hpp"
#include "gpufl/core/monitor_adapter.hpp"

namespace gpufl::nvidia {

class NvidiaMonitorAdapter final : public IMonitorAdapter {
   public:
    NvidiaMonitorAdapter();
    ~NvidiaMonitorAdapter() override;

    void initialize(const MonitorOptions& opts) override;
    void shutdown() override;
    void start() override;
    void stop() override;
    const char* platformName() const override { return "cuda"; }
    std::string memcpyKindToString(uint32_t kind) const override;
    std::string memoryKindToString(uint32_t kind) const override;

    IMonitorBackend* backend() override;

   private:
    std::unique_ptr<CuptiBackend> backend_;
};

}  // namespace gpufl::nvidia

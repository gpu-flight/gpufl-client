#pragma once

#include <memory>
#include <string>

#include "gpufl/backends/amd/rocprofiler_backend.hpp"
#include "gpufl/core/monitor_adapter.hpp"

namespace gpufl::amd {

class AmdMonitorAdapter final : public IMonitorAdapter {
   public:
    AmdMonitorAdapter();
    ~AmdMonitorAdapter() override;

    void initialize(const MonitorOptions& opts) override;
    void shutdown() override;
    void start() override;
    void stop() override;
    const char* platformName() const override { return "amd"; }
    std::string memcpyKindToString(uint32_t kind) const override;
    std::string memoryKindToString(uint32_t kind) const override;

    IMonitorBackend* backend() override;

   private:
    std::unique_ptr<RocprofilerBackend> backend_;
};

}  // namespace gpufl::amd

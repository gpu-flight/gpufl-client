#pragma once

#include <memory>
#include <optional>
#include <string>

#include "gpufl/core/events.hpp"
#include "gpufl/core/monitor.hpp"
#include "gpufl/core/monitor_backend.hpp"

namespace gpufl {

class IMonitorAdapter {
   public:
    virtual ~IMonitorAdapter() = default;

    virtual void initialize(const MonitorOptions& opts) = 0;
    virtual void shutdown() = 0;
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual const char* platformName() const = 0;
    virtual std::string memcpyKindToString(uint32_t kind) const = 0;
    virtual std::string memoryKindToString(uint32_t kind) const = 0;

    virtual IMonitorBackend* backend() = 0;
};

std::unique_ptr<IMonitorAdapter> CreateMonitorAdapter(const MonitorOptions& opts);

}  // namespace gpufl

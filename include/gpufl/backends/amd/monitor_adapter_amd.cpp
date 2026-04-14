#include "gpufl/backends/amd/monitor_adapter_amd.hpp"

namespace gpufl::amd {

AmdMonitorAdapter::AmdMonitorAdapter() = default;
AmdMonitorAdapter::~AmdMonitorAdapter() = default;

void AmdMonitorAdapter::initialize(const MonitorOptions& opts) {
    if (!backend_) backend_ = std::make_unique<RocprofilerBackend>();
    backend_->initialize(opts);
}

void AmdMonitorAdapter::shutdown() {
    if (!backend_) return;
    backend_->shutdown();
    backend_.reset();
}

void AmdMonitorAdapter::start() {
    if (backend_) backend_->start();
}

void AmdMonitorAdapter::stop() {
    if (backend_) backend_->stop();
}

std::string AmdMonitorAdapter::memcpyKindToString(const uint32_t kind) const {
    switch (kind) {
        case 1:
            return "HtoD";
        case 2:
            return "DtoH";
        case 3:
            return "DtoD";
        case 4:
            return "HtoH";
        default:
            return "Unknown";
    }
}

std::string AmdMonitorAdapter::memoryKindToString(const uint32_t kind) const {
    // Memory kind encodes the agent type of src/dst:
    //   0 = unknown, 1 = host (pageable), 2 = device (GPU VRAM)
    switch (kind) {
        case 1:  return "Pageable";
        case 2:  return "Device";
        default: return "Unknown";
    }
}

IMonitorBackend* AmdMonitorAdapter::backend() {
    return backend_.get();
}

}  // namespace gpufl::amd

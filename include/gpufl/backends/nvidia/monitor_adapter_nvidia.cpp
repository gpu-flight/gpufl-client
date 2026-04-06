#include "gpufl/backends/nvidia/monitor_adapter_nvidia.hpp"

namespace gpufl::nvidia {

NvidiaMonitorAdapter::NvidiaMonitorAdapter() = default;
NvidiaMonitorAdapter::~NvidiaMonitorAdapter() = default;

void NvidiaMonitorAdapter::initialize(const MonitorOptions& opts) {
    if (!backend_) backend_ = std::make_unique<CuptiBackend>();
    backend_->initialize(opts);
}

void NvidiaMonitorAdapter::shutdown() {
    if (!backend_) return;
    backend_->shutdown();
    backend_.reset();
}

void NvidiaMonitorAdapter::start() {
    if (backend_) backend_->start();
}

void NvidiaMonitorAdapter::stop() {
    if (backend_) backend_->stop();
}

std::string NvidiaMonitorAdapter::memcpyKindToString(const uint32_t kind) const {
    switch (kind) {
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
            return "HtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
            return "DtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
            return "HtoA";
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
            return "AtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
            return "AtoA";
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
            return "AtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
            return "DtoA";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
            return "DtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
            return "HtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
            return "PtoP";
        default:
            return "Unknown";
    }
}

std::string NvidiaMonitorAdapter::memoryKindToString(const uint32_t kind) const {
    switch (kind) {
        case CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN:
            return "Unknown";
        case CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE:
            return "Pageable";
        case CUPTI_ACTIVITY_MEMORY_KIND_PINNED:
            return "Pinned";
        case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE:
            return "Device";
        case CUPTI_ACTIVITY_MEMORY_KIND_ARRAY:
            return "Array";
        case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED:
            return "Managed";
        case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC:
            return "DeviceStatic";
        case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC:
            return "ManagedStatic";
        default:
            return "Unknown";
    }
}

IMonitorBackend* NvidiaMonitorAdapter::backend() {
    return backend_.get();
}

}  // namespace gpufl::nvidia

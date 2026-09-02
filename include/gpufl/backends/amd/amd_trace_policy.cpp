#include "gpufl/backends/amd/amd_trace_policy.hpp"

namespace gpufl::amd {

std::optional<uint32_t> ResolveAmdKernelDeviceId(
    const AmdTraceEndpoint& agent) {
    if (agent.kind != AmdTraceAgentKind::Gpu) return std::nullopt;
    return agent.device_id;
}

std::optional<uint32_t> ResolveAmdMemoryCopyDeviceId(
    const AmdTraceEndpoint& source,
    const AmdTraceEndpoint& destination) {
    if (destination.kind == AmdTraceAgentKind::Gpu &&
        destination.device_id.has_value()) {
        return destination.device_id;
    }
    if (source.kind == AmdTraceAgentKind::Gpu && source.device_id.has_value()) {
        return source.device_id;
    }
    return std::nullopt;
}

std::optional<uint32_t> ResolveAmdMemoryAllocationDeviceId(
    const AmdTraceEndpoint& agent) {
    if (agent.kind != AmdTraceAgentKind::Gpu) return std::nullopt;
    return agent.device_id;
}

std::optional<uint8_t> NormalizeAmdMemoryAllocationOperation(
    const uint32_t operation) {
    // rocprofiler_memory_allocation_operation_t:
    //   1=ALLOCATE, 2=VMEM_ALLOCATE, 3=FREE, 4=VMEM_FREE.
    if (operation == 1 || operation == 2) return uint8_t{1};
    if (operation == 3 || operation == 4) return uint8_t{2};
    return std::nullopt;
}

uint8_t ResolveAmdMemoryAllocationKind(const AmdTraceEndpoint& agent) {
    return agent.kind == AmdTraceAgentKind::Gpu ? uint8_t{3} : uint8_t{0};
}

}  // namespace gpufl::amd

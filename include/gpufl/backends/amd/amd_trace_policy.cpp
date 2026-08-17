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

}  // namespace gpufl::amd

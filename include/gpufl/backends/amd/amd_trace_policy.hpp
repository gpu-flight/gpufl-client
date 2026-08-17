#pragma once

#include <cstdint>
#include <optional>

namespace gpufl::amd {

enum class AmdTraceAgentKind {
    Unknown,
    Cpu,
    Gpu,
};

struct AmdTraceEndpoint {
    AmdTraceAgentKind kind = AmdTraceAgentKind::Unknown;
    std::optional<uint32_t> device_id;
};

std::optional<uint32_t> ResolveAmdKernelDeviceId(
    const AmdTraceEndpoint& agent);

// Memory-copy activity belongs to the destination GPU when one exists (HtoD
// and DtoD), otherwise to the source GPU (DtoH). Host-only copies have no GPU
// device attribution.
std::optional<uint32_t> ResolveAmdMemoryCopyDeviceId(
    const AmdTraceEndpoint& source,
    const AmdTraceEndpoint& destination);

}  // namespace gpufl::amd

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

// Memory-allocation activity belongs to the agent that owns the allocation.
// CPU allocations intentionally have no GPU device id; callers may still emit
// them as host allocations. An unmapped GPU remains unattributed.
std::optional<uint32_t> ResolveAmdMemoryAllocationDeviceId(
    const AmdTraceEndpoint& agent);

// Normalize ROCprofiler's ALLOCATE/VMEM_ALLOCATE/FREE/VMEM_FREE operation
// values to GPUFlight's portable 1=ALLOC, 2=FREE wire contract. NONE and
// unknown future values are not emitted.
std::optional<uint8_t> NormalizeAmdMemoryAllocationOperation(
    uint32_t operation);

// GPUFlight memory-kind wire values follow CUPTI for cross-vendor consumers.
// ROCprofiler only identifies the owning agent, so GPU allocations can be
// classified as DEVICE (3) while CPU allocations remain UNKNOWN (0).
uint8_t ResolveAmdMemoryAllocationKind(const AmdTraceEndpoint& agent);

}  // namespace gpufl::amd

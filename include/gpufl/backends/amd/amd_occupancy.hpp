#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

namespace gpufl::amd {

// The occupancy_estimate capability's mode: each kernel row's estimate uses
// the register file, wave slots and LDS of its own GPU target.
inline constexpr const char* kAmdOccupancyModelWireName = "amd_target_model_v1";

enum class AmdOccupancyModel {
    // No documented model for the target: the kernel row carries no estimate
    // rather than another architecture's register file.
    None,
    // CDNA 3 (gfx940, gfx941, gfx942: Instinct MI300 series). Wave64, four
    // SIMDs per CU, at most 8 waves per SIMD, and one 512-entry-per-lane
    // register file per SIMD shared by the architectural and accumulation
    // VGPRs, allocated in granules of 8. 64 KiB of LDS per CU, allocated in
    // 512-byte granules.
    Cdna3,
    // RDNA (gfx10, gfx11, gfx12). The estimate rc6 shipped, unchanged: 1,536
    // VGPRs per SIMD in granules of 8, architectural VGPRs only, and the
    // agent's own wave slots, SIMDs and LDS.
    Rdna,
};

// The agent name is the gfx target ("gfx942"); a target ID with feature
// suffixes ("gfx942:sramecc+:xnack-") resolves the same way.
AmdOccupancyModel ResolveAmdOccupancyModel(std::string_view target);

// What the agent reports about the device. Only the Rdna model reads it;
// Cdna3 uses AMD's documented limits for the target.
struct AmdOccupancyDevice {
    uint32_t wave_front_size = 0;
    uint32_t simd_per_cu = 0;
    uint32_t max_waves_per_cu = 0;
    uint32_t lds_size_bytes = 0;  // per CU
};

// One dispatch. The VGPR counts are rocprofiler-sdk's kernel symbol fields,
// which it derives from the kernel descriptor: on gfx94x arch_vgpr_count is
// the accumulation offset and accum_vgpr_count the rest of the allocation,
// so their sum is the wave's whole share of the unified register file.
struct AmdOccupancyKernel {
    uint32_t threads_per_block = 0;
    uint32_t lds_bytes = 0;  // static plus dynamic, per block
    uint32_t arch_vgpr_count = 0;
    uint32_t accum_vgpr_count = 0;
};

// Fractions are of the CU's wave slots, as for NVIDIA's theoretical
// occupancy. block_occupancy stays 1: neither model limits workgroups per CU.
struct AmdOccupancyEstimate {
    float occupancy = 0.0f;
    float warp_occupancy = 0.0f;
    float smem_occupancy = 0.0f;
    float reg_occupancy = 0.0f;
    float block_occupancy = 1.0f;
    int max_active_blocks = 0;
    const char* limiting_resource = "waves";
};

// nullopt when the model is None or the dispatch can't be estimated (no
// threads, or an Rdna device without wave slots).
std::optional<AmdOccupancyEstimate> EstimateAmdOccupancy(
    AmdOccupancyModel model, const AmdOccupancyDevice& device,
    const AmdOccupancyKernel& kernel);

}  // namespace gpufl::amd

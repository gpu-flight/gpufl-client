#include "gpufl/backends/amd/amd_occupancy.hpp"

#include <algorithm>

namespace gpufl::amd {
namespace {

// CDNA 3 limits, from AMD's documentation: ROCm's GPU specifications (wave64,
// 64 KiB LDS and a 512 KiB VGPR file per CU on MI300X and MI325X), the ROCm
// blog's occupancy math for CDNA (four SIMDs per CU, 8 waves per SIMD, a
// 512-entry-per-lane file shared by VGPRs and AGPRs, granules of 8), and
// LLVM's AMDGPU kernel descriptor (LDS_SIZE in 512-byte granules on GFX9).
constexpr int64_t kCdna3WaveSize = 64;
constexpr int64_t kCdna3SimdsPerCu = 4;
constexpr int64_t kCdna3WavesPerSimd = 8;
constexpr int64_t kCdna3RegistersPerSimd = 512;
constexpr int64_t kCdna3RegisterGranule = 8;
constexpr int64_t kCdna3LdsBytesPerCu = 64 * 1024;
constexpr int64_t kCdna3LdsGranule = 512;

// rc6's RDNA register file: 1,536 VGPRs per SIMD in granules of 8.
constexpr int64_t kRdnaVgprsPerSimd = 1536;
constexpr int64_t kRdnaVgprGranule = 8;

int64_t AlignUp(const int64_t value, const int64_t granule) {
    return (value + granule - 1) / granule * granule;
}

AmdOccupancyEstimate FromBlockLimits(const int64_t waves_per_block,
                                     const int64_t max_waves_per_cu,
                                     const int64_t wave_blocks,
                                     const int64_t lds_blocks,
                                     const int64_t reg_blocks) {
    const auto to_occupancy = [&](const int64_t blocks) {
        return std::min(1.0f, static_cast<float>(blocks * waves_per_block) /
                                  static_cast<float>(max_waves_per_cu));
    };
    const int64_t active_blocks =
        std::max<int64_t>(std::min({wave_blocks, lds_blocks, reg_blocks}), 0);

    AmdOccupancyEstimate estimate;
    estimate.occupancy = to_occupancy(active_blocks);
    estimate.warp_occupancy = to_occupancy(wave_blocks);
    estimate.smem_occupancy = to_occupancy(lds_blocks);
    estimate.reg_occupancy = to_occupancy(reg_blocks);
    estimate.max_active_blocks = static_cast<int>(active_blocks);

    // Wave slots win a tie, then registers, then LDS.
    float lowest = estimate.warp_occupancy;
    if (estimate.reg_occupancy < lowest) {
        lowest = estimate.reg_occupancy;
        estimate.limiting_resource = "registers";
    }
    if (estimate.smem_occupancy < lowest) {
        estimate.limiting_resource = "shared_mem";
    }
    return estimate;
}

std::optional<AmdOccupancyEstimate> EstimateCdna3(
    const AmdOccupancyKernel& kernel) {
    if (kernel.threads_per_block == 0) return std::nullopt;
    const int64_t waves_per_block =
        (kernel.threads_per_block + kCdna3WaveSize - 1) / kCdna3WaveSize;
    const int64_t max_waves_per_cu = kCdna3WavesPerSimd * kCdna3SimdsPerCu;
    const int64_t wave_blocks = max_waves_per_cu / waves_per_block;

    const int64_t lds_blocks =
        kernel.lds_bytes > 0
            ? kCdna3LdsBytesPerCu / AlignUp(kernel.lds_bytes, kCdna3LdsGranule)
            : wave_blocks;

    // Whole-wave allocation in the unified file: architectural plus
    // accumulation VGPRs, never the architectural count alone.
    const int64_t registers =
        int64_t{kernel.arch_vgpr_count} + kernel.accum_vgpr_count;
    int64_t reg_blocks = wave_blocks;
    if (registers > 0) {
        const int64_t waves_per_simd = std::min(
            kCdna3WavesPerSimd,
            kCdna3RegistersPerSimd / AlignUp(registers, kCdna3RegisterGranule));
        reg_blocks = waves_per_simd * kCdna3SimdsPerCu / waves_per_block;
    }
    return FromBlockLimits(waves_per_block, max_waves_per_cu, wave_blocks,
                           lds_blocks, reg_blocks);
}

std::optional<AmdOccupancyEstimate> EstimateRdna(
    const AmdOccupancyDevice& device, const AmdOccupancyKernel& kernel) {
    if (kernel.threads_per_block == 0 || device.max_waves_per_cu == 0 ||
        device.wave_front_size == 0) {
        return std::nullopt;
    }
    const int64_t waves_per_block =
        (int64_t{kernel.threads_per_block} + device.wave_front_size - 1) /
        device.wave_front_size;
    const int64_t max_waves_per_cu = device.max_waves_per_cu;
    const int64_t wave_blocks = max_waves_per_cu / waves_per_block;

    const int64_t lds_blocks =
        kernel.lds_bytes > 0 && device.lds_size_bytes > 0
            ? int64_t{device.lds_size_bytes} / kernel.lds_bytes
            : wave_blocks;

    int64_t reg_blocks = wave_blocks;
    if (kernel.arch_vgpr_count > 0 && device.simd_per_cu > 0) {
        const int64_t reg_waves_per_cu =
            kRdnaVgprsPerSimd * device.simd_per_cu /
            AlignUp(kernel.arch_vgpr_count, kRdnaVgprGranule);
        reg_blocks = reg_waves_per_cu / waves_per_block;
    }
    return FromBlockLimits(waves_per_block, max_waves_per_cu, wave_blocks,
                           lds_blocks, reg_blocks);
}

}  // namespace

AmdOccupancyModel ResolveAmdOccupancyModel(std::string_view target) {
    target = target.substr(0, target.find(':'));
    if (target == "gfx940" || target == "gfx941" || target == "gfx942") {
        return AmdOccupancyModel::Cdna3;
    }
    if (target.rfind("gfx10", 0) == 0 || target.rfind("gfx11", 0) == 0 ||
        target.rfind("gfx12", 0) == 0) {
        return AmdOccupancyModel::Rdna;
    }
    return AmdOccupancyModel::None;
}

std::optional<AmdOccupancyEstimate> EstimateAmdOccupancy(
    const AmdOccupancyModel model, const AmdOccupancyDevice& device,
    const AmdOccupancyKernel& kernel) {
    switch (model) {
        case AmdOccupancyModel::Cdna3:
            return EstimateCdna3(kernel);
        case AmdOccupancyModel::Rdna:
            return EstimateRdna(device, kernel);
        case AmdOccupancyModel::None:
            break;
    }
    return std::nullopt;
}

}  // namespace gpufl::amd

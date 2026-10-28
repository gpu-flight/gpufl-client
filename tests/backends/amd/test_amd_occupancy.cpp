#include <gtest/gtest.h>

#include <algorithm>
#include <optional>
#include <string>

#include "gpufl/backends/amd/amd_capture_capabilities.hpp"
#include "gpufl/backends/amd/amd_occupancy.hpp"

namespace {

using gpufl::amd::AmdOccupancyDevice;
using gpufl::amd::AmdOccupancyEstimate;
using gpufl::amd::AmdOccupancyKernel;
using gpufl::amd::AmdOccupancyModel;
using gpufl::amd::EstimateAmdOccupancy;

// The CDNA 3 model reads nothing from the agent; this is what rc6 used with
// an MI300-series agent (wave64, four SIMDs, 8 waves per SIMD, 64 KiB LDS).
AmdOccupancyDevice Gfx942Agent() {
    AmdOccupancyDevice device;
    device.wave_front_size = 64;
    device.simd_per_cu = 4;
    device.max_waves_per_cu = 32;
    device.lds_size_bytes = 64 * 1024;
    return device;
}

// RX 9070 XT: wave32, two SIMDs per CU, 16 waves per SIMD.
AmdOccupancyDevice Gfx1201Agent() {
    AmdOccupancyDevice device;
    device.wave_front_size = 32;
    device.simd_per_cu = 2;
    device.max_waves_per_cu = 32;
    device.lds_size_bytes = 64 * 1024;
    return device;
}

AmdOccupancyKernel Kernel(const uint32_t threads, const uint32_t lds_bytes,
                          const uint32_t arch_vgprs,
                          const uint32_t accum_vgprs = 0) {
    AmdOccupancyKernel kernel;
    kernel.threads_per_block = threads;
    kernel.lds_bytes = lds_bytes;
    kernel.arch_vgpr_count = arch_vgprs;
    kernel.accum_vgpr_count = accum_vgprs;
    return kernel;
}

AmdOccupancyEstimate Estimate(const AmdOccupancyModel model,
                              const AmdOccupancyDevice& device,
                              const AmdOccupancyKernel& kernel) {
    const auto estimate = EstimateAmdOccupancy(model, device, kernel);
    EXPECT_TRUE(estimate.has_value());
    return estimate.value_or(AmdOccupancyEstimate{});
}

// rc6's estimate, verbatim apart from its inputs (rocprofiler_backend.cpp at
// v1.3.0rc6). The Rdna model must keep giving exactly this.
struct Rc6Estimate {
    float occupancy = 0.0f;
    float warp_occupancy = 0.0f;
    float smem_occupancy = 0.0f;
    float reg_occupancy = 0.0f;
    int max_active_blocks = 0;
    std::string limiting_resource;
};

std::optional<Rc6Estimate> Rc6(const AmdOccupancyDevice& props,
                               const int threadsPerBlock,
                               const int smemPerBlock,
                               const int arch_vgpr_count) {
    if (!(threadsPerBlock > 0 && props.max_waves_per_cu > 0 &&
          props.wave_front_size > 0)) {
        return std::nullopt;
    }
    int wavesPerBlock =
        (threadsPerBlock + static_cast<int>(props.wave_front_size) - 1) /
        static_cast<int>(props.wave_front_size);
    int maxWavesPerCU = static_cast<int>(props.max_waves_per_cu);
    int waveBlocks = (wavesPerBlock > 0) ? (maxWavesPerCU / wavesPerBlock) : 0;
    int ldsBytes = static_cast<int>(props.lds_size_bytes);
    int smemBlocks = (smemPerBlock > 0 && ldsBytes > 0)
                         ? (ldsBytes / smemPerBlock)
                         : waveBlocks;
    constexpr int kVgprAllocGranularity = 8;
    constexpr int kVgprsPerSimd = 1536;
    int vgprsPerThread = arch_vgpr_count;
    int vgprsAligned =
        (vgprsPerThread > 0)
            ? (((vgprsPerThread + kVgprAllocGranularity - 1) /
                kVgprAllocGranularity) *
               kVgprAllocGranularity)
            : 0;
    int regBlocks = waveBlocks;
    if (vgprsAligned > 0 && props.simd_per_cu > 0) {
        int totalVgprsPerCU =
            kVgprsPerSimd * static_cast<int>(props.simd_per_cu);
        int regWavesPerCU = totalVgprsPerCU / vgprsAligned;
        regBlocks = (wavesPerBlock > 0) ? (regWavesPerCU / wavesPerBlock)
                                        : waveBlocks;
    }
    int activeBlocks = (std::min)({waveBlocks, smemBlocks, regBlocks});
    activeBlocks = (std::max)(activeBlocks, 0);
    auto toOcc = [&](int blocks) -> float {
        return (maxWavesPerCU > 0 && wavesPerBlock > 0)
                   ? (std::min)(1.0f,
                                static_cast<float>(blocks * wavesPerBlock) /
                                    maxWavesPerCU)
                   : 0.0f;
    };
    Rc6Estimate out;
    out.occupancy = toOcc(activeBlocks);
    out.warp_occupancy = toOcc(waveBlocks);
    out.smem_occupancy = toOcc(smemBlocks);
    out.reg_occupancy = toOcc(regBlocks);
    out.max_active_blocks = activeBlocks;
    struct {
        float occ;
        const char* name;
    } limiters[] = {
        {out.warp_occupancy, "waves"},
        {out.reg_occupancy, "registers"},
        {out.smem_occupancy, "shared_mem"},
    };
    const char* limiting = "waves";
    float minOcc = out.warp_occupancy;
    for (auto& l : limiters) {
        if (l.occ < minOcc) {
            minOcc = l.occ;
            limiting = l.name;
        }
    }
    out.limiting_resource = limiting;
    return out;
}

const gpufl::CaptureCapability* FindCapability(
    const gpufl::CaptureCapabilitiesEvent& event, const std::string& feature) {
    for (const auto& capability : event.capabilities) {
        if (capability.feature == feature) return &capability;
    }
    return nullptr;
}

gpufl::CaptureCapabilitiesEvent TraceCapabilities(
    const uint64_t estimated_rows, const uint64_t unestimated_rows) {
    gpufl::amd::AmdCaptureCapabilityInput input;
    input.plan.requested_engine = gpufl::ProfilingEngine::Trace;
    input.plan.selected_path = gpufl::amd::AmdProfilingPath::BufferTracing;
    input.trace_configured = true;
    input.kernel_rows = estimated_rows + unestimated_rows;
    input.occupancy_estimated_rows = estimated_rows;
    input.occupancy_unestimated_rows = unestimated_rows;
    return gpufl::amd::BuildAmdCaptureCapabilitiesEvent(input);
}

}  // namespace

TEST(AmdOccupancyModel, ResolvesFromTheAgentTarget) {
    using gpufl::amd::ResolveAmdOccupancyModel;
    EXPECT_EQ(ResolveAmdOccupancyModel("gfx942"), AmdOccupancyModel::Cdna3);
    EXPECT_EQ(ResolveAmdOccupancyModel("gfx942:sramecc+:xnack-"),
              AmdOccupancyModel::Cdna3);
    EXPECT_EQ(ResolveAmdOccupancyModel("gfx940"), AmdOccupancyModel::Cdna3);
    EXPECT_EQ(ResolveAmdOccupancyModel("gfx941"), AmdOccupancyModel::Cdna3);
    EXPECT_EQ(ResolveAmdOccupancyModel("gfx1201"), AmdOccupancyModel::Rdna);
    EXPECT_EQ(ResolveAmdOccupancyModel("gfx1100"), AmdOccupancyModel::Rdna);
    EXPECT_EQ(ResolveAmdOccupancyModel("gfx1030"), AmdOccupancyModel::Rdna);
    // Wave64 targets with their own register files get no RDNA estimate.
    EXPECT_EQ(ResolveAmdOccupancyModel("gfx90a"), AmdOccupancyModel::None);
    EXPECT_EQ(ResolveAmdOccupancyModel("gfx908"), AmdOccupancyModel::None);
    EXPECT_EQ(ResolveAmdOccupancyModel("gfx950"), AmdOccupancyModel::None);
    EXPECT_EQ(ResolveAmdOccupancyModel(""), AmdOccupancyModel::None);
}

// Lab 0027's HIP answer (matmul register blocking, 256 threads, 4 KiB LDS).
// ROCm 7.2's compiler for gfx942 (-Rpass-analysis=kernel-resource-usage):
// VGPRs 112, AGPRs 0, occupancy 4 waves/SIMD. Its kernel descriptor gives
// rocprofiler-sdk arch_vgpr_count 112 and accum_vgpr_count 0.
TEST(AmdOccupancyModel, Cdna3Lab0027MatchesTheCompiler) {
    const auto estimate = Estimate(AmdOccupancyModel::Cdna3, Gfx942Agent(),
                                   Kernel(256, 4096, 112, 0));

    EXPECT_FLOAT_EQ(estimate.occupancy, 0.5f);  // 4 of 8 waves per SIMD
    EXPECT_FLOAT_EQ(estimate.reg_occupancy, 0.5f);
    EXPECT_FLOAT_EQ(estimate.warp_occupancy, 1.0f);
    EXPECT_FLOAT_EQ(estimate.smem_occupancy, 1.0f);
    EXPECT_EQ(estimate.max_active_blocks, 4);
    EXPECT_STREQ(estimate.limiting_resource, "registers");

    // rc6 gave this kernel RDNA's 1,536-entry file: 100%, limited by waves.
    const auto rc6 = Estimate(AmdOccupancyModel::Rdna, Gfx942Agent(),
                              Kernel(256, 4096, 112));
    EXPECT_FLOAT_EQ(rc6.occupancy, 1.0f);
    EXPECT_STREQ(rc6.limiting_resource, "waves");
}

// Lab 0042's answer (16x16 shared-memory tiles, 2 KiB dynamic LDS). gfx942:
// VGPRs 27, AGPRs 0, occupancy 8 waves/SIMD; arch_vgpr_count 28 (aligned to
// 4) plus accum_vgpr_count 4 fill one 32-register allocation.
TEST(AmdOccupancyModel, Cdna3Lab0042MatchesTheCompiler) {
    const auto estimate = Estimate(AmdOccupancyModel::Cdna3, Gfx942Agent(),
                                   Kernel(256, 2048, 28, 4));

    EXPECT_FLOAT_EQ(estimate.occupancy, 1.0f);
    EXPECT_FLOAT_EQ(estimate.reg_occupancy, 1.0f);
    EXPECT_EQ(estimate.max_active_blocks, 8);
    EXPECT_STREQ(estimate.limiting_resource, "waves");
}

// gfx942 kernel holding 64 AGPRs: VGPRs 6, AGPRs 64, occupancy 7 waves/SIMD.
// rocprofiler-sdk reports arch_vgpr_count 8 and accum_vgpr_count 64.
TEST(AmdOccupancyModel, Cdna3CountsAccumulationRegisters) {
    const auto estimate = Estimate(AmdOccupancyModel::Cdna3, Gfx942Agent(),
                                   Kernel(256, 0, 8, 64));

    // 7 waves on each of 4 SIMDs hold 7 blocks of 4 waves.
    EXPECT_EQ(estimate.max_active_blocks, 7);
    EXPECT_FLOAT_EQ(estimate.occupancy, 28.0f / 32.0f);
    EXPECT_STREQ(estimate.limiting_resource, "registers");
}

TEST(AmdOccupancyModel, Cdna3IgnoresTheAgentsWaveSlots) {
    AmdOccupancyDevice device = Gfx942Agent();
    device.max_waves_per_cu = 40;  // 10 per SIMD, as on gfx908
    device.lds_size_bytes = 0;

    const auto estimate =
        Estimate(AmdOccupancyModel::Cdna3, device, Kernel(64, 0, 8));

    EXPECT_EQ(estimate.max_active_blocks, 32);
    EXPECT_FLOAT_EQ(estimate.occupancy, 1.0f);
    EXPECT_STREQ(estimate.limiting_resource, "waves");
}

TEST(AmdOccupancyModel, Cdna3RoundsWholeBlocks) {
    // 72 registers allow 7 waves per SIMD, 28 per CU: one 16-wave block.
    const auto estimate = Estimate(AmdOccupancyModel::Cdna3, Gfx942Agent(),
                                   Kernel(1024, 0, 72));

    EXPECT_EQ(estimate.max_active_blocks, 1);
    EXPECT_FLOAT_EQ(estimate.occupancy, 0.5f);
    EXPECT_FLOAT_EQ(estimate.warp_occupancy, 1.0f);
    EXPECT_STREQ(estimate.limiting_resource, "registers");
}

TEST(AmdOccupancyModel, Cdna3AllocatesLdsInGranules) {
    // 13,000 bytes take 26 granules of 512 (13,312 bytes): four blocks fit
    // in 64 KiB, not five.
    const auto estimate = Estimate(AmdOccupancyModel::Cdna3, Gfx942Agent(),
                                   Kernel(64, 13000, 16));

    EXPECT_EQ(estimate.max_active_blocks, 4);
    EXPECT_FLOAT_EQ(estimate.occupancy, 4.0f / 32.0f);
    EXPECT_FLOAT_EQ(estimate.smem_occupancy, 4.0f / 32.0f);
    EXPECT_STREQ(estimate.limiting_resource, "shared_mem");
}

// gfx1201 keeps rc6's result. Lab 0027 there: VGPRs 101, 12 waves/SIMD from
// the compiler; arch_vgpr_count 104 from the descriptor.
TEST(AmdOccupancyModel, RdnaGfx1201KeepsItsResult) {
    const auto lab0027 = Estimate(AmdOccupancyModel::Rdna, Gfx1201Agent(),
                                  Kernel(256, 4096, 104));
    EXPECT_FLOAT_EQ(lab0027.occupancy, 0.75f);
    EXPECT_EQ(lab0027.max_active_blocks, 3);
    EXPECT_STREQ(lab0027.limiting_resource, "registers");

    const auto lab0042 = Estimate(AmdOccupancyModel::Rdna, Gfx1201Agent(),
                                  Kernel(256, 2048, 32));
    EXPECT_FLOAT_EQ(lab0042.occupancy, 1.0f);
    EXPECT_EQ(lab0042.max_active_blocks, 4);
    EXPECT_STREQ(lab0042.limiting_resource, "waves");
}

TEST(AmdOccupancyModel, RdnaMatchesRc6ForEveryShape) {
    const AmdOccupancyDevice devices[] = {Gfx1201Agent(), Gfx942Agent()};
    const uint32_t threads[] = {1, 32, 64, 96, 128, 192, 256, 512, 1024};
    const uint32_t lds[] = {0, 512, 2048, 4096, 13000, 40000, 65536};
    const uint32_t vgprs[] = {0, 8, 24, 32, 64, 100, 104, 128, 200, 256};
    for (const auto& device : devices) {
        for (const uint32_t t : threads) {
            for (const uint32_t l : lds) {
                for (const uint32_t v : vgprs) {
                    SCOPED_TRACE(std::to_string(device.wave_front_size) + "/" +
                                 std::to_string(t) + "/" + std::to_string(l) +
                                 "/" + std::to_string(v));
                    const auto expected =
                        Rc6(device, static_cast<int>(t), static_cast<int>(l),
                            static_cast<int>(v));
                    const auto actual = EstimateAmdOccupancy(
                        AmdOccupancyModel::Rdna, device, Kernel(t, l, v, 16));
                    ASSERT_EQ(actual.has_value(), expected.has_value());
                    EXPECT_EQ(actual->occupancy, expected->occupancy);
                    EXPECT_EQ(actual->warp_occupancy, expected->warp_occupancy);
                    EXPECT_EQ(actual->smem_occupancy, expected->smem_occupancy);
                    EXPECT_EQ(actual->reg_occupancy, expected->reg_occupancy);
                    EXPECT_EQ(actual->block_occupancy, 1.0f);
                    EXPECT_EQ(actual->max_active_blocks,
                              expected->max_active_blocks);
                    EXPECT_EQ(std::string(actual->limiting_resource),
                              expected->limiting_resource);
                }
            }
        }
    }
}

TEST(AmdOccupancyModel, WithoutAModelOrShapeThereIsNoEstimate) {
    EXPECT_FALSE(EstimateAmdOccupancy(AmdOccupancyModel::None, Gfx942Agent(),
                                      Kernel(256, 0, 32))
                     .has_value());
    EXPECT_FALSE(EstimateAmdOccupancy(AmdOccupancyModel::Cdna3, Gfx942Agent(),
                                      Kernel(0, 0, 32))
                     .has_value());
    AmdOccupancyDevice no_slots = Gfx1201Agent();
    no_slots.max_waves_per_cu = 0;
    EXPECT_FALSE(EstimateAmdOccupancy(AmdOccupancyModel::Rdna, no_slots,
                                      Kernel(256, 0, 32))
                     .has_value());
}

TEST(AmdOccupancyCapability, CollectedWhenEveryKernelRowWasEstimated) {
    const auto event = TraceCapabilities(12, 0);
    const auto* row = FindCapability(event, "occupancy_estimate");
    ASSERT_NE(row, nullptr);

    EXPECT_TRUE(row->requested);
    EXPECT_EQ(row->status, "collected");
    EXPECT_EQ(row->mode, "amd_target_model_v1");
    EXPECT_TRUE(row->reason_code.empty());
}

TEST(AmdOccupancyCapability, NamesKernelRowsWithoutAModel) {
    const auto partial = TraceCapabilities(5, 2);
    const auto* mixed = FindCapability(partial, "occupancy_estimate");
    ASSERT_NE(mixed, nullptr);
    EXPECT_EQ(mixed->status, "partial");
    EXPECT_EQ(mixed->mode, "amd_target_model_v1");
    EXPECT_EQ(mixed->reason_code, "amd_occupancy_target_unmodeled");
    EXPECT_NE(mixed->message.find("2 kernel row(s)"), std::string::npos);

    const auto none = TraceCapabilities(0, 3);
    const auto* unmodeled = FindCapability(none, "occupancy_estimate");
    ASSERT_NE(unmodeled, nullptr);
    EXPECT_EQ(unmodeled->status, "skipped");
    EXPECT_EQ(unmodeled->reason_code, "amd_occupancy_target_unmodeled");
}

TEST(AmdOccupancyCapability, ReportsSessionsWithoutKernelsOrTracing) {
    const auto empty = TraceCapabilities(0, 0);
    const auto* idle = FindCapability(empty, "occupancy_estimate");
    ASSERT_NE(idle, nullptr);
    EXPECT_EQ(idle->status, "enabled_no_data");
    EXPECT_EQ(idle->reason_code, "enabled_but_no_records");

    gpufl::amd::AmdCaptureCapabilityInput untraced;
    untraced.plan.requested_engine = gpufl::ProfilingEngine::Trace;
    const auto skipped_event =
        gpufl::amd::BuildAmdCaptureCapabilitiesEvent(untraced);
    const auto* skipped = FindCapability(skipped_event, "occupancy_estimate");
    ASSERT_NE(skipped, nullptr);
    EXPECT_EQ(skipped->status, "skipped");
    EXPECT_EQ(skipped->mode, "disabled");
    EXPECT_EQ(skipped->reason_code, "rocprofiler_buffer_tracing_unavailable");

    gpufl::amd::AmdCaptureCapabilityInput monitor;
    monitor.trace_configured = true;
    const auto monitor_event =
        gpufl::amd::BuildAmdCaptureCapabilitiesEvent(monitor);
    const auto* not_requested =
        FindCapability(monitor_event, "occupancy_estimate");
    ASSERT_NE(not_requested, nullptr);
    EXPECT_FALSE(not_requested->requested);
    EXPECT_EQ(not_requested->status, "not_requested");
}

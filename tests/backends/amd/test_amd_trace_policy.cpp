#include <gtest/gtest.h>

#include "gpufl/backends/amd/amd_trace_policy.hpp"

namespace {

gpufl::amd::AmdTraceEndpoint Cpu() {
    return {gpufl::amd::AmdTraceAgentKind::Cpu, std::nullopt};
}

gpufl::amd::AmdTraceEndpoint Gpu(const uint32_t device_id) {
    return {gpufl::amd::AmdTraceAgentKind::Gpu, device_id};
}

}  // namespace

TEST(AmdTracePolicy, KernelUsesItsGpuAgentDevice) {
    EXPECT_EQ(gpufl::amd::ResolveAmdKernelDeviceId(Gpu(3)), 3u);
    EXPECT_FALSE(gpufl::amd::ResolveAmdKernelDeviceId(Cpu()).has_value());
}

TEST(AmdTracePolicy, HostToDeviceUsesDestinationGpu) {
    EXPECT_EQ(gpufl::amd::ResolveAmdMemoryCopyDeviceId(Cpu(), Gpu(2)), 2u);
}

TEST(AmdTracePolicy, DeviceToHostUsesSourceGpu) {
    EXPECT_EQ(gpufl::amd::ResolveAmdMemoryCopyDeviceId(Gpu(4), Cpu()), 4u);
}

TEST(AmdTracePolicy, DeviceToDeviceUsesDestinationGpu) {
    EXPECT_EQ(gpufl::amd::ResolveAmdMemoryCopyDeviceId(Gpu(1), Gpu(5)), 5u);
}

TEST(AmdTracePolicy, HostOnlyAndUnmappedGpuCopiesAreUnattributed) {
    EXPECT_FALSE(
        gpufl::amd::ResolveAmdMemoryCopyDeviceId(Cpu(), Cpu()).has_value());
    const gpufl::amd::AmdTraceEndpoint unmapped_gpu{
        gpufl::amd::AmdTraceAgentKind::Gpu, std::nullopt};
    EXPECT_FALSE(gpufl::amd::ResolveAmdMemoryCopyDeviceId(unmapped_gpu, Cpu())
                     .has_value());
}

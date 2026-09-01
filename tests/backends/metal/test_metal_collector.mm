#include <gtest/gtest.h>

#include <string>

#include "gpufl/backends/metal/metal_collector.hpp"

namespace {

TEST(MetalCollector, EmitsPublicInventoryAndExplicitTelemetrySemantics) {
    std::string reason;
    ASSERT_TRUE(gpufl::metal::MetalCollector::IsAvailable(&reason)) << reason;

    gpufl::metal::MetalCollector collector;
    const auto samples = collector.sampleAll();
    const auto static_info = collector.sampleStaticInfo();

    ASSERT_FALSE(samples.empty());
    ASSERT_EQ(samples.size(), static_info.size());

    const auto& sample = samples.front();
    EXPECT_EQ(sample.vendor, "Apple");
    EXPECT_FALSE(sample.name.empty());
    EXPECT_EQ(sample.used_mib, 0u);
    EXPECT_EQ(sample.free_mib, 0u);
    EXPECT_EQ(sample.total_mib, 0u);
    EXPECT_EQ(sample.telemetry_capabilities.allocation_scope,
              "current_process");
    EXPECT_FALSE(sample.telemetry_capabilities.memory_model.empty());
    EXPECT_GT(
        sample.telemetry_capabilities.recommended_max_working_set_mib, 0u);
    EXPECT_FALSE(sample.telemetry_capabilities.unavailable.empty());

    const auto& info = static_info.front();
    EXPECT_EQ(info.vendor, "Apple");
    EXPECT_TRUE(info.metal.available);
    EXPECT_FALSE(info.metal.registry_id.empty());
    EXPECT_FALSE(info.architecture.empty());
    EXPECT_GT(info.metal.recommended_max_working_set_bytes, 0u);
    EXPECT_GT(info.metal.max_buffer_length_bytes, 0u);
    EXPECT_GT(info.metal.max_threads_per_threadgroup[0], 0u);
    EXPECT_FALSE(info.metal.gpu_families.empty());
}

}  // namespace

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>

#include "gpufl/backends/amd/rocm_collector.hpp"

#if GPUFL_ENABLE_AMD && GPUFL_HAS_ROCM_SMI

class RocmCollectorTest : public ::testing::Test {
   protected:
    void SetUp() override {
        std::string reason;
        if (!gpufl::amd::RocmCollector::IsAvailable(&reason)) {
            GTEST_SKIP() << "AMD backend not available: " << reason;
        }
    }
};

TEST_F(RocmCollectorTest, Availability) {
    EXPECT_TRUE(gpufl::amd::RocmCollector::IsAvailable());
}

TEST_F(RocmCollectorTest, ReportsCapabilityFlags) {
    gpufl::amd::RocmCollector collector;

    EXPECT_TRUE(collector.canSampleTelemetry() || collector.canSampleStaticInfo());
}

TEST_F(RocmCollectorTest, SampleDynamicMetrics) {
    gpufl::amd::RocmCollector collector;

    if (!collector.canSampleTelemetry()) {
        GTEST_SKIP() << "ROCm telemetry unavailable in this environment";
    }

    auto samples = collector.sampleAll();

    EXPECT_FALSE(samples.empty());

    for (const auto& sample : samples) {
        EXPECT_GE(sample.device_id, 0);
        EXPECT_EQ(sample.vendor, "AMD");

        if (!sample.name.empty()) {
            EXPECT_FALSE(sample.name.empty());
        }

        if (sample.total_mib > 0) {
            EXPECT_LE(sample.used_mib, sample.total_mib);
            EXPECT_EQ(sample.free_mib, sample.total_mib - sample.used_mib);
        }

        EXPECT_LE(sample.gpu_util, 100u);
        EXPECT_LE(sample.mem_util, 100u);
    }
}

TEST_F(RocmCollectorTest, SampleStaticDeviceInfo) {
    gpufl::amd::RocmCollector collector;

    if (!collector.canSampleStaticInfo()) {
        GTEST_SKIP() << "HIP static device inventory unavailable in this environment";
    }

    auto infos = collector.sampleStaticInfo();

    EXPECT_FALSE(infos.empty());

    for (const auto& info : infos) {
        std::string arch = info.architecture;
        std::transform(arch.begin(), arch.end(), arch.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        std::string name = info.name;
        std::transform(name.begin(), name.end(), name.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        EXPECT_GE(info.id, 0);
        EXPECT_EQ(info.vendor, "AMD");
        EXPECT_FALSE(info.name.empty());
        EXPECT_FALSE(info.architecture.empty());
        EXPECT_EQ(arch.rfind("gfx", 0), 0u);
        EXPECT_EQ(name.find("ryzen"), std::string::npos);
        EXPECT_EQ(name.find("epyc"), std::string::npos);
        EXPECT_EQ(name.find("threadripper"), std::string::npos);
        EXPECT_GT(info.multi_processor_count, 0);
        EXPECT_GT(info.warp_size, 0);
    }
}

#endif

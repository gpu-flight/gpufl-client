#include <gtest/gtest.h>

#include "gpufl/backends/amd/rocm_collector.hpp"

#if GPUFL_ENABLE_AMD && GPUFL_HAS_ROCM_SMI

class RocmCollectorTest : public ::testing::Test {
   protected:
    void SetUp() override {
        std::string reason;
        if (!gpufl::amd::RocmCollector::IsAvailable(&reason)) {
            GTEST_SKIP() << "ROCm SMI not available: " << reason;
        }
    }
};

TEST_F(RocmCollectorTest, Availability) {
    EXPECT_TRUE(gpufl::amd::RocmCollector::IsAvailable());
}

TEST_F(RocmCollectorTest, SampleDynamicMetrics) {
    gpufl::amd::RocmCollector collector;
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

#endif

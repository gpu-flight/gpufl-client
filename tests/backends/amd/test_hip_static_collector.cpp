#include <gtest/gtest.h>

#include "gpufl/backends/amd/hip_static_collector.hpp"

#if GPUFL_ENABLE_AMD && GPUFL_HAS_HIP

class HipStaticCollectorTest : public ::testing::Test {
   protected:
    void SetUp() override {
        std::string reason;
        if (!gpufl::amd::HipStaticCollector::IsAvailable(&reason)) {
            GTEST_SKIP() << "HIP static inventory not available: " << reason;
        }
    }
};

TEST_F(HipStaticCollectorTest, Availability) {
    EXPECT_TRUE(gpufl::amd::HipStaticCollector::IsAvailable());
}

TEST_F(HipStaticCollectorTest, SampleStaticDeviceInfo) {
    gpufl::amd::HipStaticCollector collector;
    auto infos = collector.sampleAll();

    EXPECT_FALSE(infos.empty());

    for (const auto& info : infos) {
        EXPECT_GE(info.id, 0);
        EXPECT_EQ(info.vendor, "AMD");
        EXPECT_FALSE(info.name.empty());
        EXPECT_FALSE(info.architecture.empty());
        EXPECT_GT(info.multi_processor_count, 0);
        EXPECT_GT(info.warp_size, 0);
    }
}

#endif

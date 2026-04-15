#include <gtest/gtest.h>

#include "common/test_utils.hpp"

#if GPUFL_ENABLE_NVIDIA && GPUFL_HAS_NVML
#include "gpufl/backends/nvidia/nvml_collector.hpp"

class NvmlCollectorTest : public ::testing::Test {
   protected:
    void SetUp() override {
        SKIP_IF_NO_CUDA();
        std::string reason;
        if (!gpufl::nvidia::NvmlCollector::IsAvailable(&reason)) {
            GTEST_SKIP() << "NVML not available: " << reason;
        }
    }
};

TEST_F(NvmlCollectorTest, Availability) {
    // Already checked in SetUp, but good to have a dedicated test
    EXPECT_TRUE(gpufl::nvidia::NvmlCollector::IsAvailable());
}

TEST_F(NvmlCollectorTest, SampleDynamicMetrics) {
    gpufl::nvidia::NvmlCollector collector;
    auto samples = collector.sampleAll();

    EXPECT_FALSE(samples.empty());

    for (const auto& sample : samples) {
        EXPECT_GE(sample.device_id, 0);
        EXPECT_FALSE(sample.name.empty());

        // Memory metrics
        EXPECT_GT(sample.total_mib, 0);
        EXPECT_LE(sample.used_mib, sample.total_mib);
        // Sum might be off by 1 MiB due to floor division in ToMiB(bytes)
        EXPECT_NEAR(sample.total_mib, sample.used_mib + sample.free_mib, 1.1);

        // Utilization metrics (0-100)
        EXPECT_LE(sample.gpu_util, 100);
        EXPECT_LE(sample.mem_util, 100);

        // Clock metrics — may be 0 when GPU is idle (power-saving mode)
        EXPECT_GE(sample.clock_sm, 0);
        EXPECT_GE(sample.clock_mem, 0);

        // Temperature
        EXPECT_GT(sample.temp_c, 0);
        EXPECT_LT(sample.temp_c, 120);  // Sanity check
    }
}

#endif

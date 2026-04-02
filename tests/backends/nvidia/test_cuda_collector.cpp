#include <gtest/gtest.h>

#include "common/test_utils.hpp"

#if GPUFL_ENABLE_NVIDIA && GPUFL_HAS_CUDA
#include "gpufl/backends/nvidia/cuda_collector.hpp"

class CudaCollectorTest : public ::testing::Test {
   protected:
    void SetUp() override { SKIP_IF_NO_CUDA(); }
};

TEST_F(CudaCollectorTest, SampleStaticDeviceInfo) {
    gpufl::nvidia::CudaCollector collector;
    auto infos = collector.sampleAll();

    // We expect at least one CUDA device if we didn't skip
    EXPECT_FALSE(infos.empty());

    for (const auto& info : infos) {
        EXPECT_GE(info.id, 0);
        EXPECT_FALSE(info.name.empty());
        EXPECT_FALSE(info.uuid.empty());
        EXPECT_GT(info.compute_major, 0);
        EXPECT_GT(info.multi_processor_count, 0);
        EXPECT_GT(info.warp_size, 0);

        // Sanity checks on properties
        EXPECT_GT(info.shared_mem_per_block, 0);
        EXPECT_GT(info.regs_per_block, 0);
    }
}

#endif

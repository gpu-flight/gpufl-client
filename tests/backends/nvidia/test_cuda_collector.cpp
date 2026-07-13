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
        EXPECT_EQ(info.architecture,
                  "sm_" + std::to_string(info.compute_major) +
                      std::to_string(info.compute_minor));
        EXPECT_GT(info.multi_processor_count, 0);
        EXPECT_GT(info.warp_size, 0);

        // Sanity checks on properties
        EXPECT_GT(info.shared_mem_per_block, 0);
        EXPECT_GT(info.regs_per_block, 0);
        EXPECT_GT(info.total_global_mem, 0u);
        EXPECT_GT(info.total_const_mem, 0u);
        EXPECT_GT(info.shared_mem_per_block_optin, 0);
        EXPECT_GT(info.shared_mem_per_multiprocessor, 0);
        EXPECT_GT(info.regs_per_multiprocessor, 0);
        EXPECT_GT(info.max_threads_per_block, 0);
        EXPECT_GT(info.max_threads_per_multiprocessor, 0);
        EXPECT_GT(info.max_blocks_per_multiprocessor, 0);
        EXPECT_GT(info.max_threads_dim[0], 0);
        EXPECT_GT(info.max_grid_size[0], 0);
        EXPECT_GT(info.clock_rate_khz, 0);
        EXPECT_GT(info.memory_clock_rate_khz, 0);
        EXPECT_GT(info.memory_bus_width_bits, 0);
        EXPECT_FALSE(info.tensor_map_access_supported);
    }
}

TEST_F(CudaCollectorTest, InfoCapabilitiesCanBeEnrichedSeparately) {
    gpufl::nvidia::CudaCollector collector;
    auto infos = collector.sampleAll();

    EXPECT_NO_THROW(gpufl::nvidia::EnrichCudaInfoCapabilities(infos));
    EXPECT_FALSE(infos.empty());
}

TEST_F(CudaCollectorTest, RuntimeVersionsAreAvailable) {
    const auto versions = gpufl::nvidia::QueryCudaRuntimeVersions();

    EXPECT_GT(versions.driver, 0);
    EXPECT_GT(versions.runtime, 0);
}

#endif

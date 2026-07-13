#include <gtest/gtest.h>

#include <string>

#include "gpufl/core/json/json.hpp"
#include "info_command.hpp"

using namespace gpufl::launcher;

namespace {

InfoReport sampleReport() {
    gpufl::GpuStaticDeviceInfo device{};
    device.id = 0;
    device.name = "NVIDIA Test \"GPU\"";
    device.uuid = "GPU-1234";
    device.vendor = "NVIDIA";
    device.compute_major = 8;
    device.compute_minor = 9;
    device.total_global_mem = 24ULL * 1024 * 1024 * 1024;
    device.total_const_mem = 64ULL * 1024;
    device.l2_cache_size = 48 * 1024 * 1024;
    device.shared_mem_per_block = 48 * 1024;
    device.shared_mem_per_block_optin = 99 * 1024;
    device.shared_mem_per_multiprocessor = 100 * 1024;
    device.regs_per_block = 65536;
    device.regs_per_multiprocessor = 65536;
    device.multi_processor_count = 60;
    device.warp_size = 32;
    device.max_threads_per_block = 1024;
    device.max_threads_per_multiprocessor = 1536;
    device.max_blocks_per_multiprocessor = 24;
    device.max_threads_dim = {1024, 1024, 64};
    device.max_grid_size = {2147483647, 65535, 65535};
    device.clock_rate_khz = 2040000;
    device.memory_clock_rate_khz = 6251000;
    device.memory_bus_width_bits = 192;
    device.async_engine_count = 3;
    device.concurrent_kernels = true;
    device.cooperative_launch = true;
    device.unified_addressing = true;
    device.managed_memory = true;
    device.memory_pools_supported = true;
    device.cluster_launch = false;
    device.tensor_map_access_supported = false;

    InfoReport report;
    report.cuda_driver_version = 13030;
    report.cuda_runtime_version = 13030;
    report.devices.push_back(device);
    return report;
}

}  // namespace

TEST(InfoCommand, JsonUsesVersionedNestedSchema) {
    const std::string output = renderInfoJson(sampleReport());
    const gpufl::json::JsonValue root = gpufl::json::parseJson(output);

    ASSERT_TRUE(root.is_object()) << output;
    EXPECT_EQ(root.value<int>("schemaVersion", 0), 1);
    EXPECT_FALSE(root.value<std::string>("gpuflVersion", "").empty());
    EXPECT_EQ(root.value<int>("cudaDriverVersion", 0), 13030);
    ASSERT_TRUE(root["devices"].is_array());
    ASSERT_EQ(root["devices"].size(), 1u);

    const auto& device = root["devices"][0];
    EXPECT_EQ(device.value<std::string>("name", ""), "NVIDIA Test \"GPU\"");
    EXPECT_EQ(device.value<std::string>("architecture", ""), "sm_89");
    EXPECT_EQ(device["computeCapability"].value<int>("major", 0), 8);
    EXPECT_EQ(device["computeCapability"].value<int>("minor", 0), 9);
    EXPECT_EQ(device["memory"].value<uint64_t>("totalGlobalBytes", 0),
              24ULL * 1024 * 1024 * 1024);
    EXPECT_EQ(device["executionLimits"].value<int>("maxThreadsPerBlock", 0),
              1024);
    ASSERT_TRUE(device["executionLimits"]["maxBlockDimensions"].is_array());
    EXPECT_EQ(device["executionLimits"]["maxBlockDimensions"][2].as_int(), 64);
    EXPECT_TRUE(device["features"].value<bool>("concurrentKernels", false));
    EXPECT_FALSE(device["features"].value<bool>("tensorMapAccess", true));
}

TEST(InfoCommand, JsonUsesNullForUnavailableNumericCapabilities) {
    InfoReport report;
    gpufl::GpuStaticDeviceInfo device{};
    device.id = 0;
    device.name = "Unknown GPU";
    device.vendor = "NVIDIA";
    report.devices.push_back(device);

    const auto root = gpufl::json::parseJson(renderInfoJson(report));
    ASSERT_TRUE(root.is_object());
    EXPECT_TRUE(root["cudaDriverVersion"].is_null());
    EXPECT_TRUE(root["devices"][0]["memory"]["totalGlobalBytes"].is_null());
    EXPECT_TRUE(root["devices"][0]["executionLimits"]["maxBlockDimensions"].is_null());
}

TEST(InfoCommand, TextGroupsCapabilitiesForHumans) {
    const std::string output = renderInfoText(sampleReport());

    EXPECT_NE(output.find("GPU 0: NVIDIA Test \"GPU\""), std::string::npos);
    EXPECT_NE(output.find("Compute capability:     8.9"), std::string::npos);
    EXPECT_NE(output.find("Core and execution"), std::string::npos);
    EXPECT_NE(output.find("Memory and cache"), std::string::npos);
    EXPECT_NE(output.find("Global memory:        24.00 GiB"), std::string::npos);
    EXPECT_NE(output.find("Tensor Map access:    no"), std::string::npos);
}

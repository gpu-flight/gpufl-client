#include <gtest/gtest.h>

#include "gpufl/gpufl.hpp"

TEST(CoreLogic, InitOptionsDefault) {
    gpufl::InitOptions opts;
    EXPECT_EQ(opts.app_name, "gpufl");
    EXPECT_FALSE(opts.sampling_auto_start);
    EXPECT_EQ(opts.profiling_engine, gpufl::ProfilingEngine::PcSamplingWithSass);
}

TEST(CoreLogic, BackendKindEnum) {
    EXPECT_EQ(static_cast<int>(gpufl::BackendKind::Auto), 0);
    EXPECT_EQ(static_cast<int>(gpufl::BackendKind::Nvidia), 1);
}

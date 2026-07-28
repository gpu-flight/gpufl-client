#include <gtest/gtest.h>

#include "gpufl/core/debug_logger.hpp"

namespace {

class DebugLoggerTest : public ::testing::Test {
   protected:
    void SetUp() override { gpufl::DebugLogger::setEnabled(false); }
    void TearDown() override { gpufl::DebugLogger::setEnabled(false); }
};

TEST_F(DebugLoggerTest, InfoIsAlwaysVisibleWithoutSourceLocation) {
    testing::internal::CaptureStderr();
    GFL_LOG_INFO("prepared ", 3, " engines");
    const std::string output = testing::internal::GetCapturedStderr();

    EXPECT_EQ(output, "[GPUFL] prepared 3 engines\n");
    EXPECT_EQ(output.find(__FILE__), std::string::npos);
}

TEST_F(DebugLoggerTest, WarningIsAlwaysVisibleWithoutSourceLocation) {
    testing::internal::CaptureStderr();
    GFL_LOG_WARN("sampling returned no data");
    const std::string output = testing::internal::GetCapturedStderr();

    EXPECT_EQ(output, "[GPUFL-WARN] sampling returned no data\n");
    EXPECT_EQ(output.find(__FILE__), std::string::npos);
}

TEST_F(DebugLoggerTest, DebugRemainsControlledByTheVerboseFlag) {
    testing::internal::CaptureStdout();
    GFL_LOG_DEBUG("hidden");
    EXPECT_TRUE(testing::internal::GetCapturedStdout().empty());

    gpufl::DebugLogger::setEnabled(true);
    testing::internal::CaptureStdout();
    GFL_LOG_DEBUG("visible");
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "[GPUFL] visible\n");
}

}  // namespace

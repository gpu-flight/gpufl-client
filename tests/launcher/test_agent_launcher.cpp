#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "agent_launcher.hpp"

namespace {

std::vector<std::string> exitCommand(const int code) {
#ifdef _WIN32
    return {"cmd.exe", "/d", "/s", "/c", "exit " + std::to_string(code)};
#else
    return {"/bin/sh", "-c", "exit " + std::to_string(code)};
#endif
}

}  // namespace

TEST(AgentProcessTest, AZeroExitIsACompletedUpload) {
    gpufl::launcher::AgentProcess process;
    std::string error;
    ASSERT_TRUE(process.start(exitCommand(0), error)) << error;

    const auto result = process.waitForExit(5000);

    EXPECT_TRUE(result.exited);
    EXPECT_EQ(result.exit_code, 0);
    EXPECT_TRUE(result.succeeded());
}

TEST(AgentProcessTest, ANonzeroExitIsNotACompletedUpload) {
    gpufl::launcher::AgentProcess process;
    std::string error;
    ASSERT_TRUE(process.start(exitCommand(7), error)) << error;

    const auto result = process.waitForExit(5000);

    EXPECT_TRUE(result.exited);
    EXPECT_EQ(result.exit_code, 7);
    EXPECT_FALSE(result.succeeded());
}

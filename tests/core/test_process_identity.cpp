#include <gtest/gtest.h>

#include <cstdlib>
#include <optional>
#include <string>

#include "gpufl/core/env_vars.hpp"
#include "gpufl/core/process_identity.hpp"

namespace {

void setEnv(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    ::setenv(name, value, /*overwrite=*/1);
#endif
}

void unsetEnv(const char* name) {
#if defined(_WIN32)
    _putenv_s(name, "");
#else
    ::unsetenv(name);
#endif
}

TEST(ProcessIdentityTest, ReadersDescribeTheTestProcess) {
    EXPECT_GT(gpufl::getParentPid(), 0);
    const std::string name = gpufl::getProcessName();
    const std::string argv0 = gpufl::getProcessArgv0();
    EXPECT_FALSE(name.empty());
    EXPECT_LE(name.size(), 64u);
    EXPECT_FALSE(argv0.empty());
    EXPECT_LE(argv0.size(), 128u);
    EXPECT_EQ(argv0.find('/'), std::string::npos) << argv0;
    EXPECT_EQ(argv0.find('\\'), std::string::npos) << argv0;
}

// The Linux rule, which is how a retitled vLLM engine process is recognized.
TEST(ProcessIdentityTest, Argv0IsTheBaseNameOfTheFirstWordOnly) {
    using gpufl::detail::argv0FromCmdline;
    using namespace std::string_literals;
    EXPECT_EQ(argv0FromCmdline("/usr/bin/python3\0-m\0vllm\0"s), "python3");
    // setproctitle() overwrites the argument area with one padded string.
    EXPECT_EQ(argv0FromCmdline("VLLM::EngineCore   \0\0"s), "VLLM::EngineCore");
    EXPECT_EQ(argv0FromCmdline("/opt/venv/bin/python3 -m vllm serve --api-key x"),
              "python3");
    EXPECT_EQ(argv0FromCmdline(std::string(200, 'a')).size(), 128u);
}

TEST(ProcessIdentityTest, InvalidInheritedLaunchIdIsDroppedNotFatal) {
    std::optional<std::string> saved;
    if (const char* value = std::getenv(gpufl::env::kLaunchId)) saved = value;

    setEnv(gpufl::env::kLaunchId, "launch_01-a");
    EXPECT_EQ(gpufl::launchIdFromEnvironment(), "launch_01-a");
    setEnv(gpufl::env::kLaunchId, "not valid!");
    EXPECT_EQ(gpufl::launchIdFromEnvironment(), "");
    unsetEnv(gpufl::env::kLaunchId);
    EXPECT_EQ(gpufl::launchIdFromEnvironment(), "");

    if (saved) setEnv(gpufl::env::kLaunchId, saved->c_str());
}

}  // namespace

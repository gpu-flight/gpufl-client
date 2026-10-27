#include <gtest/gtest.h>

#include <cstdlib>
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

// A child that exits 0 only when GPUFL_LOG_TYPES is absent from its environment.
std::vector<std::string> logTypesAbsentCommand() {
#ifdef _WIN32
    return {"cmd.exe", "/d", "/s", "/c",
            "if defined GPUFL_LOG_TYPES (exit 1) else (exit 0)"};
#else
    return {"/bin/sh", "-c", "test -z \"${GPUFL_LOG_TYPES+set}\""};
#endif
}

void setEnv(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    ::setenv(name, value, /*overwrite=*/1);
#endif
}

void unsetEnv(const char* name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    ::unsetenv(name);
#endif
}

// The child's exit code, or -1 when it could not be started or did not exit.
int childExitCode(const std::vector<std::string>& command) {
    gpufl::launcher::AgentProcess process;
    std::string error;
    if (!process.start(command, error)) return -1;
    const auto result = process.waitForExit(5000);
    return result.exited ? result.exit_code : -1;
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

TEST(AgentEnvironmentTest, AnEmptyChannelListRemovesAnInheritedFilter) {
    setEnv("GPUFL_LOG_TYPES", "device,scope");  // as if exported by the caller's shell
    ASSERT_EQ(childExitCode(logTypesAbsentCommand()), 1);  // the child sees it

    gpufl::launcher::AgentOptions opts;
    opts.source_folders = "session-dir";
    opts.cursor_file = "cursor.json";
    opts.backend_url = "http://localhost:1";
    opts.api_key = "test-key";
    std::string error;
    ASSERT_TRUE(gpufl::launcher::configureAgentEnvironment(opts, error)) << error;

    EXPECT_EQ(childExitCode(logTypesAbsentCommand()), 0);

    // configureAgentEnvironment publishes into this process; leave it clean.
    for (const char* name :
         {"GPUFL_SOURCE_FOLDER", "GPUFL_SOURCE_FOLDERS", "GPUFL_CURSOR_FILE",
          "GPUFL_PUBLISHER_TYPE", "GPUFL_HTTP_HOST", "GPUFL_HTTP_TOKEN",
          "GPUFL_HTTP_API_VERSION", "GPUFL_AGENT_UPLOAD_MODE",
          "GPUFL_AGENT_EXIT_WHEN_DRAINED"}) {
        unsetEnv(name);
    }
}

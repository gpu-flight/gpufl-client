#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

#include "gpufl/core/common.hpp"
#include "gpufl/core/env_vars.hpp"
#include "gpufl/core/startup_configuration.hpp"

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

class StartupConfigurationTest : public testing::Test {
protected:
    void SetUp() override {
        saveAndUnset_(gpufl::env::kConfigFile, config_file_);
        saveAndUnset_(gpufl::env::kApiPath, api_path_);
        saveAndUnset_(gpufl::env::kRunId, run_id_);
        saveAndUnset_(gpufl::env::kSegmentEveryMs, segment_every_ms_);
        saveAndUnset_(gpufl::env::kSegmentMaxRows, segment_max_rows_);
        saveAndUnset_(gpufl::env::kRunRollEveryMs, run_roll_every_ms_);
        saveAndUnset_(gpufl::env::kRunRollMaxBytes, run_roll_max_bytes_);
    }

    void TearDown() override {
        restore_(gpufl::env::kConfigFile, config_file_);
        restore_(gpufl::env::kApiPath, api_path_);
        restore_(gpufl::env::kRunId, run_id_);
        restore_(gpufl::env::kSegmentEveryMs, segment_every_ms_);
        restore_(gpufl::env::kSegmentMaxRows, segment_max_rows_);
        restore_(gpufl::env::kRunRollEveryMs, run_roll_every_ms_);
        restore_(gpufl::env::kRunRollMaxBytes, run_roll_max_bytes_);
    }

private:
    static void saveAndUnset_(const char* name, std::optional<std::string>& out) {
        if (const char* value = std::getenv(name)) out = value;
        unsetEnv(name);
    }

    static void restore_(const char* name,
                         const std::optional<std::string>& value) {
        if (value) setEnv(name, value->c_str());
        else unsetEnv(name);
    }

    std::optional<std::string> config_file_;
    std::optional<std::string> api_path_;
    std::optional<std::string> run_id_;
    std::optional<std::string> segment_every_ms_;
    std::optional<std::string> segment_max_rows_;
    std::optional<std::string> run_roll_every_ms_;
    std::optional<std::string> run_roll_max_bytes_;
};

TEST_F(StartupConfigurationTest, ConfigFileApiPathIsNormalized) {
    const auto temp_root =
        std::filesystem::temp_directory_path() /
        ("gpufl_startup_config_" + std::to_string(gpufl::detail::GetPid()));
    const auto config_path = temp_root / "config.json";
    std::error_code ec;
    std::filesystem::remove_all(temp_root, ec);
    std::filesystem::create_directories(temp_root, ec);
    ASSERT_FALSE(ec) << ec.message();
    {
        std::ofstream config(config_path);
        ASSERT_TRUE(config);
        config << R"({"api_path":"configured/v1/"})";
    }

    gpufl::InitOptions options;
    options.config_file = config_path.string();
    gpufl::detail::resolveStartupOptions(options);
    EXPECT_EQ(options.api_path, "/configured/v1");

    std::filesystem::remove_all(temp_root, ec);
}

TEST_F(StartupConfigurationTest, ApiPathEnvironmentFallbackIsNormalized) {
    setEnv(gpufl::env::kApiPath, "environment/v1/");

    gpufl::InitOptions options;
    gpufl::detail::resolveStartupOptions(options);

    EXPECT_EQ(options.api_path, "/environment/v1");
}

TEST_F(StartupConfigurationTest, ValidSegmentationAndRolloverAreReadTogether) {
    setEnv(gpufl::env::kRunId, "12345678-1234-4123-8123-123456789abc");
    setEnv(gpufl::env::kSegmentEveryMs, "60000");
    setEnv(gpufl::env::kSegmentMaxRows, "2000000");
    setEnv(gpufl::env::kRunRollEveryMs, "180000");
    setEnv(gpufl::env::kRunRollMaxBytes, "4294967296");

    gpufl::detail::StartupSegmentationOptions options;
    std::string error;
    ASSERT_TRUE(gpufl::detail::readStartupSegmentationOptions(options, error))
        << error;
    EXPECT_TRUE(options.enabled());
    EXPECT_EQ(options.run_id, "12345678-1234-4123-8123-123456789abc");
    EXPECT_EQ(options.segment_every_ms, 60000u);
    EXPECT_EQ(options.segment_max_rows, 2000000u);
    EXPECT_EQ(options.run_roll_every_ms, 180000u);
    EXPECT_EQ(options.run_roll_max_bytes, 4294967296u);
}

TEST_F(StartupConfigurationTest, SegmentationWithoutRunIdFailsBeforeRuntimeSetup) {
    setEnv(gpufl::env::kSegmentEveryMs, "60000");

    gpufl::detail::StartupSegmentationOptions options;
    std::string error;
    EXPECT_FALSE(gpufl::detail::readStartupSegmentationOptions(options, error));
    EXPECT_EQ(error,
              "Session segmentation requires GPUFL_RUN_ID. The launcher "
              "must generate one run ID before starting the target.");
}

TEST_F(StartupConfigurationTest, InvalidRolloverValueIsRejected) {
    setEnv(gpufl::env::kRunRollMaxBytes, "not-a-number");

    gpufl::detail::StartupSegmentationOptions options;
    std::string error;
    EXPECT_FALSE(gpufl::detail::readStartupSegmentationOptions(options, error));
    EXPECT_EQ(error,
              "GPUFL_RUN_ROLL_MAX_BYTES='not-a-number' is invalid "
              "(expected a non-negative integer)");
}

}  // namespace

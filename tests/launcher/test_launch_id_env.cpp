#include <gtest/gtest.h>

#include <map>
#include <regex>
#include <string>
#include <vector>

#include "gpufl/core/env_vars.hpp"
#include "trace_command_common.hpp"

namespace {

// Records the environment the target would inherit instead of changing it.
class RecordingLaunchIdPlatform final
    : public gpufl::launcher::TracePlatform {
   public:
    mutable std::map<std::string, std::string> env;

    bool setEnv(const char* key, const std::string& value,
                std::string&) const override {
        env[key] = value;
        return true;
    }
    bool unsetEnv(const char* key, std::string&) const override {
        env.erase(key);
        return true;
    }

    // Not reached: applyLaunchIdEnv only touches the environment.
    const char* platformName() const override { return "recording"; }
    const char* injectLibraryName() const override { return "none"; }
    gpufl::launcher::fs::path selfExe() const override { return {}; }
    std::vector<gpufl::launcher::fs::path> injectLibCandidates(
        const gpufl::launcher::fs::path&) const override { return {}; }
    gpufl::launcher::fs::path defaultOutputDir(
        const std::string&) const override { return {}; }
    std::string defaultAppName(const std::string&) const override { return {}; }
    bool prepareInjectionEnv(const gpufl::launcher::fs::path&,
                             std::string&) const override {
        return true;
    }
    gpufl::launcher::TraceProcessResult runProcess(
        const std::vector<std::string>&,
        const gpufl::launcher::RunOptions&) const override {
        return {};
    }
};

using gpufl::launcher::applyLaunchIdEnv;
namespace env = gpufl::env;

TEST(LaunchIdEnvTest, HarnessIdIsKeptInvalidIsRefusedAbsentIsGenerated) {
    // A harness that exported its own id keeps it, so its records join.
    {
        RecordingLaunchIdPlatform platform;
        ASSERT_TRUE(applyLaunchIdEnv("bench_sweep-42", platform));
        EXPECT_EQ(platform.env[env::kLaunchId], "bench_sweep-42");
    }

    // An invalid one refuses the launch and publishes nothing, including one
    // character past the 64 limit.
    for (const std::string& bad :
         {std::string("has space"), std::string("a/b"),
          std::string(65, 'a')}) {
        RecordingLaunchIdPlatform platform;
        EXPECT_FALSE(applyLaunchIdEnv(bad.c_str(), platform)) << bad;
        EXPECT_EQ(platform.env.count(env::kLaunchId), 0u) << bad;
    }

    // Absent or empty: a fresh UUIDv4 every descendant will inherit.
    for (const char* unset : {static_cast<const char*>(nullptr), ""}) {
        RecordingLaunchIdPlatform platform;
        ASSERT_TRUE(applyLaunchIdEnv(unset, platform));
        EXPECT_TRUE(std::regex_match(
            platform.env[env::kLaunchId],
            std::regex("^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
                       "[89ab][0-9a-f]{3}-[0-9a-f]{12}$")))
            << platform.env[env::kLaunchId];
    }
}

}  // namespace

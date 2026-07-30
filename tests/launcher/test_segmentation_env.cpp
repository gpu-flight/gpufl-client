#include <gtest/gtest.h>

#include <map>
#include <string>
#include <vector>

#include "cli_parse.hpp"
#include "gpufl/core/env_vars.hpp"
#include "trace_command_common.hpp"

namespace {

class RecordingSegmentationPlatform final
    : public gpufl::launcher::TracePlatform {
   public:
    mutable std::map<std::string, std::string> env;
    mutable std::vector<std::string> removed;

    bool has(const char* key) const { return env.count(key) != 0; }
    std::string get(const char* key) const {
        const auto it = env.find(key);
        return it == env.end() ? std::string() : it->second;
    }

    bool setEnv(const char* key, const std::string& value,
                std::string&) const override {
        env[key] = value;
        return true;
    }
    bool unsetEnv(const char* key, std::string&) const override {
        env.erase(key);
        removed.emplace_back(key);
        return true;
    }

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

namespace env = gpufl::env;
using gpufl::launcher::TraceArgs;
using gpufl::launcher::applySegmentationEnv;

TEST(SegmentationEnvTest, TimeAndRowsTravelWithOneRunId) {
    RecordingSegmentationPlatform platform;
    TraceArgs args;
    args.segment_every_ms = 300000;
    args.segment_max_rows = 2'000'000;

    ASSERT_TRUE(applySegmentationEnv(
        args, "12345678-1234-4123-8123-123456789abc", platform));

    EXPECT_EQ(platform.get(env::kRunId),
              "12345678-1234-4123-8123-123456789abc");
    EXPECT_EQ(platform.get(env::kSegmentEveryMs), "300000");
    EXPECT_EQ(platform.get(env::kSegmentMaxRows), "2000000");
}

TEST(SegmentationEnvTest, DisabledTriggerIsRemovedRatherThanPublishedAsZero) {
    RecordingSegmentationPlatform platform;
    platform.env[env::kSegmentEveryMs] = "5000";

    TraceArgs args;
    args.segment_max_rows = 1000;
    ASSERT_TRUE(applySegmentationEnv(
        args, "12345678-1234-4123-8123-123456789abc", platform));

    EXPECT_FALSE(platform.has(env::kSegmentEveryMs));
    EXPECT_EQ(platform.get(env::kSegmentMaxRows), "1000");
}

TEST(SegmentationEnvTest, OrdinaryRunScrubsAllInheritedSegmentationState) {
    RecordingSegmentationPlatform platform;
    platform.env[env::kRunId] = "stale-run";
    platform.env[env::kSegmentEveryMs] = "300000";
    platform.env[env::kSegmentMaxRows] = "1000";

    ASSERT_TRUE(applySegmentationEnv(TraceArgs{}, "", platform));

    EXPECT_FALSE(platform.has(env::kRunId));
    EXPECT_FALSE(platform.has(env::kSegmentEveryMs));
    EXPECT_FALSE(platform.has(env::kSegmentMaxRows));
    EXPECT_EQ(platform.removed.size(), 3u);
}

TEST(SegmentationEnvTest, SegmentedRunWithoutRunIdFailsBeforeTargetLaunch) {
    RecordingSegmentationPlatform platform;
    TraceArgs args;
    args.segment_every_ms = 300000;

    EXPECT_FALSE(applySegmentationEnv(args, "", platform));
    EXPECT_FALSE(platform.has(env::kRunId));
}

}  // namespace

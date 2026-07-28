#include <gtest/gtest.h>

#include <map>
#include <string>
#include <vector>

#include "cli_parse.hpp"
#include "gpufl/core/env_vars.hpp"
#include "trace_command_common.hpp"

using gpufl::launcher::TraceArgs;
using gpufl::launcher::TracePlatform;
using gpufl::launcher::applyDeepWindowEnv;
namespace env = gpufl::env;

namespace {

// Records what the launcher did to the environment instead of doing it. The
// point of the recording is the REMOVALS: a test that only inspects what was
// set cannot tell "we left it alone" from "we took it away", and those two
// differ by whether a trigger the parent shell exported survives.
class RecordingPlatform final : public TracePlatform {
   public:
    /// The environment the target would inherit. Seed it to model a parent
    /// shell that already had a GPUFL variable exported.
    std::map<std::string, std::string> env;
    std::vector<std::string> removed;

    bool has(const char* key) const { return env.count(key) != 0; }
    std::string get(const char* key) const {
        const auto it = env.find(key);
        return it == env.end() ? std::string() : it->second;
    }

    bool setEnv(const char* key, const std::string& value,
                std::string&) const override {
        const_cast<RecordingPlatform*>(this)->env[key] = value;
        return true;
    }
    bool unsetEnv(const char* key, std::string&) const override {
        auto* self = const_cast<RecordingPlatform*>(this);
        self->env.erase(key);
        self->removed.emplace_back(key);
        return true;
    }

    // Not reached: applyDeepWindowEnv only touches the environment.
    const char* platformName() const override { return "recording"; }
    const char* injectLibraryName() const override { return "none"; }
    gpufl::launcher::fs::path selfExe() const override { return {}; }
    std::vector<gpufl::launcher::fs::path> injectLibCandidates(
        const gpufl::launcher::fs::path&) const override { return {}; }
    gpufl::launcher::fs::path defaultOutputDir(
        const std::string&) const override { return {}; }
    std::string defaultAppName(const std::string&) const override { return {}; }
    bool prepareInjectionEnv(const gpufl::launcher::fs::path&,
                             std::string&) const override { return true; }
    gpufl::launcher::TraceProcessResult runProcess(
        const std::vector<std::string>&,
        const gpufl::launcher::RunOptions&) const override { return {}; }
};

TraceArgs conditionalRun() {
    TraceArgs a;
    a.deep_requested = true;
    a.deep_when = "kernel_launch_rate<100 for 500ms";
    a.deep_for_ms = 2000;
    return a;
}

TraceArgs scheduledRun() {
    TraceArgs a;
    a.deep_requested = true;
    a.deep_after_ms = 3000;
    a.deep_after_set = true;
    a.deep_for_ms = 2000;
    return a;
}

// ── trigger ownership ───────────────────────────────────────────────────────
//
// GPUFL_DEEP_AFTER_MS and GPUFL_DEEP_WHEN install their triggers by EXISTING,
// whatever their value. Not setting one is therefore not the same as the
// target not seeing one, and every test here is about that difference.

TEST(DeepWindowEnvTest, AConditionalRunRemovesAnInheritedTimeTrigger) {
    RecordingPlatform p;
    p.env[env::kDeepAfterMs] = "0";   // exported by the parent shell

    ASSERT_TRUE(applyDeepWindowEnv(conditionalRun(), p));

    // Left in place, this opens a window at t=0 that the rule is then refused
    // behind - and the rule reports never_true for a condition that held.
    EXPECT_FALSE(p.has(env::kDeepAfterMs));
    EXPECT_EQ(p.get(env::kDeepWhen), "kernel_launch_rate<100 for 500ms");
}

TEST(DeepWindowEnvTest, AScheduledRunRemovesAnInheritedRule) {
    RecordingPlatform p;
    p.env[env::kDeepWhen] = "custom.token_rate<10 for 1s";

    ASSERT_TRUE(applyDeepWindowEnv(scheduledRun(), p));

    // Otherwise --deep-after silently installs a rule nobody asked for, which
    // then competes with the window the user did ask for.
    EXPECT_FALSE(p.has(env::kDeepWhen));
    EXPECT_EQ(p.get(env::kDeepAfterMs), "3000");
}

TEST(DeepWindowEnvTest, EachModeRemovesTheOtherTriggerEvenWhenUnset) {
    // The removal is unconditional, not "only if we saw one". The launcher
    // cannot see the parent environment of a target it has not spawned yet on
    // every platform, so it does not try to decide.
    RecordingPlatform cond;
    ASSERT_TRUE(applyDeepWindowEnv(conditionalRun(), cond));
    EXPECT_NE(std::find(cond.removed.begin(), cond.removed.end(),
                        env::kDeepAfterMs), cond.removed.end());

    RecordingPlatform sched;
    ASSERT_TRUE(applyDeepWindowEnv(scheduledRun(), sched));
    EXPECT_NE(std::find(sched.removed.begin(), sched.removed.end(),
                        env::kDeepWhen), sched.removed.end());
}

TEST(DeepWindowEnvTest, ARunWithNoDeepFlagsTouchesNeitherTrigger) {
    // `--passes PcSampling` with GPUFL_DEEP_WHEN exported is the supported way
    // to reach an engine the adaptive plan does not select yet. Scrubbing here
    // would take that away.
    RecordingPlatform p;
    p.env[env::kDeepWhen] = "custom.token_rate<10 for 1s";
    p.env[env::kDeepAfterMs] = "5000";

    TraceArgs plain;   // deep_requested stays false
    ASSERT_TRUE(applyDeepWindowEnv(plain, p));

    EXPECT_TRUE(p.removed.empty());
    EXPECT_EQ(p.get(env::kDeepWhen), "custom.token_rate<10 for 1s");
    EXPECT_EQ(p.get(env::kDeepAfterMs), "5000");
    EXPECT_FALSE(p.has(env::kDeepArm)) << "window-only arming was not asked for";
}

TEST(DeepWindowEnvTest, TheWindowBoundsAndArmModeStillTravel) {
    RecordingPlatform p;
    TraceArgs a = conditionalRun();
    a.deep_launches = 500;
    a.deep_cooldown_ms = 1500;

    ASSERT_TRUE(applyDeepWindowEnv(a, p));

    EXPECT_EQ(p.get(env::kDeepArm), "window");
    EXPECT_EQ(p.get(env::kDeepWindowMs), "2000");
    EXPECT_EQ(p.get(env::kDeepWindowMaxLaunches), "500");
    EXPECT_EQ(p.get(env::kDeepWindowCooldownMs), "1500");
}

TEST(DeepWindowEnvTest, AnUnsetBoundIsNotPublishedAsZero) {
    // 0 means "no bound" to the client, and publishing it would override a
    // bound the parent environment had legitimately set.
    RecordingPlatform p;
    ASSERT_TRUE(applyDeepWindowEnv(conditionalRun(), p));

    EXPECT_FALSE(p.has(env::kDeepWindowMaxLaunches));
    EXPECT_FALSE(p.has(env::kDeepWindowCooldownMs));
}

}  // namespace

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "gpufl/core/deep_window_rules.hpp"
#include "gpufl/core/env_vars.hpp"

using gpufl::detail::DeepWindowRules;

namespace {

// Env is process-global, so every test here sets and clears the whole group.
// Leaving one behind would silently change the next test's rule.
void setEnv(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value ? value : "");
#else
    if (value) ::setenv(name, value, 1); else ::unsetenv(name);
#endif
}

class RuleInstallTest : public ::testing::Test {
   protected:
    void SetUp() override { clearAll(); DeepWindowRules::ResetForTesting(); }
    void TearDown() override { clearAll(); DeepWindowRules::ResetForTesting(); }

    static void clearAll() {
        for (const char* n : {gpufl::env::kDeepWhen, gpufl::env::kDeepRateWindowMs,
                              gpufl::env::kDeepStaleAfterMs, gpufl::env::kDeepRearmAt,
                              gpufl::env::kDeepMaxWindows, gpufl::env::kDeepWindowMs,
                              gpufl::env::kDeepWindowMaxLaunches}) {
            setEnv(n, nullptr);
        }
    }
};

TEST_F(RuleInstallTest, NoRuleWithoutTheEnvVar) {
    DeepWindowRules::InstallFromEnv();
    EXPECT_FALSE(DeepWindowRules::Installed());
    EXPECT_FALSE(DeepWindowRules::WantsLaunchFeed());
}

TEST_F(RuleInstallTest, AValidRuleInstallsAndAsksForTheLaunchFeed) {
    setEnv(gpufl::env::kDeepWhen, "kernel_launch_rate<100 for 2s");
    setEnv(gpufl::env::kDeepWindowMs, "500");
    DeepWindowRules::InstallFromEnv();

    EXPECT_TRUE(DeepWindowRules::Installed());
    // The feed costs an atomic per launch, so only a rule that reads it should
    // switch it on.
    EXPECT_TRUE(DeepWindowRules::WantsLaunchFeed());
}

TEST_F(RuleInstallTest, ACustomCounterRuleDoesNotPayForTheLaunchFeed) {
    setEnv(gpufl::env::kDeepWhen, "custom.token_rate<100 for 2s");
    setEnv(gpufl::env::kDeepWindowMs, "500");
    DeepWindowRules::InstallFromEnv();

    EXPECT_TRUE(DeepWindowRules::Installed());
    EXPECT_FALSE(DeepWindowRules::WantsLaunchFeed())
        << "every launch would pay for a feed this rule never reads";
}

TEST_F(RuleInstallTest, AnInvalidRuleIsInstalledAsRefusedRatherThanIgnored) {
    setEnv(gpufl::env::kDeepWhen, "tokne_rate<100 for 2s");
    setEnv(gpufl::env::kDeepWindowMs, "500");
    DeepWindowRules::InstallFromEnv();

    // Installed, so Finish() still has something to report. A rejected rule
    // that vanishes looks exactly like one that was simply never true.
    EXPECT_TRUE(DeepWindowRules::Installed());
    EXPECT_FALSE(DeepWindowRules::WantsLaunchFeed());
}

TEST_F(RuleInstallTest, AWindowWithNoBoundIsRefused) {
    // Neither GPUFL_DEEP_WINDOW_MS nor MAX_LAUNCHES: the window would never
    // close, which turns a bounded-cost feature into an always-on one.
    setEnv(gpufl::env::kDeepWhen, "kernel_launch_rate<100 for 2s");
    DeepWindowRules::InstallFromEnv();
    EXPECT_TRUE(DeepWindowRules::Installed());
    EXPECT_FALSE(DeepWindowRules::WantsLaunchFeed());
}

TEST_F(RuleInstallTest, TheDefaultStaleAfterIsOneThatCanActuallyFire) {
    // Left unset, stale-after must be derived from the other two rather than
    // defaulted to a constant the validator would then reject.
    setEnv(gpufl::env::kDeepWhen, "kernel_launch_rate<100 for 30s");
    setEnv(gpufl::env::kDeepRateWindowMs, "5000");
    setEnv(gpufl::env::kDeepWindowMs, "500");
    DeepWindowRules::InstallFromEnv();

    EXPECT_TRUE(DeepWindowRules::Installed());
    EXPECT_TRUE(DeepWindowRules::WantsLaunchFeed())
        << "a workable rule was refused by its own default stale-after";
}

TEST_F(RuleInstallTest, ServiceAndFinishAreSafeWithoutARule) {
    // Both run unconditionally from the collector and from shutdown.
    DeepWindowRules::Service();
    DeepWindowRules::Finish();
    DeepWindowRules::NoteKernelLaunch(1);
    SUCCEED();
}

TEST_F(RuleInstallTest, ASecondRuleIsRefusedRatherThanSilentlyReplacing) {
    setEnv(gpufl::env::kDeepWhen, "kernel_launch_rate<100 for 2s");
    setEnv(gpufl::env::kDeepWindowMs, "500");
    DeepWindowRules::InstallFromEnv();
    ASSERT_TRUE(DeepWindowRules::WantsLaunchFeed());

    setEnv(gpufl::env::kDeepWhen, "custom.token_rate<5 for 2s");
    DeepWindowRules::InstallFromEnv();
    // The first rule stands. Replacing it silently would make which rule ran
    // depend on call order.
    EXPECT_TRUE(DeepWindowRules::WantsLaunchFeed());
}

}  // namespace

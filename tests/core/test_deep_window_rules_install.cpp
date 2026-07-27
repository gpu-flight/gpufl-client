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

TEST_F(RuleInstallTest, AMalformedNumericOptionRefusesRatherThanDefaulting) {
    setEnv(gpufl::env::kDeepWhen, "kernel_launch_rate<100 for 2s");
    setEnv(gpufl::env::kDeepWindowMs, "500");
    setEnv(gpufl::env::kDeepMaxWindows, "three");
    DeepWindowRules::InstallFromEnv();

    // Silently substituting the default would open real windows under a budget
    // the user never chose and cannot see.
    EXPECT_TRUE(DeepWindowRules::Installed());
    EXPECT_FALSE(DeepWindowRules::WantsLaunchFeed())
        << "a typo'd option was quietly replaced by a default";
}

TEST_F(RuleInstallTest, ARuleCanBeInstalledAgainAfterFinish) {
    setEnv(gpufl::env::kDeepWhen, "kernel_launch_rate<100 for 2s");
    setEnv(gpufl::env::kDeepWindowMs, "500");
    DeepWindowRules::InstallFromEnv();
    ASSERT_TRUE(DeepWindowRules::Installed());

    // An embedded host may shutdown() and init() again in one process. Without
    // releasing the session, the second run's rule is refused as a duplicate
    // and that run silently has no trigger at all.
    DeepWindowRules::Finish();
    DeepWindowRules::InstallFromEnv();
    EXPECT_TRUE(DeepWindowRules::WantsLaunchFeed())
        << "the second session was left with no rule";
}

TEST_F(RuleInstallTest, AnOutOfRangeNumericOptionIsRefused) {
    // Pins the OUTCOME, not the mechanism. Two independent guards reject this:
    // the ERANGE check on the parse, and the range validator downstream. Either
    // alone suffices, so removing one does not fail this test - what it fixes
    // is which reason gets reported, and that is not observable from here.
    // The reason the ERANGE check still exists is the arithmetic in between:
    // a saturated LLONG_MAX fed into the derived stale-after sum is signed
    // overflow, which is undefined rather than merely wrong.
    setEnv(gpufl::env::kDeepWhen, "kernel_launch_rate<100 for 2s");
    setEnv(gpufl::env::kDeepWindowMs, "500");
    setEnv(gpufl::env::kDeepRateWindowMs, "99999999999999999999999999");
    DeepWindowRules::InstallFromEnv();

    EXPECT_TRUE(DeepWindowRules::Installed());
    EXPECT_FALSE(DeepWindowRules::WantsLaunchFeed())
        << "a value too large to represent was accepted";
}

TEST_F(RuleInstallTest, AMaxWindowsThatDoesNotFitAnIntIsRefused) {
    // 4294967297 survives ERANGE - it fits an int64 - and then narrows to 1,
    // which the validator accepts. The run would silently use a budget nobody
    // configured.
    setEnv(gpufl::env::kDeepWhen, "kernel_launch_rate<100 for 2s");
    setEnv(gpufl::env::kDeepWindowMs, "500");
    setEnv(gpufl::env::kDeepMaxWindows, "4294967297");
    DeepWindowRules::InstallFromEnv();

    EXPECT_TRUE(DeepWindowRules::Installed());
    EXPECT_FALSE(DeepWindowRules::WantsLaunchFeed())
        << "a value that cannot fit an int was narrowed into a valid one";
}

}  // namespace

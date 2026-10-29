#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "gpufl/core/common.hpp"
#include "gpufl/core/env_vars.hpp"
#include "trace_command_common.hpp"
#include "trace_run_plan.hpp"

namespace {

class PlanningPlatform : public gpufl::launcher::TracePlatform {
public:
    const char* platformName() const override { return "planning"; }
    const char* injectLibraryName() const override { return "inject"; }
    gpufl::launcher::fs::path selfExe() const override { return {}; }
    std::vector<gpufl::launcher::fs::path> injectLibCandidates(
        const gpufl::launcher::fs::path&) const override { return {}; }
    gpufl::launcher::fs::path defaultOutputDir(
        const std::string& tag) const override {
        return gpufl::launcher::fs::path("captures") / tag;
    }
    std::string defaultAppName(const std::string&) const override {
        return "inferred-app";
    }
    bool setEnv(const char*, const std::string&, std::string&) const override {
        return true;
    }
    bool unsetEnv(const char*, std::string&) const override { return true; }
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

using gpufl::launcher::TraceArgs;
using gpufl::launcher::createTraceRunPlan;

TEST(TraceRunPlanTest, SinglePassUsesOneGeneratedDirectoryTag) {
    PlanningPlatform platform;
    TraceArgs args;
    args.command = {"target"};

    const auto plan = createTraceRunPlan(args, platform);

    ASSERT_EQ(plan.passes, std::vector<std::string>({"Trace"}));
    EXPECT_FALSE(plan.multipass);
    EXPECT_FALSE(plan.segmented);
    EXPECT_TRUE(plan.analysis_id.empty());
    EXPECT_TRUE(plan.run_id.empty());
    EXPECT_EQ(plan.app_name, "inferred-app");
    EXPECT_EQ(plan.output_dir, gpufl::launcher::fs::path("captures") /
                                   plan.directory_tag);
    EXPECT_EQ(plan.run_options.run_ms, 0);
}

TEST(TraceRunPlanTest, MultipassNestsTheExplicitOutputUnderTheAnalysisFolder) {
    PlanningPlatform platform;
    TraceArgs args;
    args.command = {"target"};
    args.name = "candidate";
    args.output_dir = "requested-output";
    args.passes = {"Trace", "PcSampling"};

    const auto plan = createTraceRunPlan(args, platform);

    ASSERT_EQ(plan.passes.size(), 2u);
    EXPECT_TRUE(plan.multipass);
    EXPECT_FALSE(plan.segmented);
    EXPECT_FALSE(plan.analysis_id.empty());
    EXPECT_EQ(plan.directory_tag, plan.analysis_id);
    EXPECT_EQ(plan.output_dir.parent_path(),
              gpufl::launcher::fs::path("requested-output"));
    EXPECT_EQ(plan.output_dir.filename().string(),
              "run-candidate-" + plan.analysis_id.substr(0, 8));
}

TEST(TraceRunPlanTest, SegmentedRunUsesItsRunIdAndAppliesTheTightestWindowCap) {
    PlanningPlatform platform;
    TraceArgs args;
    args.command = {"target"};
    args.segment_every_ms = 60'000;
    args.warmup_ms = 1'500;
    args.window_ms = 1'000;
    args.window_timeout_ms = 2'200;

    const auto plan = createTraceRunPlan(args, platform);

    EXPECT_FALSE(plan.multipass);
    EXPECT_TRUE(plan.segmented);
    EXPECT_TRUE(plan.analysis_id.empty());
    EXPECT_FALSE(plan.run_id.empty());
    EXPECT_EQ(plan.directory_tag, plan.run_id);
    EXPECT_EQ(plan.output_dir, gpufl::launcher::fs::path("captures") /
                                   plan.run_id);
    EXPECT_EQ(plan.run_options.run_ms, 2'200);
}

class ExecutingPlanningPlatform final : public PlanningPlatform {
public:
    explicit ExecutingPlanningPlatform(gpufl::launcher::fs::path root)
        : root_(std::move(root)) {}

    gpufl::launcher::fs::path selfExe() const override {
        return root_ / "gpufl";
    }
    std::vector<gpufl::launcher::fs::path> injectLibCandidates(
        const gpufl::launcher::fs::path&) const override {
        return {root_ / "gpufl_inject.so"};
    }
    gpufl::launcher::fs::path defaultOutputDir(
        const std::string& tag) const override {
        return root_ / "captures" / tag;
    }
    bool setEnv(const char* key, const std::string& value,
                std::string&) const override {
        env[key] = value;
        return true;
    }
    bool unsetEnv(const char* key, std::string&) const override {
        env.erase(key);
        return true;
    }
    bool prepareInjectionEnv(const gpufl::launcher::fs::path&,
                             std::string&) const override {
        return true;
    }
    gpufl::launcher::TraceProcessResult runProcess(
        const std::vector<std::string>& command,
        const gpufl::launcher::RunOptions& options) const override {
        seen_command = command;
        seen_options = options;
        env_per_run.push_back(env);
        gpufl::launcher::TraceProcessResult result;
        result.rc = 0;
        return result;
    }

    mutable std::map<std::string, std::string> env;
    mutable std::vector<std::string> seen_command;
    mutable gpufl::launcher::RunOptions seen_options;
    // The environment each child started with, in pass order.
    mutable std::vector<std::map<std::string, std::string>> env_per_run;

private:
    gpufl::launcher::fs::path root_;
};

TEST(TraceRunPlanTest, TraceCommonExecutesThePlannedSinglePass) {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() /
        ("gpufl_trace_plan_" + std::to_string(gpufl::detail::GetPid()));
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root, ec);
    ASSERT_FALSE(ec) << ec.message();
    { std::ofstream inject(root / "gpufl_inject.so"); ASSERT_TRUE(inject); }

    ExecutingPlanningPlatform platform(root);
    // Inherited from the parent shell; without --stacks it must not survive.
    platform.env[gpufl::env::kStackTrace] = "0";
    TraceArgs args;
    args.command = {"target", "--work"};
    args.name = "trace-plan-test";
    args.output_dir = (root / "requested-output").string();
    args.source_root = root.string();
    args.warmup_ms = 100;
    args.window_ms = 200;

    EXPECT_EQ(gpufl::launcher::runTraceCommon(args, platform), 0);
    EXPECT_EQ(platform.seen_command, args.command);
    EXPECT_EQ(platform.seen_options.run_ms, 300);
    EXPECT_EQ(platform.env[gpufl::env::kAppName], "trace-plan-test");
    EXPECT_EQ(platform.env[gpufl::env::kLogDir], args.output_dir);
    EXPECT_EQ(platform.env[gpufl::env::kProfilingEngine], "Trace");
    EXPECT_EQ(platform.env[gpufl::env::kIncludeSource], "1");
    EXPECT_EQ(platform.env.count(gpufl::env::kStackTrace), 0u);
    EXPECT_EQ(fs::weakly_canonical(platform.env[gpufl::env::kSourceRoot]),
              fs::weakly_canonical(root));

    fs::remove_all(root, ec);
}

TEST(TraceRunPlanTest, TraceCommonHonorsExplicitCaptureOptOuts) {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() /
        ("gpufl_trace_source_privacy_" +
         std::to_string(gpufl::detail::GetPid()));
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root, ec);
    ASSERT_FALSE(ec) << ec.message();
    { std::ofstream inject(root / "gpufl_inject.so"); ASSERT_TRUE(inject); }

    ExecutingPlanningPlatform platform(root);
    platform.env[gpufl::env::kIncludeSource] = "1";
    TraceArgs args;
    args.command = {"target"};
    args.output_dir = (root / "capture").string();
    args.source_root = root.string();
    args.no_source = true;
    args.stack_traces = false;

    EXPECT_EQ(gpufl::launcher::runTraceCommon(args, platform), 0);
    EXPECT_EQ(platform.env[gpufl::env::kIncludeSource], "0");
    EXPECT_EQ(platform.env[gpufl::env::kStackTrace], "0");
    EXPECT_EQ(fs::weakly_canonical(platform.env[gpufl::env::kSourceRoot]),
              fs::weakly_canonical(root));

    fs::remove_all(root, ec);
}

TEST(TraceRunPlanTest, PassRunsPcSamplingOnlyWhereTheAmdBackendSamples) {
    using gpufl::launcher::passRunsPcSampling;
    EXPECT_TRUE(passRunsPcSampling("PcSampling"));
    EXPECT_TRUE(passRunsPcSampling("Trace+PcSampling"));
    EXPECT_TRUE(passRunsPcSampling("PcSampling+Trace"));
    EXPECT_TRUE(passRunsPcSampling("Trace+ PcSampling"));
    // The AMD backend runs one service beside tracing: a group with another
    // engine keeps its first one.
    EXPECT_TRUE(passRunsPcSampling("PcSampling+PmSampling"));
    EXPECT_FALSE(passRunsPcSampling("PmSampling+PcSampling"));
    EXPECT_FALSE(passRunsPcSampling("Trace+PcSampling+PmSampling"));
    EXPECT_FALSE(passRunsPcSampling("Trace"));
    EXPECT_FALSE(passRunsPcSampling("Deep"));
    EXPECT_FALSE(passRunsPcSampling("SassMetrics"));
    EXPECT_FALSE(passRunsPcSampling("PcSamplingX"));
}

// The beta switch reaches the PcSampling children only, and only with
// --amd-pc-sampling=stochastic; every other child loses an inherited one.
TEST(TraceRunPlanTest, AmdPcSamplingOptInReachesOnlyPcSamplingChildren) {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() /
        ("gpufl_trace_amd_pcs_" + std::to_string(gpufl::detail::GetPid()));
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root, ec);
    ASSERT_FALSE(ec) << ec.message();
    { std::ofstream inject(root / "gpufl_inject.so"); ASSERT_TRUE(inject); }

    const auto run = [&](const bool opted_in) {
        ExecutingPlanningPlatform platform(root);
        platform.env[gpufl::env::kRocprofilerPcSamplingBetaEnabled] = "1";
        platform.env[gpufl::env::kAmdPcSampling] = "stochastic";
        TraceArgs args;
        args.command = {"target"};
        args.quiet = true;
        args.output_dir = (root / (opted_in ? "in" : "out")).string();
        args.source_root = root.string();
        args.passes = {"Trace", "PcSampling", "Trace+PcSampling", "Deep",
                       "SassMetrics", "PmSampling+PcSampling"};
        args.amd_pc_sampling = opted_in;
        EXPECT_EQ(gpufl::launcher::runTraceCommon(args, platform), 0);
        return platform.env_per_run;
    };

    const auto beta = [](const std::map<std::string, std::string>& env) {
        const auto it = env.find(gpufl::env::kRocprofilerPcSamplingBetaEnabled);
        return it == env.end() ? std::string("unset") : it->second;
    };
    const auto opt_in = [](const std::map<std::string, std::string>& env) {
        const auto it = env.find(gpufl::env::kAmdPcSampling);
        return it == env.end() ? std::string("unset") : it->second;
    };

    // PmSampling+PcSampling runs PmSampling on AMD, so it gets neither.
    const auto opted = run(true);
    ASSERT_EQ(opted.size(), 6u);
    const std::vector<std::string> expected_beta = {"unset", "ON", "ON",
                                                    "unset", "unset", "unset"};
    const std::vector<std::string> expected_opt_in = {
        "unset", "stochastic", "stochastic", "unset", "unset", "unset"};
    for (size_t pass = 0; pass < opted.size(); ++pass) {
        EXPECT_EQ(beta(opted[pass]), expected_beta[pass]) << pass;
        EXPECT_EQ(opt_in(opted[pass]), expected_opt_in[pass]) << pass;
    }

    for (const auto& env : run(false)) {
        EXPECT_EQ(beta(env), "unset");
        EXPECT_EQ(opt_in(env), "unset");
    }

    fs::remove_all(root, ec);
}

TEST(TraceRunPlanTest, AmdPcSamplingWarnsOfAGroupThatWontSample) {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() /
        ("gpufl_trace_amd_pcs_warn_" +
         std::to_string(gpufl::detail::GetPid()));
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root, ec);
    ASSERT_FALSE(ec) << ec.message();
    { std::ofstream inject(root / "gpufl_inject.so"); ASSERT_TRUE(inject); }

    ExecutingPlanningPlatform platform(root);
    TraceArgs args;
    args.command = {"target"};
    args.output_dir = (root / "out").string();
    args.source_root = root.string();
    args.passes = {"PmSampling+PcSampling"};
    args.amd_pc_sampling = true;
    testing::internal::CaptureStderr();
    EXPECT_EQ(gpufl::launcher::runTraceCommon(args, platform), 0);
    const std::string err = testing::internal::GetCapturedStderr();

    EXPECT_NE(err.find("[gpufl] warning: AMD GPUs sample PCs alone or beside "
                       "Trace only, so pass PmSampling+PcSampling runs its "
                       "first engine there without PC sampling"),
              std::string::npos)
        << err;
    EXPECT_NE(err.find("--amd-pc-sampling applies to no pass"),
              std::string::npos)
        << err;
    ASSERT_EQ(platform.env_per_run.size(), 1u);
    EXPECT_EQ(platform.env_per_run[0].count(
                  gpufl::env::kRocprofilerPcSamplingBetaEnabled),
              0u);

    fs::remove_all(root, ec);
}

}  // namespace

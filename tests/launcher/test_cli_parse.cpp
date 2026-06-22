// Tests for the launcher CLI parser. CPU-only - runs on any platform
// where gpufl_tests builds (Linux/macOS/Windows). The launcher binary
// itself is Linux-gated, but the parser source is portable.

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "cli_parse.hpp"

using namespace gpufl::launcher;

namespace {

std::vector<std::string> argsFor(std::initializer_list<const char*> tokens) {
    return {tokens.begin(), tokens.end()};
}

}  // namespace

TEST(CliParseTrace, BasicCommand) {
    auto r = parseTraceArgs(argsFor({"--", "python", "train.py"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    EXPECT_EQ(r.args->command.size(), 2u);
    EXPECT_EQ(r.args->command[0], "python");
    EXPECT_EQ(r.args->command[1], "train.py");
    EXPECT_FALSE(r.args->verbose);
    EXPECT_FALSE(r.args->quiet);
}

TEST(CliParseTrace, MissingDashDash) {
    auto r = parseTraceArgs(argsFor({"python", "train.py"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("`--`"), std::string::npos);
}

TEST(CliParseTrace, NoCommandAfterDashDash) {
    auto r = parseTraceArgs(argsFor({"--"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("no command"), std::string::npos);
}

TEST(CliParseTrace, NameLongFlagWithEquals) {
    auto r = parseTraceArgs(argsFor({"--name=experiment-12", "--", "./bin"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    EXPECT_EQ(r.args->name, "experiment-12");
}

TEST(CliParseTrace, NameLongFlagWithSpace) {
    auto r = parseTraceArgs(argsFor({"--name", "experiment-12", "--", "./bin"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    EXPECT_EQ(r.args->name, "experiment-12");
}

TEST(CliParseTrace, NameShortFlag) {
    auto r = parseTraceArgs(argsFor({"-n", "exp", "--", "./bin"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    EXPECT_EQ(r.args->name, "exp");
}

TEST(CliParseTrace, VerboseAndQuiet) {
    auto r = parseTraceArgs(argsFor({"-v", "-q", "--", "./bin"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    EXPECT_TRUE(r.args->verbose);
    EXPECT_TRUE(r.args->quiet);
}

TEST(CliParseTrace, ProfileFlagRejectedWithMigrationHint) {
    auto r = parseTraceArgs(argsFor({"--profile=light", "--", "./bin"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("--passes=Trace"), std::string::npos);
}

TEST(CliParseTrace, ProfileFlagSpaceFormRejectedWithMigrationHint) {
    auto r = parseTraceArgs(argsFor({"--profile=monitoring-only", "--", "./bin"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("gpufl monitor"), std::string::npos);
}

TEST(CliParseTrace, UploadFlag) {
    auto r = parseTraceArgs(argsFor({"--upload", "--", "./bin"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    EXPECT_TRUE(r.args->upload);
    EXPECT_EQ(r.args->api_version, "v1");
    EXPECT_EQ(r.args->log_types, "device,scope,system,sass");
    EXPECT_EQ(r.args->agent_drain_ms, 60000);
}

TEST(CliParseTrace, UploadDefaultsFalse) {
    auto r = parseTraceArgs(argsFor({"--", "./bin"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    EXPECT_FALSE(r.args->upload);
}

TEST(CliParseTrace, UploadAgentFlags) {
    auto r = parseTraceArgs(argsFor({
        "--upload",
        "--backend-url=https://api.example.com",
        "--api-key", "gpfl_key",
        "--api-version=v2",
        "--agent-jar=/tmp/agent.jar",
        "--agent-cursor=/tmp/trace-cursor.json",
        "--log-types=device,scope",
        "--agent-drain-ms=500",
        "--", "./bin"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    EXPECT_TRUE(r.args->upload);
    EXPECT_EQ(r.args->backend_url, "https://api.example.com");
    EXPECT_EQ(r.args->api_key, "gpfl_key");
    EXPECT_EQ(r.args->api_version, "v2");
    EXPECT_EQ(r.args->agent_jar, "/tmp/agent.jar");
    EXPECT_EQ(r.args->agent_cursor, "/tmp/trace-cursor.json");
    EXPECT_EQ(r.args->log_types, "device,scope");
    EXPECT_EQ(r.args->agent_drain_ms, 500);
}

TEST(CliParseTrace, InvalidAgentDrainRejected) {
    auto r = parseTraceArgs(argsFor({"--agent-drain-ms=-1", "--", "./bin"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("invalid --agent-drain-ms"), std::string::npos);
}

// ── gpufl trace bounded window (--warmup / --window / ...) ───────────────────

TEST(CliParseTrace, WindowFlagsDefaultUnset) {
    auto r = parseTraceArgs(argsFor({"--", "./bin"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    EXPECT_EQ(r.args->warmup_ms, 0);
    EXPECT_EQ(r.args->window_ms, 0);
    EXPECT_EQ(r.args->window_timeout_ms, 0);
    EXPECT_EQ(r.args->after_window, "stop");
}

TEST(CliParseTrace, WarmupAndWindowDurations) {
    auto r = parseTraceArgs(
        argsFor({"--warmup=60s", "--window=5m", "--", "./serve"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    EXPECT_EQ(r.args->warmup_ms, 60000);
    EXPECT_EQ(r.args->window_ms, 300000);
}

TEST(CliParseTrace, DurationUnitsMsSecondsBareHours) {
    auto r = parseTraceArgs(argsFor({
        "--warmup", "500ms", "--window", "30", "--window-timeout=1h",
        "--", "./serve"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    EXPECT_EQ(r.args->warmup_ms, 500);       // explicit ms
    EXPECT_EQ(r.args->window_ms, 30000);     // bare number == seconds
    EXPECT_EQ(r.args->window_timeout_ms, 3600000);
}

TEST(CliParseTrace, InvalidWindowDurationRejected) {
    auto r = parseTraceArgs(argsFor({"--window=abc", "--", "./bin"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("invalid --window"), std::string::npos);
}

TEST(CliParseTrace, NegativeWindowDurationRejected) {
    auto r = parseTraceArgs(argsFor({"--warmup=-5s", "--", "./bin"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("invalid --warmup"), std::string::npos);
}

TEST(CliParseTrace, AfterWindowStopAccepted) {
    auto r = parseTraceArgs(argsFor({"--after-window=stop", "--", "./bin"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    EXPECT_EQ(r.args->after_window, "stop");
}

TEST(CliParseTrace, AfterWindowKeepRejectedAsUnimplemented) {
    auto r = parseTraceArgs(argsFor({"--after-window=keep", "--", "./bin"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("not yet implemented"), std::string::npos);
}

TEST(CliParseTrace, AfterWindowBogusRejected) {
    auto r = parseTraceArgs(argsFor({"--after-window=spin", "--", "./bin"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("invalid --after-window"), std::string::npos);
}

TEST(CliParseTrace, EngineFlagRejectedWithMigrationHint) {
    auto r = parseTraceArgs(argsFor({"--engine=Deep", "--", "./bin"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("--passes=Deep"), std::string::npos);
}

TEST(CliParseTrace, EngineFlagSpaceFormRejectedWithMigrationHint) {
    auto r = parseTraceArgs(argsFor({"--engine", "Monitor", "--", "./bin"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("--passes=Trace"), std::string::npos);
}

TEST(CliParseTrace, UnknownFlag) {
    auto r = parseTraceArgs(argsFor({"--definitely-not-a-flag", "--", "./bin"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("unknown flag"), std::string::npos);
}

TEST(CliParseTrace, MissingValueForFlag) {
    auto r = parseTraceArgs(argsFor({"--name"}));
    EXPECT_FALSE(r.args.has_value());
    // Either "missing value for --name" or "missing `--`" is acceptable;
    // implementation reports missing-value first when the flag is the
    // last token.
    EXPECT_FALSE(r.error.empty());
}

TEST(CliParseTrace, HelpFlag) {
    auto r = parseTraceArgs(argsFor({"--help"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_EQ(r.error, "__help__");
}

TEST(CliParseTrace, MultiTokenCommandPreservesOrder) {
    auto r = parseTraceArgs(argsFor({
        "--", "python", "-m", "torch.distributed.run",
        "--nproc_per_node=2", "train.py"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    ASSERT_EQ(r.args->command.size(), 5u);
    EXPECT_EQ(r.args->command[0], "python");
    EXPECT_EQ(r.args->command[1], "-m");
    EXPECT_EQ(r.args->command[2], "torch.distributed.run");
    EXPECT_EQ(r.args->command[3], "--nproc_per_node=2");
    EXPECT_EQ(r.args->command[4], "train.py");
}

TEST(CliParseTrace, FlagsAfterDashDashTreatedAsCommandArgs) {
    // A flag-looking token AFTER `--` is part of the command, not a
    // launcher flag. This is what makes `gpufl trace -- ./app --verbose`
    // pass --verbose to the target.
    auto r = parseTraceArgs(argsFor({"--", "./app", "--verbose"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    EXPECT_EQ(r.args->command.size(), 2u);
    EXPECT_EQ(r.args->command[1], "--verbose");
}

// ── gpufl trace --passes / multi-pass plan resolution ────────────────────

TEST(CliParseTrace, PassesParsedAsList) {
    auto r = parseTraceArgs(
        argsFor({"--passes", "Trace,PcSampling,SassMetrics", "--", "./bin"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    ASSERT_EQ(r.args->passes.size(), 3u);
    EXPECT_EQ(r.args->passes[0], "Trace");
    EXPECT_EQ(r.args->passes[1], "PcSampling");
    EXPECT_EQ(r.args->passes[2], "SassMetrics");
}

TEST(CliParseTrace, PassesTrimsWhitespace) {
    // Spaces after commas (a natural way to type the list) are tolerated.
    auto r = parseTraceArgs(
        argsFor({"--passes=Trace, SassMetrics", "--", "./bin"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    ASSERT_EQ(r.args->passes.size(), 2u);
    EXPECT_EQ(r.args->passes[0], "Trace");
    EXPECT_EQ(r.args->passes[1], "SassMetrics");
}

TEST(CliParseTrace, PassesDeepPresetParsed) {
    auto r = parseTraceArgs(argsFor({"--passes=Deep", "--", "./bin"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    ASSERT_EQ(r.args->passes.size(), 1u);
    EXPECT_EQ(r.args->passes[0], "Deep");
}

TEST(CliParseTrace, PassesInvalidEngineRejected) {
    auto r = parseTraceArgs(argsFor({"--passes=Trace,warpdrive", "--", "./bin"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("invalid --passes"), std::string::npos);
}

TEST(CliParseTrace, MonitorPassRejected) {
    auto r = parseTraceArgs(argsFor({"--passes=Monitor", "--", "./bin"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("invalid --passes"), std::string::npos);
}

TEST(CliParseTrace, PassesEmptyRejected) {
    auto r = parseTraceArgs(argsFor({"--passes=", "--", "./bin"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("--passes requires"), std::string::npos);
}

TEST(CliParseTrace, DeepPresetCannotBeCombined) {
    auto r = parseTraceArgs(argsFor({"--passes=Trace,Deep", "--", "./bin"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("cannot be combined"), std::string::npos);
}

// ── gpufl trace --passes composite ('+') groups ─────────────────────────────

TEST(CliParseTrace, CompositeGroupParsed) {
    auto r = parseTraceArgs(argsFor({"--passes=Trace+PcSampling", "--", "./bin"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    ASSERT_EQ(r.args->passes.size(), 1u);
    EXPECT_EQ(r.args->passes[0], "Trace+PcSampling");
}

TEST(CliParseTrace, CompositeGroupPlusSeparatePass) {
    auto r = parseTraceArgs(
        argsFor({"--passes=Trace+PcSampling,SassMetrics", "--", "./bin"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    ASSERT_EQ(r.args->passes.size(), 2u);
    EXPECT_EQ(r.args->passes[0], "Trace+PcSampling");
    EXPECT_EQ(r.args->passes[1], "SassMetrics");
}

TEST(CliParseTrace, CompositeDeepInGroupRejected) {
    auto r = parseTraceArgs(argsFor({"--passes=Trace+Deep", "--", "./bin"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("Deep cannot be combined"), std::string::npos);
}

TEST(CliParseTrace, CompositeSassInGroupRejected) {
    auto r = parseTraceArgs(argsFor({"--passes=Trace+SassMetrics", "--", "./bin"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("its own pass"), std::string::npos);
}

TEST(CliParseTrace, CompositeEmptyEngineRejected) {
    auto r = parseTraceArgs(argsFor({"--passes=Trace+", "--", "./bin"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("empty engine"), std::string::npos);
}

TEST(CliParseTrace, CompositeInvalidEngineRejected) {
    auto r = parseTraceArgs(argsFor({"--passes=Trace+warpdrive", "--", "./bin"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("invalid --passes"), std::string::npos);
}

TEST(ResolvePassPlan, CompositeTokenPassesThrough) {
    TraceArgs a;
    a.passes = {"Trace+PcSampling"};
    const auto plan = resolvePassPlan(a);
    ASSERT_EQ(plan.size(), 1u);
    EXPECT_EQ(plan[0], "Trace+PcSampling");
}

TEST(ResolvePassPlan, ExplicitPassesWin) {
    TraceArgs a;
    a.passes = {"Trace", "SassMetrics"};
    const auto plan = resolvePassPlan(a);
    ASSERT_EQ(plan.size(), 2u);
    EXPECT_EQ(plan[0], "Trace");
    EXPECT_EQ(plan[1], "SassMetrics");
}

TEST(ResolvePassPlan, DeepExpandsToDefaultPlan) {
    TraceArgs a;
    a.passes = {"Deep"};
    const auto plan = resolvePassPlan(a);
    ASSERT_EQ(plan.size(), 3u);
    EXPECT_EQ(plan[0], "Trace");
    EXPECT_EQ(plan[1], "PcSampling");
    EXPECT_EQ(plan[2], "SassMetrics");
}

TEST(ResolvePassPlan, SingleExplicitPassIsOnePass) {
    TraceArgs a;
    a.passes = {"PcSampling"};
    const auto plan = resolvePassPlan(a);
    ASSERT_EQ(plan.size(), 1u);
    EXPECT_EQ(plan[0], "PcSampling");
}

TEST(ResolvePassPlan, NoPassesIsTracePass) {
    TraceArgs a;
    const auto plan = resolvePassPlan(a);
    ASSERT_EQ(plan.size(), 1u);
    EXPECT_EQ(plan[0], "Trace");
}

// ── gpufl upload ────────────────────────────────────────────────────────

TEST(CliParseUpload, BasicLogPath) {
    auto r = parseUploadArgs(argsFor({"/tmp/run"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    EXPECT_EQ(r.args->log_path, "/tmp/run");
    EXPECT_EQ(r.args->timeout_s, 300);
    EXPECT_EQ(r.args->retries, 1);
    EXPECT_FALSE(r.args->quiet);
    EXPECT_FALSE(r.args->all_sessions);
    EXPECT_FALSE(r.args->force);
}

TEST(CliParseUpload, AllFlags) {
    auto r = parseUploadArgs(argsFor({
        "--backend-url=https://api.example.com", "--api-key", "gpfl_k",
        "--api-path=/proxy/api", "--timeout=600", "--retries=3",
        "--quiet", "--all-sessions", "--force", "/tmp/run"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    EXPECT_EQ(r.args->log_path, "/tmp/run");
    EXPECT_EQ(r.args->backend_url, "https://api.example.com");
    EXPECT_EQ(r.args->api_key, "gpfl_k");
    EXPECT_EQ(r.args->api_path, "/proxy/api");
    EXPECT_EQ(r.args->timeout_s, 600);
    EXPECT_EQ(r.args->retries, 3);
    EXPECT_TRUE(r.args->quiet);
    EXPECT_TRUE(r.args->all_sessions);
    EXPECT_TRUE(r.args->force);
}

TEST(CliParseUpload, SpaceFormValue) {
    auto r = parseUploadArgs(argsFor({"--backend-url", "https://x", "/tmp/run"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    EXPECT_EQ(r.args->backend_url, "https://x");
    EXPECT_EQ(r.args->log_path, "/tmp/run");
}

TEST(CliParseUpload, MissingLogPath) {
    auto r = parseUploadArgs(argsFor({"--force"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("LOG_PATH"), std::string::npos);
}

TEST(CliParseUpload, SessionIdAndAllSessionsMutuallyExclusive) {
    auto r = parseUploadArgs(argsFor({"--session-id=abc", "--all-sessions", "/tmp/run"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("mutually exclusive"), std::string::npos);
}

TEST(CliParseUpload, InvalidTimeout) {
    auto r = parseUploadArgs(argsFor({"--timeout=abc", "/tmp/run"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("invalid --timeout"), std::string::npos);
}

TEST(CliParseUpload, NegativeRetriesRejected) {
    auto r = parseUploadArgs(argsFor({"--retries=-5", "/tmp/run"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("invalid --retries"), std::string::npos);
}

TEST(CliParseUpload, ExtraPositionalRejected) {
    auto r = parseUploadArgs(argsFor({"/tmp/a", "/tmp/b"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("extra argument"), std::string::npos);
}

TEST(CliParseUpload, UnknownFlag) {
    auto r = parseUploadArgs(argsFor({"--definitely-not-a-flag", "/tmp/run"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("unknown flag"), std::string::npos);
}

TEST(CliParseUpload, HelpFlag) {
    auto r = parseUploadArgs(argsFor({"--help"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_EQ(r.error, "__help__");
}

// ── gpufl monitor ───────────────────────────────────────────────────────────

TEST(CliParseMonitor, Defaults) {
    auto r = parseMonitorArgs(argsFor({}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    EXPECT_EQ(r.args->name, "gpufl-monitor");
    EXPECT_TRUE(r.args->output_dir.empty());
    EXPECT_EQ(r.args->interval_ms, 5000);
    EXPECT_FALSE(r.args->upload);
    EXPECT_EQ(r.args->api_version, "v1");
    EXPECT_EQ(r.args->log_types, "system");
}

TEST(CliParseMonitor, AllCommonFlags) {
    auto r = parseMonitorArgs(argsFor({
        "--name=llm-node-1", "--output", "/tmp/gpufl-monitor",
        "--interval=1000", "--upload", "--backend-url=https://api.example.com",
        "--api-key", "gpfl_key", "--api-version=v2", "--agent-jar=/tmp/agent.jar",
        "--agent-cursor=/tmp/cursor.json", "--log-types=system,device",
        "-v", "-q"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    EXPECT_EQ(r.args->name, "llm-node-1");
    EXPECT_EQ(r.args->output_dir, "/tmp/gpufl-monitor");
    EXPECT_EQ(r.args->interval_ms, 1000);
    EXPECT_TRUE(r.args->upload);
    EXPECT_EQ(r.args->backend_url, "https://api.example.com");
    EXPECT_EQ(r.args->api_key, "gpfl_key");
    EXPECT_EQ(r.args->api_version, "v2");
    EXPECT_EQ(r.args->agent_jar, "/tmp/agent.jar");
    EXPECT_EQ(r.args->agent_cursor, "/tmp/cursor.json");
    EXPECT_EQ(r.args->log_types, "system,device");
    EXPECT_TRUE(r.args->verbose);
    EXPECT_TRUE(r.args->quiet);
}

TEST(CliParseMonitor, InvalidIntervalRejected) {
    auto r = parseMonitorArgs(argsFor({"--interval=0"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("invalid --interval"), std::string::npos);
}

TEST(CliParseMonitor, AgentJarParsed) {
    auto r = parseMonitorArgs(argsFor({"--agent-jar", "/opt/gpufl-agent.jar"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    EXPECT_EQ(r.args->agent_jar, "/opt/gpufl-agent.jar");
}

TEST(CliParseMonitor, BareArgumentRejected) {
    auto r = parseMonitorArgs(argsFor({"python", "server.py"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("does not launch"), std::string::npos);
}

TEST(CliParseMonitor, HelpFlag) {
    auto r = parseMonitorArgs(argsFor({"--help"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_EQ(r.error, "__help__");
}

TEST(CliParseTopLevel, NoArgsShowsHelp) {
    char* argv[] = {const_cast<char*>("gpufl"), nullptr};
    auto p = parseTopLevel(1, argv);
    EXPECT_EQ(p.sub, Subcommand::Help);
}

TEST(CliParseTopLevel, VersionSubcommand) {
    char* argv[] = {const_cast<char*>("gpufl"),
                    const_cast<char*>("version"), nullptr};
    auto p = parseTopLevel(2, argv);
    EXPECT_EQ(p.sub, Subcommand::Version);
}

TEST(CliParseTopLevel, ShortVersionFlag) {
    char* argv[] = {const_cast<char*>("gpufl"),
                    const_cast<char*>("-V"), nullptr};
    auto p = parseTopLevel(2, argv);
    EXPECT_EQ(p.sub, Subcommand::Version);
}

TEST(CliParseTopLevel, TraceSubcommandStripsFirstToken) {
    char* argv[] = {const_cast<char*>("gpufl"),
                    const_cast<char*>("trace"),
                    const_cast<char*>("--"),
                    const_cast<char*>("./app"), nullptr};
    auto p = parseTopLevel(4, argv);
    EXPECT_EQ(p.sub, Subcommand::Trace);
    ASSERT_EQ(p.remaining.size(), 2u);
    EXPECT_EQ(p.remaining[0], "--");
    EXPECT_EQ(p.remaining[1], "./app");
}

TEST(CliParseTopLevel, MonitorSubcommandStripsFirstToken) {
    char* argv[] = {const_cast<char*>("gpufl"),
                    const_cast<char*>("monitor"),
                    const_cast<char*>("--interval=1000"), nullptr};
    auto p = parseTopLevel(3, argv);
    EXPECT_EQ(p.sub, Subcommand::Monitor);
    ASSERT_EQ(p.remaining.size(), 1u);
    EXPECT_EQ(p.remaining[0], "--interval=1000");
}

TEST(CliParseTopLevel, UnknownSubcommand) {
    char* argv[] = {const_cast<char*>("gpufl"),
                    const_cast<char*>("nope"), nullptr};
    auto p = parseTopLevel(2, argv);
    EXPECT_EQ(p.sub, Subcommand::Unknown);
}

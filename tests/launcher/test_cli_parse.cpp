// Tests for the launcher CLI parser. CPU-only - runs on any platform
// where gpufl_tests builds (Linux/macOS/Windows). The launcher binary
// itself is Linux-gated, but the parser source is portable.

#include <gtest/gtest.h>

#include <regex>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "cli_parse.hpp"
#include "cli_trace_options.hpp"

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

TEST(CliParseNumbers, RejectsOutOfRangeIntegerOptions) {
    constexpr const char* kIntOverflow = "2147483648";
    constexpr const char* kUint64Overflow = "18446744073709551616";

    auto trace_drain = parseTraceArgs(
        argsFor({"--agent-drain-ms", kIntOverflow, "--", "./app"}));
    EXPECT_FALSE(trace_drain.args.has_value());

    auto trace_rows = parseTraceArgs(
        argsFor({"--segment-max-rows", kUint64Overflow, "--", "./app"}));
    EXPECT_FALSE(trace_rows.args.has_value());

    auto trace_launches = parseTraceArgs(
        argsFor({"--deep-launches", kUint64Overflow, "--", "./app"}));
    EXPECT_FALSE(trace_launches.args.has_value());

    auto upload = parseUploadArgs(
        argsFor({"--timeout", kIntOverflow, "./logs"}));
    EXPECT_FALSE(upload.args.has_value());

    auto monitor = parseMonitorArgs(argsFor({"--interval", kIntOverflow}));
    EXPECT_FALSE(monitor.args.has_value());

    auto info = parseInfoArgs(argsFor({"--device", kIntOverflow}));
    EXPECT_FALSE(info.args.has_value());
}

TEST(CliParseNumbers, RetainsStrtolCompatibleLeadingSpaceAndPlusSign) {
    auto trace = parseTraceArgs(
        argsFor({"--agent-drain-ms", " +500", "--", "./app"}));
    ASSERT_TRUE(trace.args.has_value()) << trace.error;
    EXPECT_EQ(trace.args->agent_drain_ms, 500);
}

// ── Long-running session segmentation ─────────────────────────────────────
TEST(CliParseTrace, ParsesBothSegmentationTriggers) {
    auto r = parseTraceArgs(argsFor(
        {"--segment-every=5m", "--segment-max-rows", "2000000", "--", "./app"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    EXPECT_EQ(r.args->segment_every_ms, 300000);
    EXPECT_EQ(r.args->segment_max_rows, 2'000'000u);
    EXPECT_TRUE(segmentationRequested(*r.args));
}

TEST(CliParseTrace, ZeroSegmentationTriggersPreserveOrdinaryMode) {
    auto r = parseTraceArgs(argsFor(
        {"--segment-every=0", "--segment-max-rows=0", "--", "./app"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    EXPECT_FALSE(segmentationRequested(*r.args));
}

TEST(CliParseTrace, RejectsTooShortSegmentationCadence) {
    auto r = parseTraceArgs(argsFor({"--segment-every=59s", "--", "./app"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("at least 60s"), std::string::npos) << r.error;
}

TEST(CliParseTrace, RejectsNonFiniteOverflowAndSubMillisecondDurations) {
    for (const char* value : {
             "nan", "inf", "1e100h", "9223372036854775808ms", "0.5ms"}) {
        auto r = parseTraceArgs(
            argsFor({"--segment-every", value, "--", "./app"}));
        EXPECT_FALSE(r.args.has_value()) << value;
        EXPECT_NE(r.error.find("--segment-every"), std::string::npos)
            << value << ": " << r.error;
    }
}

TEST(CliParseTrace, RejectsInvalidSegmentRowBudget) {
    for (const char* value : {
             "-1", "lots", "1.5", "999999999999999999999999999999"}) {
        auto r = parseTraceArgs(
            argsFor({"--segment-max-rows", value, "--", "./app"}));
        EXPECT_FALSE(r.args.has_value()) << value;
        EXPECT_NE(r.error.find("--segment-max-rows"), std::string::npos)
            << value << ": " << r.error;
    }
}

TEST(CliParseTrace, SegmentationAcceptsTheV1PassWhitelist) {
    for (const char* pass : {"Trace", "PmSampling"}) {
        auto r = parseTraceArgs(argsFor(
            {"--segment-every=5m", "--passes", pass, "--", "./app"}));
        EXPECT_TRUE(r.args.has_value()) << pass << ": " << r.error;
    }
}

TEST(CliParseTrace, SegmentationAcceptsTheAdaptiveTracePmPlan) {
    auto r = parseTraceArgs(argsFor(
        {"--segment-every=5m", "--deep-when=kernel_launch_rate<10",
         "--deep-for=2s", "--", "./app"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    EXPECT_EQ(resolveCaptureMode(*r.args), CaptureMode::AdaptiveDeepWindow);
}

TEST(CliParseTrace, SegmentationRejectsUnsupportedV1Passes) {
    for (const char* pass : {
             "PcSampling", "SassMetrics", "RangeProfiler",
             "RangeProfilerKernelReplay", "Deep", "Trace+PcSampling"}) {
        auto r = parseTraceArgs(argsFor(
            {"--segment-every=5m", "--passes", pass, "--", "./app"}));
        EXPECT_FALSE(r.args.has_value()) << pass;
        EXPECT_NE(r.error.find("segmentation V1"), std::string::npos)
            << pass << ": " << r.error;
    }
}

TEST(CliParseTrace, SegmentationRejectsMultiPassAnalysis) {
    auto r = parseTraceArgs(argsFor(
        {"--passes=Trace,PmSampling", "--segment-every=5m", "--", "./app"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("multi-pass"), std::string::npos) << r.error;
}

TEST(CliParseTrace, ExecutionBoundaryRejectsInheritedAnalysisId) {
    TraceArgs args;
    args.segment_every_ms = 300000;
    const std::string error =
        validateTraceSegmentation(args, "analysis-from-parent");
    EXPECT_NE(error.find("GPUFL_ANALYSIS_ID"), std::string::npos) << error;
}

TEST(CliParseTrace, RollEveryParsesWithSegmentEvery) {
    auto r = parseTraceArgs(argsFor(
        {"--roll-every=3m", "--segment-every=60s", "--", "./app"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    EXPECT_EQ(r.args->run_roll_every_ms, 180'000);
    EXPECT_EQ(r.args->segment_every_ms, 60'000);
}

TEST(CliParseTrace, RollEveryRejectsShorterThanSegmentEvery) {
    auto r = parseTraceArgs(argsFor(
        {"--roll-every=90s", "--segment-every=120s", "--", "./app"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("at least --segment-every"), std::string::npos)
        << r.error;
}

TEST(CliParseTrace, RollEveryRequiresASegmentTimeTrigger) {
    auto r = parseTraceArgs(argsFor({"--roll-every=3m", "--", "./app"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("requires --segment-every"), std::string::npos)
        << r.error;

    // A row trigger alone is not enough: with no rows arriving, no boundary is
    // ever due and the part would grow without bound.
    auto rows = parseTraceArgs(argsFor(
        {"--roll-every=3m", "--segment-max-rows=1000", "--", "./app"}));
    EXPECT_FALSE(rows.args.has_value());
    EXPECT_NE(rows.error.find("requires --segment-every"), std::string::npos)
        << rows.error;
}

TEST(CliParseTrace, RollMaxBytesAcceptsSuffixesAndNeedsAnySegmentTrigger) {
    auto r = parseTraceArgs(argsFor(
        {"--roll-max-bytes=50g", "--segment-max-rows=1000", "--", "./app"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    EXPECT_EQ(r.args->run_roll_max_bytes, 50ULL * 1024 * 1024 * 1024);

    auto bare = parseTraceArgs(argsFor(
        {"--roll-max-bytes=50g", "--", "./app"}));
    EXPECT_FALSE(bare.args.has_value());
    EXPECT_NE(bare.error.find("--segment-max-rows"), std::string::npos)
        << bare.error;
}

TEST(CliParseTrace, RollMaxBytesRejectsGarbageAndOverflow) {
    for (const char* value : {"50gg", "g", "-1", "1e9", "50g50",
                              "99999999999999999999g"}) {
        auto r = parseTraceArgs(argsFor(
            {((std::string("--roll-max-bytes=") + value).c_str()),
             "--segment-every=60s", "--", "./app"}));
        EXPECT_FALSE(r.args.has_value()) << value;
    }
}

TEST(CliParseTrace, OvershootWarningFiresOnlyWhenSegmentIsCoarse) {
    TraceArgs coarse;
    coarse.segment_every_ms = 60'000;
    coarse.run_roll_every_ms = 180'000;  // segment is a third of the part
    EXPECT_NE(segmentationWarning(coarse).find("overshoot"),
              std::string::npos);

    TraceArgs fine;
    fine.segment_every_ms = 60'000;
    fine.run_roll_every_ms = 3'600'000;
    EXPECT_TRUE(segmentationWarning(fine).empty());

    TraceArgs no_roll;
    no_roll.segment_every_ms = 60'000;
    EXPECT_TRUE(segmentationWarning(no_roll).empty());
}

TEST(CliParseTrace, DirectTraceArgsCannotBypassMinimumCadence) {
    TraceArgs args;
    args.segment_every_ms = 1;
    EXPECT_NE(validateTraceExecutionMode(args).find("at least 60s"),
              std::string::npos);
}

TEST(CliParseTrace, GeneratedRunIdIsUniqueUuidV4) {
    const std::regex uuid_v4(
        "^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
        "[89ab][0-9a-f]{3}-[0-9a-f]{12}$");
    std::unordered_set<std::string> ids;
    for (int i = 0; i < 100; ++i) {
        const std::string id = generateRunId();
        EXPECT_TRUE(std::regex_match(id, uuid_v4)) << id;
        EXPECT_TRUE(ids.insert(id).second) << "duplicate UUID: " << id;
    }
}

TEST(CliParseTrace, SegmentationExecutionStaysGatedUntilCoordinatorLands) {
    EXPECT_TRUE(segmentationRuntimeReady());
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

TEST(CliParseTrace, SourceCaptureDefaultsOnAndCanBeDisabled) {
    auto defaults = parseTraceArgs(argsFor({"--", "./bin"}));
    ASSERT_TRUE(defaults.args.has_value()) << defaults.error;
    EXPECT_FALSE(defaults.args->no_source);

    auto opted_out = parseTraceArgs(
        argsFor({"--no-source", "--", "./bin"}));
    ASSERT_TRUE(opted_out.args.has_value()) << opted_out.error;
    EXPECT_TRUE(opted_out.args->no_source);

    auto rooted = parseTraceArgs(
        argsFor({"--source-root", "./cuda-project", "--", "./bin"}));
    ASSERT_TRUE(rooted.args.has_value()) << rooted.error;
    EXPECT_EQ(rooted.args->source_root, "./cuda-project");
}

TEST(CliParseTrace, BooleanFlagsRejectInlineValues) {
    const auto r = parseTraceArgs(
        argsFor({"--verbose=true", "--", "./bin"}));
    ASSERT_FALSE(r.args.has_value());
    EXPECT_EQ(r.error, "unknown flag: --verbose");
}

TEST(CliParseHelp, TraceSimpleOptionsComeFromTheRegistry) {
    const std::string help = traceHelp();
    EXPECT_NE(help.find("-n, --name=<NAME>"), std::string::npos);
    EXPECT_NE(help.find("--backend-url=<URL>"), std::string::npos);
    EXPECT_NE(help.find("--no-source"), std::string::npos);
    EXPECT_NE(help.find("--source-root=<DIR>"), std::string::npos);
    EXPECT_NE(help.find("GPUFL_BACKEND_URL"), std::string::npos);
}

// The registry is the only source of trace help, so every flag a user can pass
// must appear in the rendered output. Without this, adding an option to the
// table and forgetting its help section produces a flag that works but that
// nothing documents - the exact drift the registry exists to prevent.
TEST(CliParseHelp, EveryTraceOptionIsDocumentedOrDeliberatelyRemoved) {
    // Flags kept only to print a migration hint; they must NOT be advertised.
    const std::set<std::string> removed = {"--profile", "--engine"};
    const std::string help = traceHelp();

    for (const std::string& alias : traceOptionAliases()) {
        const bool documented = help.find(alias) != std::string::npos;
        if (removed.count(alias) > 0) {
            EXPECT_FALSE(documented)
                << alias << " is a removed flag and must stay out of help";
        } else {
            EXPECT_TRUE(documented)
                << alias << " is accepted by the parser but missing from help; "
                   "give it a help section and description in the registry";
        }
    }
}

// Help is assembled section by section, so a section that exists in the enum but
// is never rendered would silently swallow its options.
TEST(CliParseHelp, EveryTraceHelpSectionRendersSomething) {
    const TraceHelpSection sections[] = {
        TraceHelpSection::Capture,   TraceHelpSection::Runtime,
        TraceHelpSection::Segmentation, TraceHelpSection::Window,
        TraceHelpSection::Deep,      TraceHelpSection::Sampling,
    };
    const std::string help = traceHelp();
    for (const TraceHelpSection section : sections) {
        const std::string rendered = formatTraceSimpleOptions(section);
        EXPECT_FALSE(rendered.empty())
            << "help section " << static_cast<int>(section) << " is empty";
        // And it actually reached the assembled help, not just the formatter.
        const std::size_t first_newline = rendered.find('\n');
        ASSERT_NE(first_newline, std::string::npos);
        EXPECT_NE(help.find(rendered.substr(0, first_newline)),
                  std::string::npos)
            << "section " << static_cast<int>(section)
            << " renders but is not included in traceHelp()";
    }
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

// ── --deep-*: bound the DEEP engines inside a target that keeps running,
// as opposed to --window, which bounds the target's lifetime.

TEST(CliParseTrace, DeepWindowFlagsParse) {
    auto r = parseTraceArgs(argsFor({"--deep-after=30s", "--deep-for=3s",
                                     "--deep-cooldown=1m", "--", "./bin"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    EXPECT_EQ(r.args->deep_after_ms, 30000);
    EXPECT_EQ(r.args->deep_for_ms, 3000);
    EXPECT_EQ(r.args->deep_cooldown_ms, 60000);
    EXPECT_TRUE(r.args->deep_requested);
}

TEST(CliParseTrace, DeepLaunchesParses) {
    auto r = parseTraceArgs(argsFor({"--deep-launches", "500", "--", "./bin"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    EXPECT_EQ(r.args->deep_launches, 500u);
    EXPECT_TRUE(r.args->deep_requested);
}

TEST(CliParseTrace, DeepWindowWithoutABoundRejected) {
    // An unbounded deep window is just "profile deeply for the whole run".
    auto r = parseTraceArgs(argsFor({"--deep-after=30s", "--", "./bin"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("needs a bound"), std::string::npos);
}

TEST(CliParseTrace, DeepLaunchesAloneIsABound) {
    auto r = parseTraceArgs(argsFor({"--deep-launches=200", "--", "./bin"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    EXPECT_EQ(r.args->deep_after_ms, 0) << "arms at the first launch";
}

TEST(CliParseTrace, DeepLaunchesZeroRejected) {
    auto r = parseTraceArgs(argsFor({"--deep-launches=0", "--", "./bin"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("invalid --deep-launches"), std::string::npos);
}

TEST(CliParseTrace, DeepForBogusDurationRejected) {
    auto r = parseTraceArgs(argsFor({"--deep-for=soon", "--", "./bin"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("invalid --deep-for"), std::string::npos);
}

TEST(CliParseTrace, NoDeepFlagsLeavesDeepWindowOff) {
    auto r = parseTraceArgs(argsFor({"--window=10s", "--", "./bin"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    EXPECT_FALSE(r.args->deep_requested);
    EXPECT_EQ(r.args->window_ms, 10000) << "--window is unaffected";
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

TEST(CliParseTrace, PassesDeepParsedAsEngine) {
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

TEST(CliParseTrace, DeepCanBeAPassAlongsideOthers) {
    auto r = parseTraceArgs(argsFor({"--passes=Trace,Deep", "--", "./bin"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    ASSERT_EQ(r.args->passes.size(), 2u);
    EXPECT_EQ(r.args->passes[0], "Trace");
    EXPECT_EQ(r.args->passes[1], "Deep");
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

TEST(ResolvePassPlan, DeepResolvesToSinglePass) {
    TraceArgs a;
    a.passes = {"Deep"};
    const auto plan = resolvePassPlan(a);
    ASSERT_EQ(plan.size(), 1u);
    EXPECT_EQ(plan[0], "Deep");
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
    EXPECT_TRUE(r.args->all_sessions);  // every session in the dir, by default
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

TEST(CliParseUpload, SessionIdNoLongerSupported) {
    auto r = parseUploadArgs(argsFor({"--session-id=abc", "/tmp/run"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("no longer supported"), std::string::npos);
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

TEST(CliParseInfo, DefaultsToTextForAllDevices) {
    auto r = parseInfoArgs({});
    ASSERT_TRUE(r.args.has_value()) << r.error;
    EXPECT_FALSE(r.args->json);
    EXPECT_FALSE(r.args->device_id.has_value());
}

TEST(CliParseInfo, JsonFlag) {
    auto r = parseInfoArgs(argsFor({"--json"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    EXPECT_TRUE(r.args->json);
}

TEST(CliParseInfo, DeviceEqualsForm) {
    auto r = parseInfoArgs(argsFor({"--device=2", "--json"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    ASSERT_TRUE(r.args->device_id.has_value());
    EXPECT_EQ(*r.args->device_id, 2);
}

TEST(CliParseInfo, DeviceSpaceForm) {
    auto r = parseInfoArgs(argsFor({"--device", "0"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    ASSERT_TRUE(r.args->device_id.has_value());
    EXPECT_EQ(*r.args->device_id, 0);
}

TEST(CliParseInfo, NegativeDeviceRejected) {
    auto r = parseInfoArgs(argsFor({"--device=-1"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("non-negative integer"), std::string::npos);
}

TEST(CliParseInfo, PositionalArgumentRejected) {
    auto r = parseInfoArgs(argsFor({"0"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("does not accept positional"), std::string::npos);
}

TEST(CliParseInfo, HelpFlag) {
    auto r = parseInfoArgs(argsFor({"--help"}));
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

TEST(CliParseTopLevel, InfoSubcommandStripsFirstToken) {
    char* argv[] = {const_cast<char*>("gpufl"),
                    const_cast<char*>("info"),
                    const_cast<char*>("--json"), nullptr};
    auto p = parseTopLevel(3, argv);
    EXPECT_EQ(p.sub, Subcommand::Info);
    ASSERT_EQ(p.remaining.size(), 1u);
    EXPECT_EQ(p.remaining[0], "--json");
}

TEST(CliParseTopLevel, UnknownSubcommand) {
    char* argv[] = {const_cast<char*>("gpufl"),
                    const_cast<char*>("nope"), nullptr};
    auto p = parseTopLevel(2, argv);
    EXPECT_EQ(p.sub, Subcommand::Unknown);
}

// ── --passes and --deep-* are different execution models ────────────────────
//
// They cannot be combined: the deep engines are fixed before the first CUDA
// call, so a --passes list either already holds what the window would arm, or
// does not - and `--passes=Trace --deep-after=30s` silently opened a window
// that armed nothing. Rejected before the target runs, in either flag order.

namespace {

gpufl::launcher::TraceParseResult parseTrace(std::vector<std::string> argv) {
    return gpufl::launcher::parseTraceArgs(argv);
}

bool rejectsPassesWithDeep(const std::vector<std::string>& argv) {
    const auto r = parseTrace(argv);
    return !r.args.has_value() &&
           r.error.find("--passes cannot be combined") != std::string::npos;
}

}  // namespace

TEST(CliParseDeepModeTest, PassesBeforeDeepFlagIsRejected) {
    EXPECT_TRUE(rejectsPassesWithDeep(
        {"--passes=Trace", "--deep-after=30s", "--deep-for=5s", "--", "app"}));
}

TEST(CliParseDeepModeTest, DeepFlagBeforePassesIsRejected) {
    // Order must not decide: a user who writes the flags the other way round
    // is making the same mistake.
    EXPECT_TRUE(rejectsPassesWithDeep(
        {"--deep-when=custom.token_rate<100", "--deep-for=5s",
         "--passes=Deep", "--", "app"}));
}

TEST(CliParseDeepModeTest, SpaceSeparatedFormIsRejectedToo) {
    EXPECT_TRUE(rejectsPassesWithDeep(
        {"--passes", "PcSampling", "--deep-for", "5s", "--", "app"}));
}

TEST(CliParseDeepModeTest, EveryDeepFlagTriggersTheRejection) {
    // --deep-after is included deliberately, with no grandfather clause: two
    // rules ("--deep-when refuses, --deep-after tolerates") would be harder to
    // explain than the break.
    EXPECT_TRUE(rejectsPassesWithDeep(
        {"--passes=Trace", "--deep-after=1s", "--deep-for=1s", "--", "app"}));
    EXPECT_TRUE(rejectsPassesWithDeep(
        {"--passes=Trace", "--deep-for=1s", "--", "app"}));
    EXPECT_TRUE(rejectsPassesWithDeep(
        {"--passes=Trace", "--deep-launches=500", "--", "app"}));
    EXPECT_TRUE(rejectsPassesWithDeep(
        {"--passes=Trace", "--deep-when=kernel_launch_rate<10",
         "--deep-for=1s", "--", "app"}));
}

TEST(CliParseDeepModeTest, TheRejectionNamesTheWayOut) {
    const auto r = parseTrace(
        {"--passes=Trace", "--deep-for=5s", "--", "app"});
    ASSERT_FALSE(r.args.has_value());
    // An error that only says "no" leaves the user guessing which flag to drop.
    EXPECT_NE(r.error.find("Drop --passes"), std::string::npos) << r.error;
}

TEST(CliParseDeepModeTest, TheTwoWindowTriggersAreMutuallyExclusive) {
    // Both set is not "either may open it". Measured on the 3090: the
    // scheduled window opens at t=0, the rule is refused behind it, and the
    // summary then reports `never_true` for a condition that held all run.
    const auto r = parseTrace({"--deep-when=kernel_launch_rate<10",
                               "--deep-after=1s", "--deep-for=2s", "--", "app"});
    ASSERT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("--deep-when"), std::string::npos) << r.error;
    EXPECT_NE(r.error.find("--deep-after"), std::string::npos) << r.error;
}

TEST(CliParseDeepModeTest, AConditionalRunDoesNotCarryATimeTrigger) {
    const auto r = parseTrace({"--deep-when=kernel_launch_rate<10",
                               "--deep-for=2s", "--", "app"});
    ASSERT_TRUE(r.args.has_value()) << r.error;
    // deep_after_ms still holds its default 0; what must not happen is the
    // launcher treating that default as a request. GPUFL_DEEP_AFTER_MS installs
    // the time trigger by being present at all, so the flag being unset has to
    // survive as far as the environment.
    EXPECT_FALSE(r.args->deep_after_set);
}

TEST(CliParseDeepModeTest, ADeepRunResolvesToExactlyOneAdaptivePass) {
    const auto r = parseTrace({"--deep-for=5s", "--", "app"});
    ASSERT_TRUE(r.args.has_value()) << r.error;

    const auto plan = gpufl::launcher::resolvePassPlan(*r.args);
    // One pass, not one per engine: a window triggered by a live condition
    // cannot be reproduced across relaunches, so splitting it would change
    // what is being measured.
    ASSERT_EQ(plan.size(), 1u);
    EXPECT_NE(plan[0].find("Trace"), std::string::npos) << plan[0];
    EXPECT_NE(plan[0].find('+'), std::string::npos)
        << "the deep engine should be prepared alongside the base: " << plan[0];
}

TEST(CliParseDeepModeTest, TheAdaptivePlanPinsTheBaseAndArmsOnlyInTheWindow) {
    const auto r = parseTrace({"--deep-for=5s", "--", "app"});
    ASSERT_TRUE(r.args.has_value()) << r.error;

    const auto plan = gpufl::launcher::resolveAdaptivePlan(*r.args);
    // Trace is pinned rather than left to the deep engine's own policy:
    // kernel_launch_rate and recent_kernel_ms come from its completed-kernel
    // records, and a rule that loses its metric reads as "never held".
    EXPECT_EQ(plan.base, "Trace");
    EXPECT_TRUE(plan.arm_window_only);
    ASSERT_FALSE(plan.selected_deep.empty());
    // PM only for now. PC and SASS join once their dormant cost has been
    // measured - picking the deepest engine is not the same as picking the
    // deepest one that fits an overhead budget.
    EXPECT_EQ(plan.selected_deep.size(), 1u);
    EXPECT_EQ(plan.selected_deep[0], "PmSampling");
}

TEST(CliParseDeepModeTest, WithoutADeepFlagThereIsNoAdaptivePlan) {
    const auto r = parseTrace({"--passes=Trace,PcSampling", "--", "app"});
    ASSERT_TRUE(r.args.has_value()) << r.error;

    EXPECT_TRUE(gpufl::launcher::resolveAdaptivePlan(*r.args).selected_deep.empty());
    // The explicit list is honoured untouched.
    EXPECT_EQ(gpufl::launcher::resolvePassPlan(*r.args).size(), 2u);
}

TEST(CliParseDeepModeTest, PlainTraceIsStillTheDefault) {
    const auto r = parseTrace({"--", "app"});
    ASSERT_TRUE(r.args.has_value()) << r.error;
    const auto plan = gpufl::launcher::resolvePassPlan(*r.args);
    ASSERT_EQ(plan.size(), 1u);
    EXPECT_EQ(plan[0], "Trace");
}

TEST(CliParseDeepModeTest, ProgrammaticMixedModeFailsSharedValidation) {
    gpufl::launcher::TraceArgs args;
    args.passes = {"Trace"};
    args.deep_requested = true;

    const std::string error =
        gpufl::launcher::validateTraceExecutionMode(args);
    EXPECT_NE(error.find("--passes cannot be combined"), std::string::npos);
    EXPECT_NE(error.find("Drop --passes"), std::string::npos);
}

TEST(CliParseDeepModeTest, SharedValidationAcceptsEachModeSeparately) {
    gpufl::launcher::TraceArgs explicit_args;
    explicit_args.passes = {"Trace", "PmSampling"};
    EXPECT_TRUE(
        gpufl::launcher::validateTraceExecutionMode(explicit_args).empty());

    gpufl::launcher::TraceArgs adaptive_args;
    adaptive_args.deep_requested = true;
    EXPECT_TRUE(
        gpufl::launcher::validateTraceExecutionMode(adaptive_args).empty());
    EXPECT_EQ(gpufl::launcher::resolveCaptureMode(explicit_args),
              gpufl::launcher::CaptureMode::ExplicitPasses);
    EXPECT_EQ(gpufl::launcher::resolveCaptureMode(adaptive_args),
              gpufl::launcher::CaptureMode::AdaptiveDeepWindow);
}

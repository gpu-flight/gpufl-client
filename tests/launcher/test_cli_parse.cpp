// Tests for the launcher CLI parser. CPU-only — runs on any platform
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
    EXPECT_EQ(r.args->profile, "comprehensive");
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

TEST(CliParseTrace, ProfileValid) {
    auto r = parseTraceArgs(argsFor({"--profile=light", "--", "./bin"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    EXPECT_EQ(r.args->profile, "light");
}

TEST(CliParseTrace, ProfileInvalid) {
    auto r = parseTraceArgs(argsFor({"--profile=galactic", "--", "./bin"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("invalid --profile"), std::string::npos);
}

TEST(CliParseTrace, EngineValid) {
    auto r = parseTraceArgs(argsFor({"--engine=pc-sampling-with-sass", "--", "./bin"}));
    ASSERT_TRUE(r.args.has_value()) << r.error;
    EXPECT_EQ(r.args->engine, "pc-sampling-with-sass");
}

TEST(CliParseTrace, EngineInvalid) {
    auto r = parseTraceArgs(argsFor({"--engine=hyperdrive", "--", "./bin"}));
    EXPECT_FALSE(r.args.has_value());
    EXPECT_NE(r.error.find("invalid --engine"), std::string::npos);
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

TEST(CliParseTopLevel, UnknownSubcommand) {
    char* argv[] = {const_cast<char*>("gpufl"),
                    const_cast<char*>("nope"), nullptr};
    auto p = parseTopLevel(2, argv);
    EXPECT_EQ(p.sub, Subcommand::Unknown);
}

#pragma once

#include <optional>
#include <string>
#include <vector>

namespace gpufl::launcher {

// Parsed `gpufl trace` invocation. The launcher translates this into
// the env vars in include/gpufl/inject/inject_entry.hpp and then
// fork+execs the target.
struct TraceArgs {
    std::string name;                   // --name / -n; default: basename of cmd[0]
    std::string output_dir;             // --output / -o; default: ~/.gpufl/traces/{ts}_{sid}
    std::string profile = "comprehensive"; // --profile
    std::string engine;                 // --engine; empty = profile default
    bool verbose = false;               // -v
    bool quiet = false;                 // -q
    std::vector<std::string> command;   // tokens after `--`
};

enum class Subcommand {
    Help,        // `gpufl --help` / `gpufl` with no args
    Version,     // `gpufl version` / `gpufl -V`
    Trace,       // `gpufl trace [opts] -- <command>...`
    Unknown,
};

struct ParsedTopLevel {
    Subcommand sub = Subcommand::Help;
    std::vector<std::string> remaining;  // argv after the subcommand token
};

ParsedTopLevel parseTopLevel(int argc, char** argv);

// Parses the args passed to `gpufl trace`. On success returns the
// populated TraceArgs. On failure (bad flag, missing `--`, no command),
// returns the error message.
struct TraceParseResult {
    std::optional<TraceArgs> args;
    std::string error;
};

TraceParseResult parseTraceArgs(const std::vector<std::string>& argv);

// Help text printed for `gpufl --help` and `gpufl trace --help`.
const char* topLevelHelp();
const char* traceHelp();

}  // namespace gpufl::launcher

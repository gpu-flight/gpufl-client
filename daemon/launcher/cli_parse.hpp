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
    bool upload = false;                // --upload: ship trace to backend post-run
    std::vector<std::string> command;   // tokens after `--`
};

// Parsed `gpufl upload` invocation. Mirrors the flag surface of the
// (now-retired) Python `gpufl.cli` uploader; runUpload() in
// upload_command.cpp resolves creds from env when a flag is omitted and
// calls gpufl::uploadLogs().
struct UploadArgs {
    std::string log_path;           // positional: session log-path prefix
    std::string backend_url;        // --backend-url (else env GPUFL_BACKEND_URL)
    std::string api_key;            // --api-key (else env GPUFL_API_KEY)
    std::string api_path;           // --api-path; empty resolves to /api/v1
    int timeout_s = 300;            // --timeout (seconds); total wall budget
    int retries = 1;                // --retries per failing POST
    bool quiet = false;             // --quiet: suppress progress lines
    std::string session_id;         // --session-id (mutually excl. with --all-sessions)
    bool all_sessions = false;      // --all-sessions
    bool force = false;             // --force: re-upload despite cursor
};

enum class Subcommand {
    Help,        // `gpufl --help` / `gpufl` with no args
    Version,     // `gpufl version` / `gpufl -V`
    Trace,       // `gpufl trace [opts] -- <command>...`
    Upload,      // `gpufl upload <log_path> [opts]`
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

// Parses the args passed to `gpufl upload`. On success returns the
// populated UploadArgs. On failure (missing log_path, bad flag,
// mutually-exclusive selection) returns the error message. error ==
// "__help__" signals the caller to print uploadHelp().
struct UploadParseResult {
    std::optional<UploadArgs> args;
    std::string error;
};

UploadParseResult parseUploadArgs(const std::vector<std::string>& argv);

// Help text printed for `gpufl --help`, `gpufl trace --help`,
// and `gpufl upload --help`.
const char* topLevelHelp();
const char* traceHelp();
const char* uploadHelp();

}  // namespace gpufl::launcher

#pragma once

#include <cstdint>
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
    // --passes: explicit capture plan: a comma-separated list of engines, one
    // isolated pass each (e.g. "Trace,PcSampling,SassMetrics"). "Deep" is just
    // the Deep engine (PcSampling + SassMetrics in one pass), like any other
    // token. Empty here means "no explicit plan"; the launcher runs a single
    // Trace pass.
    std::vector<std::string> passes;
    bool verbose = false;               // -v
    bool quiet = false;                 // -q
    bool upload = false;                // --upload: start gpufl-agent for live upload
    std::string backend_url;            // --backend-url; else GPUFL_BACKEND_URL
    std::string api_key;                // --api-key; else GPUFL_API_KEY
    std::string api_version = "v1";     // --api-version
    std::string agent_jar;              // --agent-jar; else GPUFL_AGENT_JAR
    std::string agent_cursor;           // --agent-cursor; default <output>/cursor.json
    std::string log_types = "device,scope,system,sass"; // --log-types (sass carries cubin disassembly + source artifacts)
    int agent_drain_ms = 60000;         // --agent-drain-ms: cap on waiting for the agent to finish uploading
    // Bounded window profiling (`gpufl trace` only): bound a capture of a
    // long-running target that never exits on its own. warmup defers the
    // capture start (via GPUFL_INJECT_INIT_DELAY_MS); window then runs for a
    // fixed wall-clock before the launcher stops the target. All in ms.
    int64_t warmup_ms = 0;              // --warmup; 0 = start capturing immediately
    int64_t window_ms = 0;              // --window; 0 = run to the target's natural exit
    int64_t window_timeout_ms = 0;      // --window-timeout; hard cap on total runtime (0 = warmup+window)
    std::string after_window = "stop";  // --after-window; "stop" is the only value today
    std::vector<std::string> command;   // tokens after `--`
};

// Parsed `gpufl upload` invocation. Mirrors the flag surface of the
// (now-retired) Python `gpufl.cli` uploader; runUpload() in
// upload_command.cpp resolves creds from env when a flag is omitted and
// calls gpufl::uploadLogs().
struct UploadArgs {
    std::string log_path;           // positional: trace output directory
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

// Parsed `gpufl monitor` invocation. This starts the long-running
// telemetry-only sampler in the launcher process. When --upload is set, the
// launcher also starts gpufl-agent as the live uploader.
struct MonitorArgs {
    std::string name = "gpufl-monitor";  // --name / -n
    std::string output_dir;              // --output / -o; default ~/.gpufl/monitor/{ts}_{sid}
    int interval_ms = 5000;              // --interval
    bool upload = false;                 // --upload: start gpufl-agent
    std::string backend_url;             // --backend-url; else GPUFL_BACKEND_URL
    std::string api_key;                 // --api-key; else GPUFL_API_KEY
    std::string api_version = "v1";      // --api-version
    std::string agent_jar;               // --agent-jar; else GPUFL_AGENT_JAR
    std::string agent_cursor;            // --agent-cursor; default <output>/cursor.json
    std::string log_types = "system";    // --log-types
    bool quiet = false;                  // -q
    bool verbose = false;                // -v
};

enum class Subcommand {
    Help,        // `gpufl --help` / `gpufl` with no args
    Version,     // `gpufl version` / `gpufl -V`
    Trace,       // `gpufl trace [opts] -- <command>...`
    Upload,      // `gpufl upload <log_path> [opts]`
    Monitor,     // `gpufl monitor [opts]`
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

// Resolves the ordered capture plan (one isolated CUPTI engine per pass) from
// parsed trace args. Precedence:
//   1. explicit --passes    -> the listed engines, one pass each (Deep is the
//                             Deep engine, not an expansion);
//   2. otherwise            -> a single Trace pass.
// A returned size() > 1 is a multi-pass run (the launcher assigns one
// analysis_id and labels each pass), shared by the Linux and Windows launchers.
std::vector<std::string> resolvePassPlan(const TraceArgs& args);

// Parses the args passed to `gpufl upload`. On success returns the
// populated UploadArgs. On failure (missing log_path, bad flag,
// mutually-exclusive selection) returns the error message. error ==
// "__help__" signals the caller to print uploadHelp().
struct UploadParseResult {
    std::optional<UploadArgs> args;
    std::string error;
};

UploadParseResult parseUploadArgs(const std::vector<std::string>& argv);

// Parses the args passed to `gpufl monitor`.
struct MonitorParseResult {
    std::optional<MonitorArgs> args;
    std::string error;
};

MonitorParseResult parseMonitorArgs(const std::vector<std::string>& argv);

// Help text printed for `gpufl --help`, `gpufl trace --help`,
// `gpufl upload --help`, and `gpufl monitor --help`.
const char* topLevelHelp();
const char* traceHelp();
const char* uploadHelp();
const char* monitorHelp();

}  // namespace gpufl::launcher

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
    // Bounded DEEP window - unrelated to --window above, which bounds the
    // target's LIFETIME. These keep the target running and instead bound how
    // long the deep engines (PC sampling / SASS / PM / Range) stay armed
    // inside it. A target whose source can't be edited has no way to call
    // gpufl::deepWindow(), so time is the trigger the launcher can offer.
    // Any of them turns on window-only arming (GPUFL_DEEP_ARM=window).
    int64_t  deep_after_ms = 0;      // --deep-after; 0 = arm at the first launch
    bool     deep_after_set = false; // distinguishes "--deep-after=0" from absent
    int64_t  deep_for_ms = 0;        // --deep-for; duration bound, 0 = none
    uint64_t deep_launches = 0;      // --deep-launches; launch bound, 0 = none
    // --deep-when: open the window when a metric crosses a threshold, e.g.
    // "custom.token_rate<1000 for 2s". Empty = time/launch triggered only.
    std::string deep_when;
    int64_t  deep_cooldown_ms = 0;   // --deep-cooldown; quiet time between windows
    bool     deep_requested = false; // any --deep-* flag was given
    // Long-running run segmentation. Zero disables the corresponding trigger;
    // both zero preserves the ordinary single-session path.
    int64_t segment_every_ms = 0;    // --segment-every
    uint64_t segment_max_rows = 0;   // --segment-max-rows
    // PC sampling period as a log2 exponent (2^N GPU cycles/sample, valid 5..31;
    // lower = more frequent → catches shorter kernels). 0 = leave the engine
    // default. Plumbed to the injected target via GPUFL_PC_SAMPLING_PERIOD.
    uint32_t pc_sample_period = 0;      // --pc-sample-period; 0 = engine default
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
    int timeout_s = 300;            // --timeout (seconds); cap on waiting for the agent
    int retries = 1;                // --retries (accepted; the agent retries internally)
    bool quiet = false;             // --quiet: suppress progress lines
    bool all_sessions = true;       // every session in the dir is uploaded (the default)
    bool force = false;             // --force: re-upload despite cursor (throwaway cursor)
    std::string agent_jar;          // --agent-jar; else GPUFL_AGENT_JAR / gpufl-agent on PATH
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

// Parsed `gpufl info` invocation. This command only queries local device
// capabilities; it never starts a trace or uploads data.
struct InfoArgs {
    bool json = false;                   // --json: machine-readable output
    std::optional<int> device_id;        // --device: limit output to one device
};

enum class Subcommand {
    Help,        // `gpufl --help` / `gpufl` with no args
    Version,     // `gpufl version` / `gpufl -V`
    Trace,       // `gpufl trace [opts] -- <command>...`
    Upload,      // `gpufl upload <log_path> [opts]`
    Monitor,     // `gpufl monitor [opts]`
    Info,        // `gpufl info [--json] [--device <id>]`
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

/**
 * Validate execution-mode invariants independently of argument parsing.
 *
 * TraceArgs is intentionally a simple value type and is also constructed by
 * tests and shared launcher code. Keep this check at both the parser boundary
 * and the execution boundary so those callers cannot create a mixed mode.
 * Returns an empty string when valid, otherwise a user-facing error.
 */
std::string validateTraceExecutionMode(const TraceArgs& args);

// The shortest user-configurable time cadence. This protects the backend from
// an accidental session storm; unit tests of the future coordinator use a fake
// clock instead of weakening this production CLI bound.
constexpr int64_t kMinSegmentEveryMs = 60'000;

/** True when at least one segmentation trigger is enabled. */
bool segmentationRequested(const TraceArgs& args);

/**
 * Validate segmentation-specific mode restrictions. inherited_analysis_id is
 * supplied by the execution boundary so an exported GPUFL_ANALYSIS_ID cannot
 * silently turn a segmented single-pass run into an invalid two-axis run.
 */
std::string validateTraceSegmentation(
    const TraceArgs& args,
    const std::string& inherited_analysis_id = std::string());

/** Generate the launcher-owned UUIDv4 shared by every segment in one run. */
std::string generateRunId();

/**
 * False until SegmentCoordinator cutover is implemented. Keeping this as an
 * explicit execution-boundary gate lets the parser/wire contract land without
 * exposing a flag that claims to split sessions but silently produces one.
 */
bool segmentationRuntimeReady();

/**
 * How a run decides which engines to select - two modes, never mixed.
 *
 * ExplicitPasses is the caller saying "run exactly these engines, relaunching
 * the target once per pass". AdaptiveDeepWindow is the caller saying "watch for
 * a condition and profile deeply when it happens" and leaving the engines to
 * gpufl.
 *
 * They cannot be combined because the engine set is fixed before the trigger:
 * an adaptive window may arm only engines selected when the target starts.
 * Accepting a --passes list alongside a deep flag would let a user ask for a
 * base with nothing a window could arm - `--passes=Trace --deep-after=30s`
 * silently produced a window that armed nothing.
 */
enum class CaptureMode {
    ExplicitPasses,
    AdaptiveDeepWindow,
};

CaptureMode resolveCaptureMode(const TraceArgs& args);

/**
 * The engines an adaptive run SELECTS, and when it arms them.
 *
 * Deliberately NOT `ProfilingEngine::Deep`. That enum means "the deepest
 * analysis this GPU supports" and picks SASS-or-PC plus PM; it does not
 * guarantee the base Trace activity that `recent_kernel_ms` and
 * `kernel_launch_rate` are computed from, and its base policy varies with the
 * path chosen. An adaptive run needs the base pinned.
 */
struct AdaptiveCapturePlan {
    // Always on for the whole run. Kernel-timing conditions need it.
    std::string base = "Trace";
    // Selected by the launcher. The engine prepares after the first valid CUDA
    // context exists and remains idle until a window opens.
    std::vector<std::string> selected_deep;
    bool arm_window_only = true;
};

/** The plan for an adaptive run. Empty selected_deep if mode is explicit. */
AdaptiveCapturePlan resolveAdaptivePlan(const TraceArgs& args);

// Resolves the ordered capture plan (one isolated CUPTI engine per pass) from
// parsed trace args. Precedence:
//   1. explicit --passes    -> the listed engines, one pass each (Deep is the
//                             Deep engine, not an expansion);
//   2. any --deep-* flag    -> a single adaptive pass, engines chosen by
//                             resolveAdaptivePlan;
//   3. otherwise            -> a single Trace pass.
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

// Parses the args passed to `gpufl info`.
struct InfoParseResult {
    std::optional<InfoArgs> args;
    std::string error;
};

InfoParseResult parseInfoArgs(const std::vector<std::string>& argv);

// Help text printed for `gpufl --help`, `gpufl trace --help`,
// `gpufl upload --help`, `gpufl monitor --help`, and `gpufl info --help`.
const char* topLevelHelp();
const char* traceHelp();
const char* uploadHelp();
const char* monitorHelp();
const char* infoHelp();

}  // namespace gpufl::launcher

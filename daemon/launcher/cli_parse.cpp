#include "cli_parse.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <random>
#include <sstream>

namespace gpufl::launcher {

namespace {

// argv[i] could be "--flag" or "--flag=value" or "-f". Pull the value
// out of "--flag=value" form; for "--flag value" the caller advances.
struct FlagBreak {
    std::string key;
    std::optional<std::string> inline_value;
};
FlagBreak splitFlag(const std::string& tok) {
    auto eq = tok.find('=');
    if (eq == std::string::npos) return {tok, std::nullopt};
    return {tok.substr(0, eq), tok.substr(eq + 1)};
}

// Canonical engine names accepted by each --passes token.
// Must match the set gpufl::init() parses for GPUFL_PROFILING_ENGINE (see
// gpufl.cpp). The launcher only validates + forwards verbatim; init() is the
// single string→enum parser, so this is the one launcher-side copy to keep in
// sync with the ProfilingEngine ladder.
constexpr const char* kEngines[] = {
    "Trace", "PcSampling", "SassMetrics",
    "PmSampling", "RangeProfiler", "RangeProfilerKernelReplay", "Deep"};
bool isValidEngine(const std::string& e) {
    for (auto* k : kEngines) if (e == k) return true;
    return false;
}

// Trim ASCII spaces/tabs from both ends - lets `--passes Trace, SassMetrics`
// (with spaces after commas) parse the same as the no-space form.
std::string trim(const std::string& s) {
    const auto b = s.find_first_not_of(" \t");
    if (b == std::string::npos) return "";
    const auto e = s.find_last_not_of(" \t");
    return s.substr(b, e - b + 1);
}

// Parse a duration into milliseconds: "500ms", "30s", "5m", "2h", or a bare
// number (interpreted as seconds, e.g. "60" == 60s). Rejects garbage and
// negative values.
bool parseDurationMs(const std::string& s, int64_t& out_ms) {
    if (s.empty()) return false;
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || v < 0) return false;
    std::string unit = trim(end);
    double mult_ms;  // value * mult_ms = milliseconds
    if (unit.empty() || unit == "s") mult_ms = 1000.0;
    else if (unit == "ms") mult_ms = 1.0;
    else if (unit == "m") mult_ms = 60.0 * 1000.0;
    else if (unit == "h") mult_ms = 60.0 * 60.0 * 1000.0;
    else return false;
    out_ms = static_cast<int64_t>(v * mult_ms);
    return true;
}

// Validate one --passes token. A token may be a single engine ("Trace") or a
// '+'-joined composite group ("Trace+PcSampling") that runs those engines
// together in ONE process. Returns "" if valid, else an error message.
std::string validatePassToken(const std::string& token) {
    std::vector<std::string> parts;
    size_t start = 0;
    while (true) {
        const size_t plus = token.find('+', start);
        parts.push_back(trim(token.substr(
            start, plus == std::string::npos ? std::string::npos : plus - start)));
        if (plus == std::string::npos) break;
        start = plus + 1;
    }
    const bool composite = parts.size() > 1;
    for (const std::string& p : parts) {
        if (p.empty()) {
            return "empty engine in --passes group '" + token +
                   "' (expected e.g. Trace+PcSampling)";
        }
        if (!isValidEngine(p)) {
            return "invalid --passes engine: " + p +
                   " (expected a comma-separated list of: Trace | PcSampling | "
                   "SassMetrics | PmSampling | RangeProfiler | "
                   "RangeProfilerKernelReplay | Deep; join engines with + to run "
                   "them in one process, e.g. Trace+PcSampling)";
        }
        if (composite && p == "Deep") {
            return "Deep cannot be combined in a '+' group (it already runs "
                   "PcSampling + SassMetrics together); give it its own pass";
        }
        if (composite && p == "SassMetrics") {
            return "SassMetrics cannot share a process (it deadlocks with kernel "
                   "tracing); give it its own pass with a comma, e.g. "
                   "Trace+PcSampling,SassMetrics";
        }
    }
    return "";
}

}  // namespace

const char* topLevelHelp() {
    return
        "gpufl - GPUFlight launcher\n"
        "\n"
        "USAGE:\n"
        "    gpufl <SUBCOMMAND> [OPTIONS]\n"
        "\n"
        "SUBCOMMANDS:\n"
        "    trace      Inject GPUFlight into a target process and capture telemetry\n"
        "    monitor    Run long-lived GPU/host telemetry collection\n"
        "    info       Print local GPU device capabilities\n"
        "    upload     Upload a captured session's NDJSON logs to the backend\n"
        "    version    Print version + build info\n"
        "\n"
        "Run `gpufl <subcommand> --help` for subcommand-specific help.\n";
}

const char* traceHelp() {
    return
        "gpufl trace - Capture telemetry from a target process\n"
        "\n"
        "USAGE:\n"
        "    gpufl trace [OPTIONS] -- <COMMAND>...\n"
        "\n"
        "OPTIONS:\n"
        "    -n, --name=<NAME>       Session name (default: basename of <COMMAND>)\n"
        "    -o, --output=<DIR>      Local NDJSON output dir\n"
        "                            (default: ~/.gpufl/traces/{ts}_{session_id}/)\n"
        "        --passes=<LIST>     Capture pass list: comma-separated values from:\n"
        "                            Trace | PcSampling | SassMetrics | PmSampling |\n"
        "                            RangeProfiler | RangeProfilerKernelReplay | Deep\n"
        "                            Each comma is a separate pass (relaunch). Join\n"
        "                            engines with + to run them in ONE process, e.g.\n"
        "                            Trace+PcSampling (timeline + PC stalls, one run).\n"
        "                            Default: Trace. Deep runs PcSampling+SassMetrics\n"
        "                            in one pass (same as the embedded Deep engine);\n"
        "                            for timeline+stalls+SASS list passes explicitly,\n"
        "                            e.g. Trace,PcSampling,SassMetrics. SassMetrics\n"
        "                            must be its own pass (deadlocks if shared). Use\n"
        "                            gpufl monitor for monitoring-only telemetry.\n"
        "                            PcSampling / PM / Range passes may need NVIDIA\n"
        "                            performance-counter access.\n"
        "    -q, --quiet             Suppress launcher chatter (errors still printed)\n"
        "    -v, --verbose           Verbose launcher logging\n"
        "        --upload            Start gpufl-agent as the live uploader\n"
        "        --backend-url=<URL> Backend base URL for --upload\n"
        "                            Env fallback: GPUFL_BACKEND_URL\n"
        "        --api-key=<KEY>     Bearer token for --upload\n"
        "                            Env fallback: GPUFL_API_KEY\n"
        "        --api-version=<VER> Agent HTTP API version. Default: v1\n"
        "        --agent-jar=<PATH>  Run agent as `java -jar <PATH>`\n"
        "                            Env fallback: GPUFL_AGENT_JAR\n"
        "        --agent-cursor=<P>  Agent cursor file. Default: <output>/cursor.json\n"
        "        --log-types=<LIST>  Agent channels to upload. Default: device,scope,system,sass\n"
        "        --agent-drain-ms=<MS>\n"
        "                            Max wait for the agent to finish uploading before\n"
        "                            stopping it (it exits on its own when done). Default: 60000\n"
        "        --segment-every=<DUR>\n"
        "                            Split a long run on this cadence (minimum: 60s).\n"
        "                            Example: --segment-every=5m. Default: off\n"
        "        --segment-max-rows=<N>\n"
        "                            Also split after this many logical telemetry rows.\n"
        "                            The batch crossing N stays in the old segment.\n"
        "                            Default: off. Both segment flags are staged and\n"
        "                            rejected until SegmentCoordinator lands.\n"
        "        --warmup=<DUR>      Skip cold start: defer capture by this long\n"
        "                            (e.g. 30s, 500ms, 5m; bare number = seconds)\n"
        "        --window=<DUR>      Bounded window: capture this long after warmup,\n"
        "                            then STOP the target. For servers that never\n"
        "                            exit. Omit to run to the target's own exit.\n"
        "        --window-timeout=<DUR>\n"
        "                            Hard cap on total target runtime (safety).\n"
        "        --after-window=<WHAT>\n"
        "                            What to do at window end. Only 'stop' today.\n"
        "    A deep window is ONE adaptive run: gpufl picks the deep engine, so\n"
        "    --deep-* cannot be combined with --passes. Which engine was selected\n"
        "    is printed at startup rather than promised here - it depends on the\n"
        "    GPU, and today only PM sampling is selected.\n"
        "\n"
        "        --deep-after=<DUR>  Arm the deep engine this long into the run,\n"
        "                            then disarm. Unlike --window the target keeps\n"
        "                            running. Needs a bound below. Default: 0 (arm\n"
        "                            at the first kernel launch).\n"
        "        --deep-for=<DUR>    How long the deep window stays armed. Note this\n"
        "                            bounds TIME, which does not bound how much the\n"
        "                            engines actually collect - see --deep-launches.\n"
        "        --deep-when=<EXPR>  Open the window when a metric crosses a threshold,\n"
        "                            e.g. \"custom.token_rate<1000 for 2s\". This and\n"
        "                            --deep-after are two answers to the same\n"
        "                            question, so pass one or the other.\n"
        "        --deep-launches=<N> Kernel-launch bound on the deep window; ends it\n"
        "                            at whichever bound is hit first. PREFER THIS:\n"
        "                            wall time does not bound how much an engine\n"
        "                            collects, and how far one second of it goes\n"
        "                            differs by more than an order of magnitude\n"
        "                            between engines.\n"
        "        --deep-cooldown=<DUR>\n"
        "                            Quiet time before another window may open.\n"
        "        --pc-sample-period=<N>\n"
        "                            PC sampling period: log2 of GPU cycles per sample\n"
        "                            (5..31; default 10). Lower = more frequent — for\n"
        "                            short kernels that yield no PC samples by default.\n"
        "    -h, --help              Print this help\n"
        "\n"
        "EXAMPLES:\n"
        "    gpufl trace -- python train.py\n"
        "    gpufl trace --name=quantize -- ./inference_server\n"
        "    gpufl trace --passes=Trace,PmSampling -- python train.py\n"
        "    gpufl trace --passes=Deep -- python train.py        # multi-pass\n"
        "    gpufl trace --passes=Trace,SassMetrics -- ./app     # custom plan\n"
        "    gpufl trace --passes=Trace+PcSampling -- ./app      # one-process composite\n"
        "    gpufl trace --passes=Trace+PcSampling --warmup=60s --window=5m -- ./serve\n";
}

ParsedTopLevel parseTopLevel(int argc, char** argv) {
    ParsedTopLevel out;
    if (argc < 2) {
        out.sub = Subcommand::Help;
        return out;
    }
    std::string first = argv[1];
    if (first == "-h" || first == "--help") {
        out.sub = Subcommand::Help;
    } else if (first == "-V" || first == "--version" || first == "version") {
        out.sub = Subcommand::Version;
    } else if (first == "trace") {
        out.sub = Subcommand::Trace;
    } else if (first == "upload") {
        out.sub = Subcommand::Upload;
    } else if (first == "monitor") {
        out.sub = Subcommand::Monitor;
    } else if (first == "info") {
        out.sub = Subcommand::Info;
    } else {
        out.sub = Subcommand::Unknown;
        out.remaining.push_back(first);
        return out;
    }
    for (int i = 2; i < argc; ++i) out.remaining.emplace_back(argv[i]);
    return out;
}

TraceParseResult parseTraceArgs(const std::vector<std::string>& argv) {
    TraceArgs out;
    bool seen_dash_dash = false;
    auto parseNonNegativeInt = [](const std::string& s, int& slot) -> bool {
        if (s.empty()) return false;
        char* end = nullptr;
        long v = std::strtol(s.c_str(), &end, 10);
        if (*end != '\0' || v < 0) return false;
        slot = static_cast<int>(v);
        return true;
    };

    for (size_t i = 0; i < argv.size(); ++i) {
        const std::string& tok = argv[i];
        if (!seen_dash_dash && tok == "--") {
            seen_dash_dash = true;
            continue;
        }
        if (seen_dash_dash) {
            out.command.push_back(tok);
            continue;
        }
        if (tok == "-h" || tok == "--help") {
            // Caller prints help; signal via empty error + no command.
            return {std::nullopt, "__help__"};
        }
        if (tok == "-v" || tok == "--verbose") { out.verbose = true; continue; }
        if (tok == "-q" || tok == "--quiet")   { out.quiet = true; continue; }
        if (tok == "--upload")                 { out.upload = true; continue; }

        auto fb = splitFlag(tok);
        const std::string& key = fb.key;
        auto take_value = [&](std::string& slot) -> std::string {
            if (fb.inline_value) { slot = *fb.inline_value; return ""; }
            if (i + 1 >= argv.size()) return "missing value for " + key;
            slot = argv[++i];
            return "";
        };

        if (key == "-n" || key == "--name") {
            auto err = take_value(out.name);
            if (!err.empty()) return {std::nullopt, err};
        } else if (key == "-o" || key == "--output") {
            auto err = take_value(out.output_dir);
            if (!err.empty()) return {std::nullopt, err};
        } else if (key == "--profile") {
            std::string ignored;
            auto err = take_value(ignored);
            if (!err.empty()) return {std::nullopt, err};
            return {std::nullopt,
                    "`gpufl trace --profile` has been removed; use --passes=Trace, "
                    "--passes=Deep, or `gpufl monitor` for monitoring-only telemetry"};
        } else if (key == "--engine") {
            std::string ignored;
            auto err = take_value(ignored);
            if (!err.empty()) return {std::nullopt, err};
            return {std::nullopt,
                    "`gpufl trace --engine` has been removed; use --passes=Trace, "
                    "--passes=Deep, or an explicit list like --passes=Trace,PmSampling"};
        } else if (key == "--passes") {
            std::string v;
            auto err = take_value(v);
            if (!err.empty()) return {std::nullopt, err};
            // Comma-separated pass list -> one isolated pass each. A token may
            // be a single engine, or a '+'-joined group ("Trace+PcSampling")
            // that runs those engines together in one process (a composite).
            out.passes.clear();
            size_t start = 0;
            while (true) {
                const size_t comma = v.find(',', start);
                const std::string item = trim(v.substr(
                    start,
                    comma == std::string::npos ? std::string::npos : comma - start));
                if (!item.empty()) {
                    const std::string perr = validatePassToken(item);
                    if (!perr.empty()) return {std::nullopt, perr};
                    out.passes.push_back(item);
                }
                if (comma == std::string::npos) break;
                start = comma + 1;
            }
            if (out.passes.empty()) {
                return {std::nullopt, "--passes requires at least one engine"};
            }
        } else if (key == "--backend-url") {
            auto err = take_value(out.backend_url);
            if (!err.empty()) return {std::nullopt, err};
        } else if (key == "--api-key") {
            auto err = take_value(out.api_key);
            if (!err.empty()) return {std::nullopt, err};
        } else if (key == "--api-version") {
            auto err = take_value(out.api_version);
            if (!err.empty()) return {std::nullopt, err};
            if (out.api_version.empty()) return {std::nullopt, "--api-version cannot be empty"};
        } else if (key == "--agent-jar") {
            auto err = take_value(out.agent_jar);
            if (!err.empty()) return {std::nullopt, err};
            if (out.agent_jar.empty()) return {std::nullopt, "--agent-jar cannot be empty"};
        } else if (key == "--agent-cursor") {
            auto err = take_value(out.agent_cursor);
            if (!err.empty()) return {std::nullopt, err};
            if (out.agent_cursor.empty()) return {std::nullopt, "--agent-cursor cannot be empty"};
        } else if (key == "--log-types") {
            auto err = take_value(out.log_types);
            if (!err.empty()) return {std::nullopt, err};
            if (out.log_types.empty()) return {std::nullopt, "--log-types cannot be empty"};
        } else if (key == "--agent-drain-ms") {
            std::string v;
            auto err = take_value(v);
            if (!err.empty()) return {std::nullopt, err};
            if (!parseNonNegativeInt(v, out.agent_drain_ms)) {
                return {std::nullopt,
                        "invalid --agent-drain-ms value: " + v +
                        " (expected a non-negative integer, milliseconds)"};
            }
        } else if (key == "--segment-every") {
            std::string v;
            auto err = take_value(v);
            if (!err.empty()) return {std::nullopt, err};
            if (!parseDurationMs(v, out.segment_every_ms)) {
                return {std::nullopt,
                        "invalid --segment-every value: " + v +
                        " (expected a duration like 60s, 5m, 1h, or a bare "
                        "number of seconds)"};
            }
            if (out.segment_every_ms > 0 &&
                out.segment_every_ms < kMinSegmentEveryMs) {
                return {std::nullopt,
                        "--segment-every must be at least 60s; shorter cadences "
                        "can create a session storm"};
            }
        } else if (key == "--segment-max-rows") {
            std::string v;
            auto err = take_value(v);
            if (!err.empty()) return {std::nullopt, err};
            if (v.empty() || v.front() == '-') {
                return {std::nullopt,
                        "invalid --segment-max-rows value: " + v +
                        " (expected a non-negative integer; 0 disables it)"};
            }
            char* end = nullptr;
            errno = 0;
            const unsigned long long n = std::strtoull(v.c_str(), &end, 10);
            if (end == v.c_str() || (end && *end != '\0') || errno == ERANGE) {
                return {std::nullopt,
                        "invalid --segment-max-rows value: " + v +
                        " (expected a non-negative integer; 0 disables it)"};
            }
            out.segment_max_rows = static_cast<uint64_t>(n);
        } else if (key == "--pc-sample-period") {
            std::string v;
            auto err = take_value(v);
            if (!err.empty()) return {std::nullopt, err};
            int period = 0;
            if (!parseNonNegativeInt(v, period) || period < 5 || period > 31) {
                return {std::nullopt,
                        "invalid --pc-sample-period value: " + v +
                        " (expected an integer 5..31; the log2 of GPU cycles per "
                        "PC sample - lower = more frequent, catches short kernels)"};
            }
            out.pc_sample_period = static_cast<uint32_t>(period);
        } else if (key == "--warmup") {
            std::string v;
            auto err = take_value(v);
            if (!err.empty()) return {std::nullopt, err};
            if (!parseDurationMs(v, out.warmup_ms)) {
                return {std::nullopt,
                        "invalid --warmup value: " + v +
                        " (expected a duration like 30s, 500ms, 5m, 1h, "
                        "or a bare number of seconds)"};
            }
        } else if (key == "--window") {
            std::string v;
            auto err = take_value(v);
            if (!err.empty()) return {std::nullopt, err};
            if (!parseDurationMs(v, out.window_ms)) {
                return {std::nullopt,
                        "invalid --window value: " + v +
                        " (expected a duration like 30s, 500ms, 5m, 1h, "
                        "or a bare number of seconds)"};
            }
        } else if (key == "--window-timeout") {
            std::string v;
            auto err = take_value(v);
            if (!err.empty()) return {std::nullopt, err};
            if (!parseDurationMs(v, out.window_timeout_ms)) {
                return {std::nullopt,
                        "invalid --window-timeout value: " + v +
                        " (expected a duration like 30s, 5m, 1h, "
                        "or a bare number of seconds)"};
            }
        } else if (key == "--deep-after" || key == "--deep-for" ||
                   key == "--deep-cooldown") {
            std::string v;
            auto err = take_value(v);
            if (!err.empty()) return {std::nullopt, err};
            int64_t ms = 0;
            if (!parseDurationMs(v, ms)) {
                return {std::nullopt,
                        "invalid " + key + " value: " + v +
                        " (expected a duration like 30s, 500ms, 5m, 1h, "
                        "or a bare number of seconds)"};
            }
            // A bare number is seconds, which collides with --deep-launches:
            // `--deep-for=2000` meaning "2000 launches" silently becomes a
            // 33-minute window, and the no-bound check below can't catch it
            // because a bound *was* given. Reject the unit-less form once it
            // is too large to plausibly be a duration someone typed on
            // purpose - naming the alternative, since that is the mistake.
            const bool unitless =
                !v.empty() &&
                v.find_first_not_of("0123456789") == std::string::npos;
            if (key == "--deep-for" && unitless && ms >= 600'000) {
                return {std::nullopt,
                        "--deep-for=" + v + " means " + std::to_string(ms / 1000) +
                        " SECONDS (a bare number is seconds), which is almost "
                        "certainly not what you meant. Add a unit (e.g. " + v +
                        "s, 5m) if you really want that long a window, or use "
                        "--deep-launches " + v + " to bound it by kernel "
                        "launches instead"};
            }
            if (key == "--deep-after") {
                out.deep_after_ms = ms;
                out.deep_after_set = true;
            }
            else if (key == "--deep-for")      out.deep_for_ms = ms;
            else                               out.deep_cooldown_ms = ms;
            out.deep_requested = true;
        } else if (key == "--deep-when") {
            std::string v;
            auto err = take_value(v);
            if (!err.empty()) return {std::nullopt, err};
            if (v.empty()) {
                return {std::nullopt,
                        "--deep-when needs an expression, e.g. "
                        "--deep-when=\"custom.token_rate<1000 for 2s\""};
            }
            // Parsed by the client, not here: telling a misspelled built-in
            // from a custom counter that has simply not registered yet needs
            // the metric registry, and duplicating that check would give two
            // places to disagree about what a metric name means.
            out.deep_when = v;
            out.deep_requested = true;
        } else if (key == "--deep-launches") {
            std::string v;
            auto err = take_value(v);
            if (!err.empty()) return {std::nullopt, err};
            char* end = nullptr;
            const unsigned long long n = std::strtoull(v.c_str(), &end, 10);
            if (end == v.c_str() || (end && *end != '\0') || n == 0) {
                return {std::nullopt,
                        "invalid --deep-launches value: " + v +
                        " (expected a positive number of kernel launches)"};
            }
            out.deep_launches = static_cast<uint64_t>(n);
            out.deep_requested = true;
        } else if (key == "--after-window") {
            auto err = take_value(out.after_window);
            if (!err.empty()) return {std::nullopt, err};
            if (out.after_window == "keep") {
                return {std::nullopt,
                        "--after-window=keep is not yet implemented; the launcher "
                        "stops the target at window end (restart it with a script)"};
            }
            if (out.after_window != "stop") {
                return {std::nullopt,
                        "invalid --after-window value: " + out.after_window +
                        " (expected: stop)"};
            }
        } else {
            // A non-flag token before `--` is almost certainly the
            // caller forgetting the splitter, e.g. `gpufl trace python
            // train.py`. Distinguish that from a real typo on a flag.
            if (!tok.empty() && tok[0] != '-') {
                return {std::nullopt, "missing `--` separator before command"};
            }
            return {std::nullopt, "unknown flag: " + key};
        }
    }
    if (!seen_dash_dash) {
        return {std::nullopt, "missing `--` separator before command"};
    }
    if (out.command.empty()) {
        return {std::nullopt, "no command specified after `--`"};
    }
    // Explicit passes and deep windows are different execution models, and
    // mixing them cannot mean anything coherent: the deep engines have to be
    // fixed before the first CUDA call, so a --passes list either already
    // contains what the window would arm (making the flag redundant) or does
    // not (making the window arm nothing - which is what
    // `--passes=Trace --deep-after=30s` silently did). Rejected before the
    // target is launched, whichever order the flags came in.
    if (const std::string mode_error = validateTraceExecutionMode(out);
        !mode_error.empty()) {
        return {std::nullopt, mode_error};
    }

    // Two answers to "when does the window open" is one too many. Measured on
    // the 3090: with both set, the scheduled window opens at t=0, the rule's
    // request is refused as busy, and the rule then waits for a rearm that a
    // still-busy workload never gives - it reported `never_true` for a
    // condition that held the whole run.
    if (!out.deep_when.empty() && out.deep_after_set) {
        return {std::nullopt,
                "--deep-when and --deep-after are two different triggers for "
                "the same window. Pass --deep-when to open it on a metric, or "
                "--deep-after to open it at a fixed time"};
    }

    // A deep window with neither bound would arm and never disarm, which is
    // just "profile deeply for the whole run" with extra steps.
    if (out.deep_requested && out.deep_for_ms == 0 && out.deep_launches == 0) {
        return {std::nullopt,
                "a deep window needs a bound: pass --deep-launches <n> or "
                "--deep-for <duration> (prefer --deep-launches: it is what "
                "the engines actually scale with. The replay engines cover "
                "far less work per second of wall time, and PC sampling "
                "returns nothing at all below a few thousand launches)"};
    }
    return {out, ""};
}

const char* uploadHelp() {
    return
        "gpufl upload - Upload a captured session's NDJSON logs to the backend\n"
        "\n"
        "USAGE:\n"
        "    gpufl upload <LOG_PATH> [OPTIONS]\n"
        "\n"
        "ARGS:\n"
        "    <LOG_PATH>              Output directory written by `gpufl trace`, or\n"
        "                            the InitOptions log_path directory. Looks for\n"
        "                            '<LOG_PATH>/<session_id>/<channel>.log[.gz]'.\n"
        "                            A trace dir works directly:\n"
        "                            e.g. ~/.gpufl/traces/20260603-101500_ab12cd34\n"
        "\n"
        "OPTIONS:\n"
        "        --backend-url=<URL> Backend base URL.   Env: GPUFL_BACKEND_URL\n"
        "        --api-key=<KEY>     Bearer token.       Env: GPUFL_API_KEY\n"
        "        --api-path=<PATH>   Reverse-proxy mount. Defaults to /api/v1\n"
        "        --agent-jar=<PATH>  Run the uploader as `java -jar <PATH>`.\n"
        "                            Env: GPUFL_AGENT_JAR (else gpufl-agent on PATH)\n"
        "        --timeout=<SECS>    Cap on waiting for the upload to finish. Default 300\n"
        "        --retries=<N>       Accepted for compatibility; the agent retries internally\n"
        "    -q, --quiet             Suppress periodic progress lines\n"
        "        --all-sessions      Upload every session in the dir (this is the default)\n"
        "        --force             Re-upload even if the cursor says it shipped\n"
        "    -h, --help              Print this help\n"
        "\n"
        "EXAMPLES:\n"
        "    gpufl upload ~/.gpufl/traces/20260603-101500_ab12cd34\n"
        "    gpufl upload ./logs --force\n"
        "    GPUFL_API_KEY=gpfl_… GPUFL_BACKEND_URL=https://api.gpuflight.com \\\n"
        "        gpufl upload ./logs\n";
}

const char* monitorHelp() {
    return
        "gpufl monitor - Run long-lived GPU/host telemetry collection\n"
        "\n"
        "USAGE:\n"
        "    gpufl monitor [OPTIONS]\n"
        "\n"
        "OPTIONS:\n"
        "    -n, --name=<NAME>       Monitor session name. Default: gpufl-monitor\n"
        "    -o, --output=<DIR>      Local NDJSON output dir\n"
        "                            (default: ~/.gpufl/monitor/{ts}_{session_id}/)\n"
        "        --interval=<MS>     Sampling interval in milliseconds. Default: 5000\n"
        "        --upload            Start gpufl-agent as the live uploader\n"
        "        --backend-url=<URL> Backend base URL for --upload\n"
        "                            Env fallback: GPUFL_BACKEND_URL\n"
        "        --api-key=<KEY>     Bearer token for --upload\n"
        "                            Env fallback: GPUFL_API_KEY\n"
        "        --api-version=<VER> Agent HTTP API version. Default: v1\n"
        "        --agent-jar=<PATH>  Run agent as `java -jar <PATH>`\n"
        "                            Env fallback: GPUFL_AGENT_JAR\n"
        "        --agent-cursor=<P>  Agent cursor file. Default: <output>/cursor.json\n"
        "        --log-types=<LIST>  Agent channels to upload. Default: system\n"
        "    -q, --quiet             Suppress launcher chatter\n"
        "    -v, --verbose           Verbose launcher logging\n"
        "    -h, --help              Print this help\n"
        "\n"
        "EXAMPLES:\n"
        "    gpufl monitor\n"
        "    gpufl monitor --interval=1000\n"
        "    gpufl monitor --name=llm-node-1 --upload\n";
}

const char* infoHelp() {
    return
        "gpufl info - Print local GPU device capabilities\n"
        "\n"
        "USAGE:\n"
        "    gpufl info [OPTIONS]\n"
        "\n"
        "OPTIONS:\n"
        "        --json              Emit stable machine-readable JSON\n"
        "        --device=<ID>       Limit output to one zero-based device ID\n"
        "    -h, --help              Print this help\n"
        "\n"
        "EXAMPLES:\n"
        "    gpufl info\n"
        "    gpufl info --json\n"
        "    gpufl info --device=0 --json\n";
}

UploadParseResult parseUploadArgs(const std::vector<std::string>& argv) {
    UploadArgs out;
    bool have_log_path = false;

    auto parseInt = [](const std::string& s, int& slot) -> bool {
        if (s.empty()) return false;
        char* end = nullptr;
        long v = std::strtol(s.c_str(), &end, 10);
        if (*end != '\0' || v < 0) return false;
        slot = static_cast<int>(v);
        return true;
    };

    for (size_t i = 0; i < argv.size(); ++i) {
        const std::string& tok = argv[i];
        if (tok == "-h" || tok == "--help") return {std::nullopt, "__help__"};
        if (tok == "-q" || tok == "--quiet")   { out.quiet = true; continue; }
        if (tok == "--all-sessions")           { out.all_sessions = true; continue; }
        if (tok == "--force")                  { out.force = true; continue; }

        auto fb = splitFlag(tok);
        const std::string& key = fb.key;
        auto take_value = [&](std::string& slot) -> std::string {
            if (fb.inline_value) { slot = *fb.inline_value; return ""; }
            if (i + 1 >= argv.size()) return "missing value for " + key;
            slot = argv[++i];
            return "";
        };

        if (key == "--backend-url") {
            auto err = take_value(out.backend_url);
            if (!err.empty()) return {std::nullopt, err};
        } else if (key == "--api-key") {
            auto err = take_value(out.api_key);
            if (!err.empty()) return {std::nullopt, err};
        } else if (key == "--api-path") {
            auto err = take_value(out.api_path);
            if (!err.empty()) return {std::nullopt, err};
        } else if (key == "--agent-jar") {
            auto err = take_value(out.agent_jar);
            if (!err.empty()) return {std::nullopt, err};
        } else if (key == "--session-id") {
            return {std::nullopt,
                    "--session-id is no longer supported; point <LOG_PATH> at a "
                    "directory containing only that session"};
        } else if (key == "--timeout") {
            std::string v;
            auto err = take_value(v);
            if (!err.empty()) return {std::nullopt, err};
            if (!parseInt(v, out.timeout_s)) {
                return {std::nullopt, "invalid --timeout value: " + v +
                                      " (expected a non-negative integer, seconds)"};
            }
        } else if (key == "--retries") {
            std::string v;
            auto err = take_value(v);
            if (!err.empty()) return {std::nullopt, err};
            if (!parseInt(v, out.retries)) {
                return {std::nullopt, "invalid --retries value: " + v +
                                      " (expected a non-negative integer)"};
            }
        } else if (!tok.empty() && tok[0] == '-') {
            return {std::nullopt, "unknown flag: " + key};
        } else {
            // Bare token → the positional <LOG_PATH>. Only one allowed.
            if (have_log_path) {
                return {std::nullopt, "unexpected extra argument: " + tok +
                                      " (only one <LOG_PATH> is accepted)"};
            }
            out.log_path = tok;
            have_log_path = true;
        }
    }

    if (!have_log_path) {
        return {std::nullopt, "missing <LOG_PATH> (the trace output directory)"};
    }
    return {out, ""};
}

MonitorParseResult parseMonitorArgs(const std::vector<std::string>& argv) {
    MonitorArgs out;

    auto parsePositiveInt = [](const std::string& s, int& slot) -> bool {
        if (s.empty()) return false;
        char* end = nullptr;
        long v = std::strtol(s.c_str(), &end, 10);
        if (*end != '\0' || v <= 0) return false;
        slot = static_cast<int>(v);
        return true;
    };

    for (size_t i = 0; i < argv.size(); ++i) {
        const std::string& tok = argv[i];
        if (tok == "-h" || tok == "--help") return {std::nullopt, "__help__"};
        if (tok == "-v" || tok == "--verbose") { out.verbose = true; continue; }
        if (tok == "-q" || tok == "--quiet")   { out.quiet = true; continue; }
        if (tok == "--upload")                 { out.upload = true; continue; }

        auto fb = splitFlag(tok);
        const std::string& key = fb.key;
        auto take_value = [&](std::string& slot) -> std::string {
            if (fb.inline_value) { slot = *fb.inline_value; return ""; }
            if (i + 1 >= argv.size()) return "missing value for " + key;
            slot = argv[++i];
            return "";
        };

        if (key == "-n" || key == "--name") {
            auto err = take_value(out.name);
            if (!err.empty()) return {std::nullopt, err};
            if (out.name.empty()) return {std::nullopt, "--name cannot be empty"};
        } else if (key == "-o" || key == "--output") {
            auto err = take_value(out.output_dir);
            if (!err.empty()) return {std::nullopt, err};
            if (out.output_dir.empty()) return {std::nullopt, "--output cannot be empty"};
        } else if (key == "--interval") {
            std::string v;
            auto err = take_value(v);
            if (!err.empty()) return {std::nullopt, err};
            if (!parsePositiveInt(v, out.interval_ms)) {
                return {std::nullopt,
                        "invalid --interval value: " + v +
                        " (expected a positive integer, milliseconds)"};
            }
        } else if (key == "--backend-url") {
            auto err = take_value(out.backend_url);
            if (!err.empty()) return {std::nullopt, err};
        } else if (key == "--api-key") {
            auto err = take_value(out.api_key);
            if (!err.empty()) return {std::nullopt, err};
        } else if (key == "--api-version") {
            auto err = take_value(out.api_version);
            if (!err.empty()) return {std::nullopt, err};
            if (out.api_version.empty()) return {std::nullopt, "--api-version cannot be empty"};
        } else if (key == "--agent-jar") {
            auto err = take_value(out.agent_jar);
            if (!err.empty()) return {std::nullopt, err};
            if (out.agent_jar.empty()) return {std::nullopt, "--agent-jar cannot be empty"};
        } else if (key == "--agent-cursor") {
            auto err = take_value(out.agent_cursor);
            if (!err.empty()) return {std::nullopt, err};
            if (out.agent_cursor.empty()) return {std::nullopt, "--agent-cursor cannot be empty"};
        } else if (key == "--log-types") {
            auto err = take_value(out.log_types);
            if (!err.empty()) return {std::nullopt, err};
            if (out.log_types.empty()) return {std::nullopt, "--log-types cannot be empty"};
        } else if (!tok.empty() && tok[0] == '-') {
            return {std::nullopt, "unknown flag: " + key};
        } else {
            return {std::nullopt,
                    "unexpected argument: " + tok +
                    " (`gpufl monitor` does not launch a target process; use `gpufl trace -- <cmd>`)"};
        }
    }

    return {out, ""};
}

InfoParseResult parseInfoArgs(const std::vector<std::string>& argv) {
    InfoArgs out;

    for (size_t i = 0; i < argv.size(); ++i) {
        const std::string& tok = argv[i];
        if (tok == "-h" || tok == "--help") {
            return {std::nullopt, "__help__"};
        }
        if (tok == "--json") {
            out.json = true;
            continue;
        }

        auto fb = splitFlag(tok);
        if (fb.key == "--device") {
            std::string value;
            if (fb.inline_value) {
                value = *fb.inline_value;
            } else if (i + 1 < argv.size()) {
                value = argv[++i];
            } else {
                return {std::nullopt, "missing value for --device"};
            }

            if (value.empty()) {
                return {std::nullopt, "invalid --device value: expected a non-negative integer"};
            }
            char* end = nullptr;
            const long parsed = std::strtol(value.c_str(), &end, 10);
            if (*end != '\0' || parsed < 0) {
                return {std::nullopt,
                        "invalid --device value: " + value +
                        " (expected a non-negative integer)"};
            }
            out.device_id = static_cast<int>(parsed);
        } else if (!tok.empty() && tok[0] == '-') {
            return {std::nullopt, "unknown flag: " + fb.key};
        } else {
            return {std::nullopt,
                    "unexpected argument: " + tok +
                    " (`gpufl info` does not accept positional arguments)"};
        }
    }

    return {out, ""};
}

std::string validateTraceExecutionMode(const TraceArgs& args) {
    if (args.deep_requested && !args.passes.empty()) {
        return "--passes cannot be combined with --deep-* flags.\n"
               "\n"
               "  --passes runs the engines you name, relaunching the target "
               "once per pass.\n"
               "  --deep-* runs ONE adaptive pass: gpufl selects a compatible "
               "deep engine\n"
               "  and arms it only inside the window.\n"
               "\n"
               "Drop --passes to use a deep window.";
    }
    return validateTraceSegmentation(args);
}

bool segmentationRequested(const TraceArgs& args) {
    return args.segment_every_ms > 0 || args.segment_max_rows > 0;
}

std::string validateTraceSegmentation(
    const TraceArgs& args,
    const std::string& inherited_analysis_id) {
    if (args.segment_every_ms < 0) {
        return "--segment-every cannot be negative";
    }
    if (args.segment_every_ms > 0 &&
        args.segment_every_ms < kMinSegmentEveryMs) {
        return "--segment-every must be at least 60s; shorter cadences can "
               "create a session storm";
    }
    if (!segmentationRequested(args)) return {};

    if (!inherited_analysis_id.empty()) {
        return "session segmentation cannot be combined with an inherited "
               "GPUFL_ANALYSIS_ID; unset GPUFL_ANALYSIS_ID before launching "
               "the target";
    }
    if (args.passes.size() > 1) {
        return "session segmentation cannot be combined with a multi-pass "
               "--passes list; segmented runs concatenate time while analysis "
               "passes overlay the same interval";
    }
    if (!args.passes.empty()) {
        const std::string& pass = args.passes.front();
        if (pass != "Trace" && pass != "PmSampling") {
            return "session segmentation V1 supports only a single Trace or "
                   "PmSampling pass. Unsupported pass: " + pass;
        }
    }

    // No explicit pass plus --deep-* is the one supported composite: the
    // launcher pins native Trace as the base and prepares window-only PM.
    // No explicit pass and no deep flags is the ordinary single Trace pass.
    return {};
}

std::string generateRunId() {
    std::array<uint8_t, 16> bytes{};
    static thread_local std::mt19937_64 rng([] {
        std::random_device rd;
        std::seed_seq seed{
            rd(), rd(), rd(), rd(),
            static_cast<unsigned>(
                std::chrono::steady_clock::now().time_since_epoch().count())};
        return std::mt19937_64(seed);
    }());
    for (size_t i = 0; i < bytes.size(); i += sizeof(uint64_t)) {
        const uint64_t word = rng();
        for (size_t j = 0; j < sizeof(uint64_t); ++j) {
            bytes[i + j] = static_cast<uint8_t>(word >> (j * 8));
        }
    }

    bytes[6] = static_cast<uint8_t>((bytes[6] & 0x0f) | 0x40);
    bytes[8] = static_cast<uint8_t>((bytes[8] & 0x3f) | 0x80);

    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (size_t i = 0; i < bytes.size(); ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) out << '-';
        out << std::setw(2) << static_cast<unsigned>(bytes[i]);
    }
    return out.str();
}

bool segmentationRuntimeReady() {
    // Flipped only by the SegmentCoordinator slice, together with its
    // cutover/ownership/dictionary tests.
    return false;
}

CaptureMode resolveCaptureMode(const TraceArgs& args) {
    return args.deep_requested ? CaptureMode::AdaptiveDeepWindow
                               : CaptureMode::ExplicitPasses;
}

AdaptiveCapturePlan resolveAdaptivePlan(const TraceArgs& args) {
    AdaptiveCapturePlan plan;
    if (!args.deep_requested) return plan;   // selected_deep stays empty

    // Trace is pinned as the base rather than left to the deep engine's own
    // policy: kernel_launch_rate and recent_kernel_ms are computed from its
    // completed-kernel records, and a rule that silently loses its metric
    // reads as "condition never held".
    plan.base = "Trace";

    // PM only, for now. It is the one deep engine that works on a 3090 at all:
    // PC sampling fails configuration under injection there and SASS emits no
    // records, so neither has a dormant cost anyone has measured.
    //
    // Earlier overhead measurements covered PM selected-but-not-initialized,
    // because initialization still lived in the arm path at the time. They are
    // intentionally not used as a policy budget here; the prepared-and-idle
    // configuration must be measured again after the lifecycle split.
    plan.selected_deep = {"PmSampling"};
    plan.arm_window_only = true;
    return plan;
}

std::vector<std::string> resolvePassPlan(const TraceArgs& args) {
    if (!args.passes.empty()) return args.passes;

    // Exactly one pass for an adaptive run. Relaunching the target per pass is
    // what --passes is for; a window that triggers on a live condition cannot
    // be reproduced across relaunches, so splitting it would change what is
    // being measured.
    if (resolveCaptureMode(args) == CaptureMode::AdaptiveDeepWindow) {
        const AdaptiveCapturePlan plan = resolveAdaptivePlan(args);
        std::string composite = plan.base;
        for (const std::string& engine : plan.selected_deep) {
            composite += "+" + engine;
        }
        return {composite};
    }
    return {"Trace"};
}

}  // namespace gpufl::launcher

#include "cli_parse.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>

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
            return "Deep cannot be combined in a '+' group (it is a multi-pass "
                   "preset); list it on its own";
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
        "                            Default: Trace. Deep expands to isolated\n"
        "                            Trace,PcSampling,SassMetrics passes and cannot\n"
        "                            be combined with other passes. SassMetrics must\n"
        "                            be its own pass (deadlocks if shared). Use gpufl\n"
        "                            monitor for monitoring-only GPU/host telemetry.\n"
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
        "        --log-types=<LIST>  Agent channels to upload. Default: device,scope,system\n"
        "        --agent-drain-ms=<MS>\n"
        "                            Wait after target exit before stopping agent.\n"
        "                            Default: 3000\n"
        "        --warmup=<DUR>      Skip cold start: defer capture by this long\n"
        "                            (e.g. 30s, 500ms, 5m; bare number = seconds)\n"
        "        --window=<DUR>      Bounded window: capture this long after warmup,\n"
        "                            then STOP the target. For servers that never\n"
        "                            exit. Omit to run to the target's own exit.\n"
        "        --window-timeout=<DUR>\n"
        "                            Hard cap on total target runtime (safety).\n"
        "        --after-window=<WHAT>\n"
        "                            What to do at window end. Only 'stop' today.\n"
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
            bool saw_deep = false;
            size_t start = 0;
            while (true) {
                const size_t comma = v.find(',', start);
                const std::string item = trim(v.substr(
                    start,
                    comma == std::string::npos ? std::string::npos : comma - start));
                if (!item.empty()) {
                    const std::string perr = validatePassToken(item);
                    if (!perr.empty()) return {std::nullopt, perr};
                    saw_deep = saw_deep || item == "Deep";
                    out.passes.push_back(item);
                }
                if (comma == std::string::npos) break;
                start = comma + 1;
            }
            if (out.passes.empty()) {
                return {std::nullopt, "--passes requires at least one engine"};
            }
            if (saw_deep && out.passes.size() != 1) {
                return {std::nullopt,
                        "--passes=Deep is a preset and cannot be combined with other passes"};
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
        "        --timeout=<SECS>    Total wall budget for the upload. Default 300\n"
        "        --retries=<N>       Retries per failing POST. Default 1\n"
        "    -q, --quiet             Suppress periodic progress lines\n"
        "        --session-id=<ID>   Upload only this session (default: latest)\n"
        "        --all-sessions      Upload every session in the dir (excl. --session-id)\n"
        "        --force             Re-upload even if the cursor says it shipped\n"
        "    -h, --help              Print this help\n"
        "\n"
        "EXAMPLES:\n"
        "    gpufl upload ~/.gpufl/traces/20260603-101500_ab12cd34\n"
        "    gpufl upload ./logs --all-sessions\n"
        "    GPUFL_API_KEY=gpfl_… GPUFL_BACKEND_URL=https://api.gpuflight.com \\\n"
        "        gpufl upload ./logs --session-id=<uuid>\n";
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
        } else if (key == "--session-id") {
            auto err = take_value(out.session_id);
            if (!err.empty()) return {std::nullopt, err};
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
    if (!out.session_id.empty() && out.all_sessions) {
        return {std::nullopt, "--session-id and --all-sessions are mutually exclusive"};
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

std::vector<std::string> resolvePassPlan(const TraceArgs& args) {
    if (args.passes.empty()) return {"Trace"};
    if (args.passes.size() == 1 && args.passes.front() == "Deep") {
        return {"Trace", "PcSampling", "SassMetrics"};
    }
    return args.passes;
}

}  // namespace gpufl::launcher

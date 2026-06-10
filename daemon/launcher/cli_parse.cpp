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

// Canonical engine names accepted by --engine and by each --passes token.
// Must match the set gpufl::init() parses for GPUFL_PROFILING_ENGINE (see
// gpufl.cpp). The launcher only validates + forwards verbatim; init() is the
// single string→enum parser, so this is the one launcher-side copy to keep in
// sync with the ProfilingEngine ladder.
constexpr const char* kEngines[] = {
    "Monitor", "Trace", "PcSampling", "SassMetrics",
    "PmSampling", "RangeProfiler", "RangeProfilerKernelReplay", "Deep"};
bool isValidEngine(const std::string& e) {
    for (auto* k : kEngines) if (e == k) return true;
    return false;
}

// Trim ASCII spaces/tabs from both ends — lets `--passes Trace, SassMetrics`
// (with spaces after commas) parse the same as the no-space form.
std::string trim(const std::string& s) {
    const auto b = s.find_first_not_of(" \t");
    if (b == std::string::npos) return "";
    const auto e = s.find_last_not_of(" \t");
    return s.substr(b, e - b + 1);
}

}  // namespace

const char* topLevelHelp() {
    return
        "gpufl — GPUFlight launcher\n"
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
        "gpufl trace — Capture telemetry from a target process\n"
        "\n"
        "USAGE:\n"
        "    gpufl trace [OPTIONS] -- <COMMAND>...\n"
        "\n"
        "OPTIONS:\n"
        "    -n, --name <NAME>       Session name (default: basename of <COMMAND>)\n"
        "    -o, --output <DIR>      Local NDJSON output dir\n"
        "                            (default: ~/.gpufl/traces/{ts}_{session_id}/)\n"
        "        --profile <PROF>    comprehensive (default) | light | monitoring-only\n"
        "        --engine <ENG>      Override profiling engine. One of:\n"
        "                            Monitor | Trace | PcSampling | SassMetrics |\n"
        "                            PmSampling | RangeProfiler |\n"
        "                            RangeProfilerKernelReplay | Deep\n"
        "                            (Deep = the default multi-pass plan: Trace,\n"
        "                            PcSampling, SassMetrics run as isolated passes\n"
        "                            and merged by the backend.)\n"
        "        --passes <LIST>     Multi-pass run: comma-separated engines, one\n"
        "                            isolated pass each, merged by the backend.\n"
        "                            e.g. Trace,PcSampling,SassMetrics. Mutually\n"
        "                            exclusive with --engine. (PcSampling needs\n"
        "                            admin / GPU perf-counter access; that pass is\n"
        "                            reported + skipped if unprivileged.)\n"
        "    -q, --quiet             Suppress launcher chatter (errors still printed)\n"
        "    -v, --verbose           Verbose launcher logging\n"
        "        --upload            After the run, upload the trace to the\n"
        "                            GPUFlight backend (needs GPUFL_API_KEY +\n"
        "                            GPUFL_BACKEND_URL in the environment)\n"
        "    -h, --help              Print this help\n"
        "\n"
        "EXAMPLES:\n"
        "    gpufl trace -- python train.py\n"
        "    gpufl trace --name=quantize -- ./inference_server\n"
        "    gpufl trace --profile=light -- python long_train.py\n"
        "    gpufl trace --engine Deep -- python train.py        # multi-pass\n"
        "    gpufl trace --passes Trace,SassMetrics -- ./app     # custom plan\n";
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
            auto err = take_value(out.profile);
            if (!err.empty()) return {std::nullopt, err};
            if (out.profile != "comprehensive" && out.profile != "light" &&
                out.profile != "monitoring-only") {
                return {std::nullopt,
                        "invalid --profile value: " + out.profile +
                        " (expected: comprehensive | light | monitoring-only)"};
            }
        } else if (key == "--engine") {
            auto err = take_value(out.engine);
            if (!err.empty()) return {std::nullopt, err};
            if (!isValidEngine(out.engine)) {
                return {std::nullopt,
                        "invalid --engine value: " + out.engine +
                        " (expected: Monitor | Trace | PcSampling | SassMetrics | "
                        "PmSampling | RangeProfiler | RangeProfilerKernelReplay | Deep)"};
            }
        } else if (key == "--passes") {
            std::string v;
            auto err = take_value(v);
            if (!err.empty()) return {std::nullopt, err};
            // Comma-separated engine list → one isolated pass each. Every token
            // must be a canonical engine name (same set as --engine).
            out.passes.clear();
            size_t start = 0;
            while (true) {
                const size_t comma = v.find(',', start);
                const std::string item = trim(v.substr(
                    start,
                    comma == std::string::npos ? std::string::npos : comma - start));
                if (!item.empty()) {
                    if (!isValidEngine(item)) {
                        return {std::nullopt,
                                "invalid --passes engine: " + item +
                                " (expected a comma-separated list of: Monitor | "
                                "Trace | PcSampling | SassMetrics | PmSampling | "
                                "RangeProfiler | RangeProfilerKernelReplay | Deep)"};
                    }
                    out.passes.push_back(item);
                }
                if (comma == std::string::npos) break;
                start = comma + 1;
            }
            if (out.passes.empty()) {
                return {std::nullopt, "--passes requires at least one engine"};
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
    if (!out.passes.empty() && !out.engine.empty()) {
        return {std::nullopt,
                "--engine and --passes are mutually exclusive (use --passes to "
                "list engines for a multi-pass run, or --engine for one engine "
                "— --engine Deep expands to the default multi-pass plan)"};
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
        "gpufl upload — Upload a captured session's NDJSON logs to the backend\n"
        "\n"
        "USAGE:\n"
        "    gpufl upload <LOG_PATH> [OPTIONS]\n"
        "\n"
        "ARGS:\n"
        "    <LOG_PATH>              Session log-path prefix — the same value as\n"
        "                            `gpufl trace`'s output dir, or the InitOptions\n"
        "                            log_path. Matches '<LOG_PATH>.device.log' plus\n"
        "                            rotated/.gz files. A trace dir works directly:\n"
        "                            e.g. ~/.gpufl/traces/20260603-101500_ab12cd34\n"
        "\n"
        "OPTIONS:\n"
        "        --backend-url <URL> Backend base URL.   Env: GPUFL_BACKEND_URL\n"
        "        --api-key <KEY>     Bearer token.       Env: GPUFL_API_KEY\n"
        "        --api-path <PATH>   Reverse-proxy mount. Defaults to /api/v1\n"
        "        --timeout <SECS>    Total wall budget for the upload. Default 300\n"
        "        --retries <N>       Retries per failing POST. Default 1\n"
        "    -q, --quiet             Suppress periodic progress lines\n"
        "        --session-id <ID>   Upload only this session (default: latest)\n"
        "        --all-sessions      Upload every session in the dir (excl. --session-id)\n"
        "        --force             Re-upload even if the cursor says it shipped\n"
        "    -h, --help              Print this help\n"
        "\n"
        "EXAMPLES:\n"
        "    gpufl upload ~/.gpufl/traces/20260603-101500_ab12cd34\n"
        "    gpufl upload ./logs --all-sessions\n"
        "    GPUFL_API_KEY=gpfl_… GPUFL_BACKEND_URL=https://api.gpuflight.com \\\n"
        "        gpufl upload ./logs --session-id <uuid>\n";
}

const char* monitorHelp() {
    return
        "gpufl monitor - Run long-lived GPU/host telemetry collection\n"
        "\n"
        "USAGE:\n"
        "    gpufl monitor [OPTIONS]\n"
        "\n"
        "OPTIONS:\n"
        "    -n, --name <NAME>       Monitor session name. Default: gpufl-monitor\n"
        "    -o, --output <DIR>      Local NDJSON output dir\n"
        "                            (default: ~/.gpufl/monitor/{ts}_{session_id}/)\n"
        "        --interval <MS>     Sampling interval in milliseconds. Default: 5000\n"
        "        --upload            Start gpufl-agent as the live uploader\n"
        "        --backend-url <URL> Backend base URL for --upload\n"
        "                            Env fallback: GPUFL_BACKEND_URL\n"
        "        --api-key <KEY>     Bearer token for --upload\n"
        "                            Env fallback: GPUFL_API_KEY\n"
        "        --api-version <VER> Agent HTTP API version. Default: v1\n"
        "        --agent-jar <PATH>  Run agent as `java -jar <PATH>`\n"
        "                            Env fallback: GPUFL_AGENT_JAR\n"
        "        --agent-cursor <P>  Agent cursor file. Default: <output>/cursor.json\n"
        "        --log-types <LIST>  Agent channels to upload. Default: system\n"
        "    -q, --quiet             Suppress launcher chatter\n"
        "    -v, --verbose           Verbose launcher logging\n"
        "    -h, --help              Print this help\n"
        "\n"
        "EXAMPLES:\n"
        "    gpufl monitor\n"
        "    gpufl monitor --interval 1000\n"
        "    gpufl monitor --name llm-node-1 --upload\n";
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
        return {std::nullopt, "missing <LOG_PATH> (the session's log-path prefix)"};
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
    // 1. Explicit plan wins.
    if (!args.passes.empty()) return args.passes;
    // 2. Deep at the launcher = the multi-pass orchestrator: each engine runs
    //    in its OWN process/pass (isolated), which dodges the SASS +
    //    kernel-activity deadlock by construction; the backend stitches the
    //    passes back into one kernel view. (The library / Python Deep engine
    //    is unchanged — still the single-process PcSamplingWithSass composite;
    //    only the launcher expands Deep here.)
    if (args.engine == "Deep") {
        return {"Trace", "PcSampling", "SassMetrics"};
    }
    // 3. Single pass — the explicit engine, or "" = the profile's default
    //    engine. This is the legacy single-pass path, unchanged.
    return {args.engine};
}

}  // namespace gpufl::launcher

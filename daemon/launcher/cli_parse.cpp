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
        "                            PmSampling | RangeProfiler | Deep\n"
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
        "    gpufl trace --profile=light -- python long_train.py\n";
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
            // Must match the canonical engine names gpufl::init() accepts
            // for GPUFL_PROFILING_ENGINE (see gpufl.cpp). The launcher only
            // validates here and forwards the string verbatim; init() is the
            // single string->enum parser, so this list is the one place to
            // keep in sync with the ProfilingEngine ladder.
            static const char* kEngines[] = {
                "Monitor", "Trace", "PcSampling", "SassMetrics",
                "PmSampling", "RangeProfiler", "Deep"};
            bool ok = false;
            for (auto* e : kEngines) if (out.engine == e) { ok = true; break; }
            if (!ok) {
                return {std::nullopt,
                        "invalid --engine value: " + out.engine +
                        " (expected: Monitor | Trace | PcSampling | SassMetrics | "
                        "PmSampling | RangeProfiler | Deep)"};
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

}  // namespace gpufl::launcher

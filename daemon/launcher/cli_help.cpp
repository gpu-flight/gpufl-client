#include "cli_parse.hpp"
#include "cli_trace_options.hpp"

#include <string>

namespace gpufl::launcher {

const char* topLevelHelp() {
    return R"HELP(gpufl - GPUFlight launcher

USAGE:
    gpufl <SUBCOMMAND> [OPTIONS]

SUBCOMMANDS:
    trace      Inject GPUFlight into a target process and capture telemetry
    monitor    Run long-lived GPU/host telemetry collection
    info       Print local GPU device capabilities
    upload     Upload a captured session's NDJSON logs to the backend
    version    Print version + build info

Run `gpufl <subcommand> --help` for subcommand-specific help.
)HELP";
}

const char* traceHelp() {
    static const std::string help = [] {
        std::string out = R"HELP(gpufl trace - Capture telemetry from a target process

USAGE:
    gpufl trace [OPTIONS] -- <COMMAND>...

OPTIONS:
)HELP";
        // Every option below is rendered from the registry in
        // cli_trace_options.cpp, so a new flag documents itself. Only prose
        // that belongs to no single option is written here.
        out += formatTraceSimpleOptions(TraceHelpSection::Capture);
        out += formatTraceSimpleOptions(TraceHelpSection::Runtime);
        out += formatTraceSimpleOptions(TraceHelpSection::Segmentation);
        out += formatTraceSimpleOptions(TraceHelpSection::Window);
        out += R"HELP(    A deep window is ONE adaptive run: gpufl picks the deep engine, so
    --deep-* cannot be combined with --passes. Which engine was selected
    is printed at startup rather than promised here - it depends on the
    GPU, and today only PM sampling is selected.

)HELP";
        out += formatTraceSimpleOptions(TraceHelpSection::Deep);
        out += formatTraceSimpleOptions(TraceHelpSection::Sampling);
        out += R"HELP(    -h, --help              Print this help

EXAMPLES:
    gpufl trace -- python train.py
    gpufl trace --name=quantize -- ./inference_server
    gpufl trace --passes=Trace,PmSampling -- python train.py
    gpufl trace --passes=Deep -- python train.py        # multi-pass
    gpufl trace --passes=Trace,SassMetrics -- ./app     # custom plan
    gpufl trace --passes=Trace+PcSampling -- ./app      # one-process composite
    gpufl trace --passes=Trace+PcSampling --warmup=60s --window=5m -- ./serve
)HELP";
        return out;
    }();
    return help.c_str();
}

const char* uploadHelp() {
    return R"HELP(gpufl upload - Upload a captured session's NDJSON logs to the backend

USAGE:
    gpufl upload <LOG_PATH> [OPTIONS]

ARGS:
    <LOG_PATH>              Output directory written by `gpufl trace`, or
                            the InitOptions log_path directory. Looks for
                            '<LOG_PATH>/<session_id>/<channel>.log[.gz]'.
                            A trace dir works directly:
                            e.g. ~/.gpufl/traces/20260603-101500_ab12cd34

OPTIONS:
        --backend-url=<URL> Backend base URL.   Env: GPUFL_BACKEND_URL
        --api-key=<KEY>     Bearer token.       Env: GPUFL_API_KEY
        --api-path=<PATH>   Reverse-proxy mount. Defaults to /api/v1
        --agent-jar=<PATH>  Run the uploader as `java -jar <PATH>`.
                            Env: GPUFL_AGENT_JAR (else gpufl-agent on PATH)
        --timeout=<SECS>    Cap on waiting for the upload to finish. Default 300
        --retries=<N>       Accepted for compatibility; the agent retries internally
    -q, --quiet             Suppress periodic progress lines
        --all-sessions      Upload every session in the dir (this is the default)
        --force             Re-upload even if the cursor says it shipped
    -h, --help              Print this help

EXAMPLES:
    gpufl upload ~/.gpufl/traces/20260603-101500_ab12cd34
    gpufl upload ./logs --force
    GPUFL_API_KEY=gpfl_… GPUFL_BACKEND_URL=https://api.gpuflight.com \
        gpufl upload ./logs
)HELP";
}

const char* monitorHelp() {
    return R"HELP(gpufl monitor - Run long-lived GPU/host telemetry collection

USAGE:
    gpufl monitor [OPTIONS]

OPTIONS:
    -n, --name=<NAME>       Monitor session name. Default: gpufl-monitor
    -o, --output=<DIR>      Local NDJSON output dir
                            (default: ~/.gpufl/monitor/{ts}_{session_id}/)
        --interval=<MS>     Sampling interval in milliseconds. Default: 5000
        --upload            Start gpufl-agent as the live uploader
        --backend-url=<URL> Backend base URL for --upload
                            Env fallback: GPUFL_BACKEND_URL
        --api-key=<KEY>     Bearer token for --upload
                            Env fallback: GPUFL_API_KEY
        --api-version=<VER> Agent HTTP API version. Default: v1
        --agent-jar=<PATH>  Run agent as `java -jar <PATH>`
                            Env fallback: GPUFL_AGENT_JAR
        --agent-cursor=<P>  Agent cursor file. Default: <output>/cursor.json
        --log-types=<LIST>  Agent channels to upload. Default: system
    -q, --quiet             Suppress launcher chatter
    -v, --verbose           Verbose launcher logging
    -h, --help              Print this help

EXAMPLES:
    gpufl monitor
    gpufl monitor --interval=1000
    gpufl monitor --name=llm-node-1 --upload
)HELP";
}

const char* infoHelp() {
    return R"HELP(gpufl info - Print local GPU device capabilities

USAGE:
    gpufl info [OPTIONS]

OPTIONS:
        --json              Emit stable machine-readable JSON
        --device=<ID>       Limit output to one zero-based device ID
    -h, --help              Print this help

EXAMPLES:
    gpufl info
    gpufl info --json
    gpufl info --device=0 --json
)HELP";
}

}  // namespace gpufl::launcher

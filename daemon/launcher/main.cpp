// gpufl launcher entry point. Top-level argv parse + subcommand dispatch.
// Phase 1 ships `trace`, `upload`, and `version`; `monitor`, `inspect`
// land in a later phase. `upload` consolidates the former Python
// `gpufl` console-script into this single native binary.

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "cli_parse.hpp"
#include "trace_command.hpp"
#include "upload_command.hpp"

#include "gpufl/core/version.hpp"

using namespace gpufl::launcher;

namespace {

int runVersion() {
    std::printf("gpufl %s (wire v%s)\n",
                gpufl::kClientVersion, gpufl::kWireVersion);
    return 0;
}

int runTraceFromArgs(const std::vector<std::string>& argv) {
    auto parsed = parseTraceArgs(argv);
    if (!parsed.args) {
        if (parsed.error == "__help__") {
            std::fputs(traceHelp(), stdout);
            return 0;
        }
        std::fprintf(stderr, "gpufl trace: %s\n\n", parsed.error.c_str());
        std::fputs(traceHelp(), stderr);
        return 2;
    }
    return runTrace(*parsed.args);
}

int runUploadFromArgs(const std::vector<std::string>& argv) {
    auto parsed = parseUploadArgs(argv);
    if (!parsed.args) {
        if (parsed.error == "__help__") {
            std::fputs(uploadHelp(), stdout);
            return 0;
        }
        std::fprintf(stderr, "gpufl upload: %s\n\n", parsed.error.c_str());
        std::fputs(uploadHelp(), stderr);
        return 2;
    }
    return runUpload(*parsed.args);
}

}  // namespace

int main(int argc, char** argv) {
    auto top = parseTopLevel(argc, argv);
    switch (top.sub) {
        case Subcommand::Help:
            std::fputs(topLevelHelp(), stdout);
            return 0;
        case Subcommand::Version:
            return runVersion();
        case Subcommand::Trace:
            return runTraceFromArgs(top.remaining);
        case Subcommand::Upload:
            return runUploadFromArgs(top.remaining);
        case Subcommand::Unknown:
            std::fprintf(stderr, "gpufl: unknown subcommand: %s\n\n",
                         top.remaining.empty() ? "" : top.remaining[0].c_str());
            std::fputs(topLevelHelp(), stderr);
            return 2;
    }
    return 2;
}

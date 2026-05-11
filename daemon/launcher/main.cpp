// gpufl launcher entry point. Top-level argv parse + subcommand dispatch.
// Phase 1 ships `trace` and `version`; `upload`, `monitor`, `inspect`
// land in Phase 4.

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "cli_parse.hpp"
#include "trace_command.hpp"

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
        case Subcommand::Unknown:
            std::fprintf(stderr, "gpufl: unknown subcommand: %s\n\n",
                         top.remaining.empty() ? "" : top.remaining[0].c_str());
            std::fputs(topLevelHelp(), stderr);
            return 2;
    }
    return 2;
}

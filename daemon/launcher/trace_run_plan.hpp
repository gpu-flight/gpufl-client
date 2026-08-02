#pragma once

#include "trace_command_common.hpp"

#include <string>
#include <vector>

namespace gpufl::launcher {

// The pure, per-invocation decisions made before trace starts mutating the
// target environment or filesystem. output_dir is intentionally only a path
// here; runTraceCommon owns creating and canonicalising it.
struct TraceRunPlan {
    std::vector<std::string> passes;
    bool segmented = false;
    bool multipass = false;
    std::string analysis_id;
    std::string run_id;
    std::string directory_tag;
    std::string app_name;
    fs::path output_dir;
    RunOptions run_options;
};

TraceRunPlan createTraceRunPlan(const TraceArgs& args,
                                const TracePlatform& platform);

}  // namespace gpufl::launcher

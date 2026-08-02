// Publishing the long-running session-segmentation contract into the target's
// environment. Kept separate from trace_command_common so ownership and stale
// environment scrubbing are unit-testable without launching a process.

#include <cstdio>
#include <string>

#include "cli_parse.hpp"
#include "gpufl/core/env_vars.hpp"
#include "trace_command_common.hpp"

namespace gpufl::launcher {

bool applySegmentationEnv(const TraceArgs& args, const std::string& run_id,
                          const TracePlatform& platform) {
    if (!segmentationRequested(args)) {
        return unsetEnvOrPrint(platform, env::kRunId) &&
               unsetEnvOrPrint(platform, env::kSegmentEveryMs) &&
               unsetEnvOrPrint(platform, env::kSegmentMaxRows) &&
               unsetEnvOrPrint(platform, env::kRunRollEveryMs) &&
               unsetEnvOrPrint(platform, env::kRunRollMaxBytes);
    }

    if (run_id.empty()) {
        std::fprintf(stderr,
                     "gpufl: internal error: segmented run has no GPUFL_RUN_ID\n");
        return false;
    }
    if (!setEnvOrPrint(platform, env::kRunId, run_id)) return false;

    if (args.segment_every_ms > 0) {
        if (!setEnvOrPrint(platform, env::kSegmentEveryMs,
                           std::to_string(args.segment_every_ms))) {
            return false;
        }
    } else if (!unsetEnvOrPrint(platform, env::kSegmentEveryMs)) {
        return false;
    }

    if (args.segment_max_rows > 0) {
        if (!setEnvOrPrint(platform, env::kSegmentMaxRows,
                           std::to_string(args.segment_max_rows))) {
            return false;
        }
    } else if (!unsetEnvOrPrint(platform, env::kSegmentMaxRows)) {
        return false;
    }

    if (args.run_roll_every_ms > 0) {
        if (!setEnvOrPrint(platform, env::kRunRollEveryMs,
                           std::to_string(args.run_roll_every_ms))) {
            return false;
        }
    } else if (!unsetEnvOrPrint(platform, env::kRunRollEveryMs)) {
        return false;
    }

    if (args.run_roll_max_bytes > 0) {
        if (!setEnvOrPrint(platform, env::kRunRollMaxBytes,
                           std::to_string(args.run_roll_max_bytes))) {
            return false;
        }
    } else if (!unsetEnvOrPrint(platform, env::kRunRollMaxBytes)) {
        return false;
    }

    return true;
}

}  // namespace gpufl::launcher

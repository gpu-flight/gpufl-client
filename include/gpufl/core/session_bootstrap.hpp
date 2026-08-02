#pragma once

#include <string>

#include "gpufl/core/logger/logger.hpp"

namespace gpufl {
struct InitOptions;
struct Runtime;

namespace detail {

// The values init() still needs after the initial logger/context transaction:
// job_start reports log_path and SegmentRuntime copies the resolved logger
// policy for later segments.
struct InitialSessionLoggingState {
    std::string log_path;
    Logger::Options options;
};

// Reporting happens after shutdown(), when Runtime is gone. Keep only the
// location needed to find the final session directory; this deliberately does
// not retain any live runtime object.
struct LastSessionReportSource {
    std::string log_path;
    std::string session_id;
};

// Open the first session logger and publish its immutable write context.
// Returns false with no published context when the logger cannot be opened.
bool openInitialSessionLogging(Runtime& runtime, const InitOptions& options,
                               bool segmented,
                               InitialSessionLoggingState& state);

// Snapshot the source remembered by the most recent bootstrap attempt.
LastSessionReportSource lastSessionReportSource();

}  // namespace detail
}  // namespace gpufl

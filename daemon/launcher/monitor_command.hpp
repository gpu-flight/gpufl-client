#pragma once

#include "cli_parse.hpp"

namespace gpufl::launcher {

// Runs `gpufl monitor`: telemetry-only foreground monitoring in this process.
// With --upload, starts gpufl-agent as a managed child for live upload.
int runMonitor(const MonitorArgs& args);

}  // namespace gpufl::launcher

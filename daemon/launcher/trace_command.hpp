#pragma once

#include "cli_parse.hpp"

namespace gpufl::launcher {

// Runs a `gpufl trace` invocation: builds the env, forks, execvps the
// target, waits for it, prints a summary. Returns the exit code the
// launcher itself should produce (0 on success; non-zero per the
// CLI spec exit codes — usage error 2, inject load failure 3, target
// non-zero passes through, signal-death = 130 etc).
int runTrace(const TraceArgs& args);

}  // namespace gpufl::launcher

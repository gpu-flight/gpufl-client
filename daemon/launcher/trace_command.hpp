#pragma once

#include <string>
#include <vector>

#include "cli_parse.hpp"
#include "trace_command_common.hpp"

namespace gpufl::launcher {

// Runs a `gpufl trace` invocation: builds the env, forks, execvps the
// target, waits for it, prints a summary. Returns the exit code the
// launcher itself should produce (0 on success; non-zero per the
// CLI spec exit codes - usage error 2, inject load failure 3, target
// non-zero passes through, signal-death = 130 etc).
int runTrace(const TraceArgs& args);

#if !defined(_WIN32)
// The POSIX runner behind `gpufl trace` (fork/exec, wait, bounded-window
// stop), exposed so the process-group stop is testable without the injection
// library.
TraceProcessResult runPosixTraceProcess(const std::vector<std::string>& command,
                                        const RunOptions& opts);
#endif

}  // namespace gpufl::launcher

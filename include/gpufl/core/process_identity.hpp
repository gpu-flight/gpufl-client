#pragma once

#include <cstddef>
#include <string>

namespace gpufl {

/**
 * GPUFL_LAUNCH_ID format: 1-64 characters from [A-Za-z0-9_-]. Shared by the
 * launcher, which refuses anything else, and init(), which drops it.
 */
constexpr std::size_t kMaxLaunchIdLength = 64;
bool isValidLaunchId(const std::string& value);

/**
 * The launch id this process inherited, or an empty string when none is set.
 * An invalid value is dropped with one warning per process: process identity
 * is metadata and must never fail the application.
 */
std::string launchIdFromEnvironment();

/** Parent process id, or 0 when it cannot be determined. */
int getParentPid();

/**
 * Short OS process name, at most 64 bytes: Linux /proc/self/comm, macOS
 * getprogname(), Windows the executable's base file name. Read on every call,
 * never cached, because a process may retitle itself after init (vLLM names
 * its engine process VLLM::EngineCore).
 */
std::string getProcessName();

/**
 * Base name of the process's first argument, at most 128 bytes - never the
 * command line, whose arguments can hold secrets. On Linux this is the first
 * element of /proc/self/cmdline, so a setproctitle() title is reported; on
 * other platforms it is the executable's base name. Read on every call.
 */
std::string getProcessArgv0();

namespace detail {
// The Linux argv0 rule applied to raw /proc/self/cmdline bytes: first
// NUL-terminated element, cut at the first whitespace, then the part after the
// last '/'. Exposed so the rule is testable on every platform.
std::string argv0FromCmdline(const std::string& cmdline);
}  // namespace detail

}  // namespace gpufl

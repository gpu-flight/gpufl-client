#pragma once

// Where the `gpufl` a Linux wheel puts on PATH finds the launcher it runs.
//
// pip installs a wheel's scripts into the environment's scripts directory,
// which is on PATH, and everything else into site-packages. The launcher has
// to stay in site-packages, in gpufl/bin: libgpufl_inject.so and the counter
// runtime sit beside it in gpufl/lib, and auditwheel points all three at the
// libraries it bundles in gpufl.libs/ by a path relative to site-packages. An
// ELF that auditwheel finds in the scripts directory needing those libraries
// is moved into site-packages and replaced by a Python shim. So the scripts
// directory gets wheel_entry.cpp instead: a native program with no bundled
// dependencies, which finds the launcher and execs it.
//
// The path from the scripts directory to site-packages depends on how pip was
// run, so there is one candidate per install scheme. python_dir names the
// interpreter's directory under lib/ ("python3.12", or "python3.13t" for a
// free-threaded build). A wheel is built for one interpreter, so its entry
// runs only that interpreter's launcher, even when the same prefix holds
// gpufl for several Python versions.

#include <filesystem>
#include <string>
#include <vector>

namespace gpufl::launcher {

// Launcher locations to try, in order, for an entry installed in scripts_dir.
std::vector<std::filesystem::path> wheelLauncherCandidates(
    const std::filesystem::path& scripts_dir, const std::string& python_dir);

// The first candidate that is an executable regular file, or an empty path.
std::filesystem::path findWheelLauncher(
    const std::vector<std::filesystem::path>& candidates);

}  // namespace gpufl::launcher

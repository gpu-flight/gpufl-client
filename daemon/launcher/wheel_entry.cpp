// The `gpufl` a Linux wheel installs on PATH. It finds the launcher pip put in
// site-packages and execs it with the same argv, so `gpufl <anything>` is the
// launcher itself: same output, exit code and signal handling. See
// wheel_entry_paths.hpp for why the launcher cannot sit on PATH directly.

#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <system_error>
#include <vector>

#include "wheel_entry_paths.hpp"

#ifndef GPUFL_WHEEL_PYTHON_DIR
#error "GPUFL_WHEEL_PYTHON_DIR must name the interpreter's lib/ directory"
#endif

namespace fs = std::filesystem;

int main(int /*argc*/, char** argv) {
    // /proc/self/exe has symlinks resolved, so a link to this file (pipx's
    // ~/.local/bin/gpufl) leads back to the environment it was installed in.
    std::error_code ec;
    const fs::path self = fs::read_symlink("/proc/self/exe", ec);
    if (ec || self.empty()) {
        std::fprintf(stderr, "gpufl: cannot resolve /proc/self/exe: %s\n",
                     ec ? ec.message().c_str() : "empty path");
        return 127;
    }

    const std::vector<fs::path> candidates =
        gpufl::launcher::wheelLauncherCandidates(self.parent_path(),
                                                 GPUFL_WHEEL_PYTHON_DIR);
    const fs::path launcher = gpufl::launcher::findWheelLauncher(candidates);
    if (launcher.empty()) {
        std::fprintf(stderr,
                     "gpufl: cannot find the launcher installed with %s\n"
                     "       searched:\n",
                     self.c_str());
        for (const fs::path& candidate : candidates) {
            std::fprintf(stderr, "         %s\n", candidate.c_str());
        }
        std::fprintf(stderr,
                     "       Reinstall gpufl into this Python environment.\n");
        return 127;
    }

    ::execv(launcher.c_str(), argv);
    const int err = errno;
    std::fprintf(stderr, "gpufl: cannot run %s: %s\n", launcher.c_str(),
                 std::strerror(err));
    return 126;
}

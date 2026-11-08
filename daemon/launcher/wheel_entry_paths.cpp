#include "wheel_entry_paths.hpp"

#include <unistd.h>

#include <system_error>

namespace gpufl::launcher {

namespace fs = std::filesystem;

std::vector<fs::path> wheelLauncherCandidates(const fs::path& scripts_dir,
                                              const std::string& python_dir) {
    // Every scheme below puts scripts in <base>/bin.
    const fs::path base = scripts_dir.parent_path();
    const fs::path launcher = fs::path("gpufl") / "bin" / "gpufl";
    return {
        // A venv, conda, pip --user (~/.local) and pip --prefix:
        // <base>/lib/pythonX.Y/site-packages.
        base / "lib" / python_dir / "site-packages" / launcher,
        // Fedora, RHEL and SUSE system interpreters keep platlib in lib64.
        base / "lib64" / python_dir / "site-packages" / launcher,
        // Debian and Ubuntu system pip: /usr/local/bin and
        // /usr/local/lib/pythonX.Y/dist-packages.
        base / "lib" / python_dir / "dist-packages" / launcher,
        // pip install --target DIR: scripts in DIR/bin, packages in DIR.
        base / launcher,
    };
}

fs::path findWheelLauncher(const std::vector<fs::path>& candidates) {
    for (const fs::path& candidate : candidates) {
        std::error_code ec;
        if (fs::is_regular_file(candidate, ec) &&
            ::access(candidate.c_str(), X_OK) == 0) {
            return candidate;
        }
    }
    return {};
}

}  // namespace gpufl::launcher

// Implements `gpufl trace [opts] -- <command>...`:
//   1. Resolves the path to the inject shared library
//      (libgpufl_inject.so) adjacent to /proc/self/exe.
//   2. Builds the output directory (~/.gpufl/traces/{ts}_{sid}/ by default).
//   3. Sets the env vars from inject_entry.hpp so the inject lib's
//      constructor can drive gpufl::init().
//   4. fork() + execvp() the target; waitpid() in the parent.
//   5. Prints a one-line summary and returns the appropriate exit code.
//
// Linux-only — uses fork(), execvp(), /proc/self/exe, LD_PRELOAD.

#include "trace_command.hpp"

#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "gpufl/inject/inject_entry.hpp"

namespace fs = std::filesystem;

namespace gpufl::launcher {

namespace {

// Resolve /proc/self/exe → absolute path of the `gpufl` binary.
fs::path selfExe() {
    char buf[4096];
    ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return {};
    buf[n] = '\0';
    return fs::path(buf);
}

// Candidate paths for libgpufl_inject.so relative to the launcher
// binary, in search order. Centralized so the not-found error can
// echo back exactly what we tried (saves a debug round-trip).
//
// Build tree:        build/daemon/launcher/gpufl  +  build/libgpufl_inject.so
// Install (CMake):   bin/gpufl                    +  lib/libgpufl_inject.so
// Install one-liner: ~/.local/bin/gpufl           +  ~/.local/lib/gpufl/libgpufl_inject.so
std::vector<fs::path> injectLibCandidates(const fs::path& exe) {
    const fs::path dir = exe.parent_path();
    const auto kName = "libgpufl_inject.so";
    return {
        dir / kName,                                          // colocated
        dir.parent_path() / kName,                            // one up
        dir.parent_path().parent_path() / kName,              // two up (default CMake build root)
        dir.parent_path() / "lib" / kName,                    // <build>/<subdir>/../lib
        dir.parent_path() / "lib" / "gpufl" / kName,          // install one-liner
        dir.parent_path().parent_path() / "lib" / kName,      // bin/../lib install layout
    };
}

fs::path findInjectLib(const fs::path& exe) {
    for (const auto& c : injectLibCandidates(exe)) {
        if (std::error_code ec; fs::exists(c, ec)) return fs::canonical(c, ec);
    }
    return {};
}

std::string makeSessionId() {
    static std::mt19937_64 rng{
        static_cast<uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count())};
    uint64_t v = rng();
    std::ostringstream os;
    os << std::hex << std::setw(8) << std::setfill('0') << (v & 0xffffffffu);
    return os.str();
}

std::string makeTimestamp() {
    auto t = std::time(nullptr);
    std::tm tm{};
    localtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", &tm);
    return buf;
}

fs::path defaultOutputDir(const std::string& session_id) {
    const char* home = std::getenv("HOME");
    const fs::path root = home && *home ? fs::path(home) : fs::path("/tmp");
    return root / ".gpufl" / "traces" / (makeTimestamp() + "_" + session_id);
}

std::string baseName(const std::string& path) {
    auto pos = path.find_last_of('/');
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

void setEnvOrDie(const char* k, const std::string& v) {
    if (::setenv(k, v.c_str(), /*overwrite=*/1) != 0) {
        std::fprintf(stderr,
                     "gpufl: setenv %s failed: %s\n", k, std::strerror(errno));
        std::exit(2);
    }
}

}  // namespace

int runTrace(const TraceArgs& args) {
    const fs::path exe = selfExe();
    if (exe.empty()) {
        std::fprintf(stderr,
                     "gpufl: cannot resolve /proc/self/exe (Linux required)\n");
        return 3;
    }

    const fs::path inject_lib = findInjectLib(exe);
    if (inject_lib.empty()) {
        std::fprintf(stderr,
                     "gpufl: cannot find libgpufl_inject.so adjacent to %s\n",
                     exe.string().c_str());
        std::fprintf(stderr, "       searched:\n");
        for (const auto& c : injectLibCandidates(exe)) {
            std::fprintf(stderr, "         %s\n", c.string().c_str());
        }
        return 3;
    }

    const std::string session_id = makeSessionId();
    const fs::path output_dir = args.output_dir.empty()
                              ? defaultOutputDir(session_id)
                              : fs::path(args.output_dir);

    std::error_code ec;
    fs::create_directories(output_dir, ec);
    if (ec) {
        std::fprintf(stderr, "gpufl: cannot create %s: %s\n",
                     output_dir.string().c_str(), ec.message().c_str());
        return 2;
    }

    const std::string app_name = args.name.empty()
                               ? baseName(args.command.front())
                               : args.name;

    // LD_PRELOAD respects ":"-separated existing values; preserve any
    // the user already set so we don't break their setup.
    std::string ld_preload = inject_lib.string();
    if (const char* prev = std::getenv("LD_PRELOAD"); prev && *prev) {
        ld_preload = std::string(prev) + ":" + ld_preload;
    }

    setEnvOrDie("LD_PRELOAD", ld_preload);
    setEnvOrDie("CUDA_INJECTION64_PATH", inject_lib.string());
    setEnvOrDie(inject::kEnvSentinel, "1");
    setEnvOrDie(inject::kEnvAppName, app_name);
    setEnvOrDie(inject::kEnvLogDir, output_dir.string());
    setEnvOrDie(inject::kEnvProfile, args.profile);
    if (!args.engine.empty()) {
        setEnvOrDie(inject::kEnvProfilingEngine, args.engine);
    }

    if (!args.quiet) {
        std::fprintf(stderr, "[gpufl] capturing → %s\n",
                     output_dir.string().c_str());
        if (args.verbose) {
            std::fprintf(stderr, "[gpufl] inject lib: %s\n",
                         inject_lib.string().c_str());
            std::fprintf(stderr, "[gpufl] app_name: %s\n", app_name.c_str());
            std::fprintf(stderr, "[gpufl] profile: %s\n", args.profile.c_str());
        }
    }

    const auto t_start = std::chrono::steady_clock::now();

    pid_t pid = ::fork();
    if (pid < 0) {
        std::fprintf(stderr, "gpufl: fork failed: %s\n", std::strerror(errno));
        return 2;
    }
    if (pid == 0) {
        // Child: execvp the target. The env we just set is inherited.
        std::vector<char*> argv;
        argv.reserve(args.command.size() + 1);
        for (auto& s : args.command) argv.push_back(const_cast<char*>(s.c_str()));
        argv.push_back(nullptr);
        ::execvp(args.command.front().c_str(), argv.data());
        // execvp only returns on failure.
        std::fprintf(stderr, "gpufl: cannot exec %s: %s\n",
                     args.command.front().c_str(), std::strerror(errno));
        std::_Exit(127);
    }

    int status = 0;
    while (::waitpid(pid, &status, 0) < 0) {
        if (errno == EINTR) continue;
        std::fprintf(stderr, "gpufl: waitpid failed: %s\n", std::strerror(errno));
        return 2;
    }

    const auto t_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - t_start)
                         .count();

    if (!args.quiet) {
        if (WIFEXITED(status)) {
            std::fprintf(stderr,
                         "[gpufl] target exited (rc=%d) in %.2fs\n",
                         WEXITSTATUS(status), t_elapsed / 1000.0);
        } else if (WIFSIGNALED(status)) {
            std::fprintf(stderr,
                         "[gpufl] target killed by signal %d in %.2fs\n",
                         WTERMSIG(status), t_elapsed / 1000.0);
        }
        std::fprintf(stderr,
                     "[gpufl] inspect: %s\n", output_dir.string().c_str());
    }

    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return 1;
}

}  // namespace gpufl::launcher

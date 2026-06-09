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

#include "gpufl/core/env_vars.hpp"

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

// 128-bit UUID-shaped id shared by all passes of one multi-pass analysis.
// Opaque to the backend (it only groups by string equality), so the exact
// layout doesn't matter — just full-width randomness for collision safety
// across accounts (makeSessionId's 32 bits is fine for a local dir name but
// too narrow to group analyses on the backend).
std::string makeAnalysisId() {
    static std::mt19937_64 rng{
        static_cast<uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count()) ^
        0x9e3779b97f4a7c15ULL};
    const uint64_t a = rng();
    const uint64_t b = rng();
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%08x-%04x-%04x-%04x-%012llx",
                  static_cast<unsigned>(a >> 32),
                  static_cast<unsigned>((a >> 16) & 0xffff),
                  static_cast<unsigned>(a & 0xffff),
                  static_cast<unsigned>((b >> 48) & 0xffff),
                  static_cast<unsigned long long>(b & 0xffffffffffffULL));
    return buf;
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

    // Resolve the multi-pass plan (explicit --passes, --engine Deep expanded,
    // or a single pass) — the shared source of truth lives in cli_parse.
    const std::vector<std::string> plan = resolvePassPlan(args);
    const bool multipass = plan.size() > 1;

    // One analysis_id shared by every pass lets the backend stitch the isolated
    // passes into a single kernel view. Single-pass runs get none and keep the
    // legacy {ts}_{sid} dir name.
    const std::string analysis_id = multipass ? makeAnalysisId() : std::string();
    const std::string dir_tag = multipass ? analysis_id : makeSessionId();

    const fs::path output_dir = args.output_dir.empty()
                              ? defaultOutputDir(dir_tag)
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

    // ── Env shared by every pass: set once, inherited by each child. The log
    //    dir is shared too — each pass's process writes under its OWN
    //    session-id subdir (Logger's <dir>/<session_id>/ layout), so the
    //    uploader discovers N sessions and the backend groups them by
    //    analysis_id. LD_PRELOAD keeps any value the user already set. ──
    std::string ld_preload = inject_lib.string();
    if (const char* prev = std::getenv(gpufl::env::kLdPreload); prev && *prev) {
        ld_preload = std::string(prev) + ":" + ld_preload;
    }

    setEnvOrDie(gpufl::env::kLdPreload, ld_preload);
    setEnvOrDie(gpufl::env::kCudaInjection64Path, inject_lib.string());
    setEnvOrDie(gpufl::env::kNvtxInjection64Path, inject_lib.string());
    setEnvOrDie(gpufl::env::kInject, "1");
    setEnvOrDie(gpufl::env::kAppName, app_name);
    setEnvOrDie(gpufl::env::kLogDir, output_dir.string());
    setEnvOrDie(gpufl::env::kInjectProfile, args.profile);

    // --upload: fail fast here (before we exec) if the creds the inject lib's
    // post-run uploadLogs() will need aren't in the environment. Each pass
    // uploads its own session; the backend groups them by analysis_id.
    if (args.upload) {
        const char* api_key     = std::getenv(gpufl::env::kApiKey);
        const char* backend_url = std::getenv(gpufl::env::kBackendUrl);
        if (!api_key || !*api_key || !backend_url || !*backend_url) {
            std::fprintf(stderr,
                "gpufl trace --upload: GPUFL_API_KEY and GPUFL_BACKEND_URL "
                "must both be set in the environment to upload.\n");
            return 2;
        }
        setEnvOrDie(gpufl::env::kInjectUpload, "1");
    }

    if (multipass) {
        setEnvOrDie(gpufl::env::kAnalysisId, analysis_id);
        setEnvOrDie(gpufl::env::kPassCount, std::to_string(plan.size()));
    }

    if (!args.quiet) {
        std::fprintf(stderr, "[gpufl] capturing → %s\n",
                     output_dir.string().c_str());
        if (multipass) {
            std::fprintf(stderr, "[gpufl] multi-pass analysis %s — %zu passes:",
                         analysis_id.c_str(), plan.size());
            for (const auto& e : plan) std::fprintf(stderr, " %s", e.c_str());
            std::fputc('\n', stderr);
        }
        if (args.verbose) {
            std::fprintf(stderr, "[gpufl] inject lib: %s\n",
                         inject_lib.string().c_str());
            std::fprintf(stderr, "[gpufl] app_name: %s\n", app_name.c_str());
            std::fprintf(stderr, "[gpufl] profile: %s\n", args.profile.c_str());
        }
    }

    // ── Run the workload once per pass. A failing pass does NOT abort the
    //    rest (a partial analysis still captures the passes that worked); the
    //    first non-zero pass becomes the launcher's exit code. ──
    int overall_rc = 0;
    for (size_t i = 0; i < plan.size(); ++i) {
        const std::string& engine = plan[i];

        // Per-pass engine. An empty engine (legacy single pass with no
        // --engine) leaves GPUFL_PROFILING_ENGINE as the user/profile set it.
        if (!engine.empty()) {
            setEnvOrDie(gpufl::env::kProfilingEngine, engine);
        }
        if (multipass) {
            setEnvOrDie(gpufl::env::kPassIndex, std::to_string(i));
        }

        const std::string what =
            multipass ? ("pass " + std::to_string(i + 1) + "/" +
                         std::to_string(plan.size()) + " (" + engine + ")")
                      : std::string("target");

        if (!args.quiet && multipass) {
            std::fprintf(stderr, "\n[gpufl] ── %s ──\n", what.c_str());
            if (engine == "PcSampling") {
                std::fprintf(stderr,
                    "[gpufl] note: PcSampling needs admin / NVIDIA CP \"allow GPU "
                    "performance counters to all users\"; this pass reports \"needs "
                    "admin\" and yields no PC data if unprivileged.\n");
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

        int rc;
        if (WIFEXITED(status)) {
            rc = WEXITSTATUS(status);
            if (!args.quiet) {
                std::fprintf(stderr, "[gpufl] %s exited (rc=%d) in %.2fs\n",
                             what.c_str(), rc, t_elapsed / 1000.0);
            }
        } else if (WIFSIGNALED(status)) {
            rc = 128 + WTERMSIG(status);
            if (!args.quiet) {
                std::fprintf(stderr, "[gpufl] %s killed by signal %d in %.2fs\n",
                             what.c_str(), WTERMSIG(status), t_elapsed / 1000.0);
            }
        } else {
            rc = 1;
        }

        // First failing pass sets the exit code; keep going so later passes
        // still run and the analysis is as complete as possible.
        if (rc != 0 && overall_rc == 0) overall_rc = rc;
    }

    if (!args.quiet) {
        std::fprintf(stderr, "[gpufl] inspect: %s\n", output_dir.string().c_str());
    }

    return overall_rc;
}

}  // namespace gpufl::launcher

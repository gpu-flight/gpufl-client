// POSIX implementation of the tiny platform layer behind `gpufl trace`.
// The orchestration lives in trace_command_common.cpp.

#include "trace_command.hpp"

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <string>
#include <vector>

#include "gpufl/core/env_vars.hpp"
#include "trace_command_common.hpp"

namespace gpufl::launcher {
namespace {

std::string makeTimestamp() {
    auto t = std::time(nullptr);
    std::tm tm{};
    localtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", &tm);
    return buf;
}

class PosixTracePlatform final : public TracePlatform {
   public:
    const char* platformName() const override { return "POSIX"; }
    const char* injectLibraryName() const override { return "libgpufl_inject.so"; }

    fs::path selfExe() const override {
        char buf[4096];
        ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
        if (n <= 0) return {};
        buf[n] = '\0';
        return fs::path(buf);
    }

    std::vector<fs::path> injectLibCandidates(const fs::path& exe) const override {
        const fs::path dir = exe.parent_path();
        const auto kName = "libgpufl_inject.so";
        return {
            dir / kName,                                          // colocated
            dir.parent_path() / kName,                            // one up
            dir.parent_path().parent_path() / kName,              // two up
            dir.parent_path() / "lib" / kName,                    // build lib dir
            dir.parent_path() / "lib" / "gpufl" / kName,          // install one-liner
            dir.parent_path().parent_path() / "lib" / kName,      // bin/../lib
        };
    }

    fs::path defaultOutputDir(const std::string& tag) const override {
        const char* home = std::getenv("HOME");
        const fs::path root = home && *home ? fs::path(home) : fs::path("/tmp");
        return root / ".gpufl" / "traces" / (makeTimestamp() + "_" + tag);
    }

    std::string defaultAppName(const std::string& command0) const override {
        auto pos = command0.find_last_of('/');
        return pos == std::string::npos ? command0 : command0.substr(pos + 1);
    }

    bool setEnv(const char* key, const std::string& value,
                std::string& error) const override {
        if (::setenv(key, value.c_str(), /*overwrite=*/1) == 0) return true;
        error = "setenv " + std::string(key) + " failed: " + std::strerror(errno);
        return false;
    }

    bool prepareInjectionEnv(const fs::path& inject_lib,
                             std::string& error) const override {
        std::string ld_preload = inject_lib.string();
        if (const char* prev = std::getenv(env::kLdPreload); prev && *prev) {
            ld_preload = std::string(prev) + ":" + ld_preload;
        }
        return setEnv(env::kLdPreload, ld_preload, error);
    }

    TraceProcessResult runProcess(
            const std::vector<std::string>& command) const override {
        TraceProcessResult out;
        pid_t pid = ::fork();
        if (pid < 0) {
            out.launcher_error = true;
            out.rc = 2;
            out.error = std::string("fork failed: ") + std::strerror(errno);
            return out;
        }
        if (pid == 0) {
            std::vector<char*> argv;
            argv.reserve(command.size() + 1);
            for (const auto& s : command) {
                argv.push_back(const_cast<char*>(s.c_str()));
            }
            argv.push_back(nullptr);
            execvp(command.front().c_str(), argv.data());
            std::fprintf(stderr, "gpufl: cannot exec %s: %s\n",
                         command.front().c_str(), std::strerror(errno));
            std::_Exit(127);
        }

        int status = 0;
        while (::waitpid(pid, &status, 0) < 0) {
            if (errno == EINTR) continue;
            out.launcher_error = true;
            out.rc = 2;
            out.error = std::string("waitpid failed: ") + std::strerror(errno);
            return out;
        }

        if (WIFEXITED(status)) {
            out.rc = WEXITSTATUS(status);
        } else if (WIFSIGNALED(status)) {
            out.signaled = true;
            out.signal = WTERMSIG(status);
            out.rc = 128 + out.signal;
        } else {
            out.rc = 1;
        }
        return out;
    }
};

}  // namespace

int runTrace(const TraceArgs& args) {
    return runTraceCommon(args, PosixTracePlatform{});
}

}  // namespace gpufl::launcher

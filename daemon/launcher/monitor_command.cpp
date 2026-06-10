#include "monitor_command.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "gpufl/core/env_vars.hpp"
#include "monitor_runner.hpp"

namespace fs = std::filesystem;

namespace gpufl::launcher {
namespace {

constexpr const char* kAgentJarEnv = "GPUFL_AGENT_JAR";

std::string envOrEmpty(const char* name) {
    if (const char* v = std::getenv(name); v && *v) return v;
    return {};
}

std::string resolve(const std::string& flag, const char* env_name) {
    if (!flag.empty()) return flag;
    return envOrEmpty(env_name);
}

std::string makeSessionId() {
    static std::mt19937_64 rng{
        static_cast<uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count()) ^
        0x6eed0e9da4d94a4fULL};
    const uint64_t v = rng();
    std::ostringstream os;
    os << std::hex << std::setw(8) << std::setfill('0') << (v & 0xffffffffu);
    return os.str();
}

std::string makeTimestamp() {
    const auto t = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", &tm);
    return buf;
}

fs::path homeDir() {
#ifdef _WIN32
    if (const char* up = std::getenv("USERPROFILE"); up && *up) {
        return fs::path(up);
    }
    if (const char* hd = std::getenv("HOMEDRIVE")) {
        if (const char* hp = std::getenv("HOMEPATH")) {
            return fs::path(std::string(hd) + hp);
        }
    }
    return fs::temp_directory_path();
#else
    if (const char* home = std::getenv("HOME"); home && *home) {
        return fs::path(home);
    }
    return fs::path("/tmp");
#endif
}

fs::path defaultOutputDir() {
    return homeDir() / ".gpufl" / "monitor" /
           (makeTimestamp() + "_" + makeSessionId());
}

std::vector<std::string> splitPathEnv() {
    std::vector<std::string> out;
    const char* raw = std::getenv("PATH");
    if (!raw) return out;
    std::string path(raw);
#ifdef _WIN32
    constexpr char delim = ';';
#else
    constexpr char delim = ':';
#endif
    size_t start = 0;
    while (start <= path.size()) {
        const size_t pos = path.find(delim, start);
        std::string item = path.substr(
            start, pos == std::string::npos ? std::string::npos : pos - start);
        if (!item.empty()) out.push_back(std::move(item));
        if (pos == std::string::npos) break;
        start = pos + 1;
    }
    return out;
}

fs::path findExecutableOnPath(const std::string& name) {
    const std::vector<std::string> dirs = splitPathEnv();
#ifdef _WIN32
    const std::vector<std::string> suffixes = {".exe"};
#else
    const std::vector<std::string> suffixes = {""};
#endif
    for (const auto& dir : dirs) {
        for (const auto& suffix : suffixes) {
            fs::path candidate = fs::path(dir) / (name + suffix);
            std::error_code ec;
            if (fs::exists(candidate, ec) && !fs::is_directory(candidate, ec)) {
                return fs::weakly_canonical(candidate, ec);
            }
        }
    }
    return {};
}

bool setEnv(const char* name, const std::string& value) {
#ifdef _WIN32
    return SetEnvironmentVariableA(name, value.c_str()) != 0;
#else
    return ::setenv(name, value.c_str(), 1) == 0;
#endif
}

void appendQuotedArg(const std::string& arg, std::string& cmd) {
    if (!arg.empty() && arg.find_first_of(" \t\n\v\"") == std::string::npos) {
        cmd += arg;
        return;
    }
    cmd += '"';
    for (auto it = arg.begin();; ++it) {
        unsigned backslashes = 0;
        while (it != arg.end() && *it == '\\') { ++it; ++backslashes; }
        if (it == arg.end()) {
            cmd.append(backslashes * 2, '\\');
            break;
        }
        if (*it == '"') {
            cmd.append(backslashes * 2 + 1, '\\');
            cmd += '"';
        } else {
            cmd.append(backslashes, '\\');
            cmd += *it;
        }
    }
    cmd += '"';
}

std::string buildCommandLine(const std::vector<std::string>& args) {
    std::string cmd;
    for (size_t i = 0; i < args.size(); ++i) {
        if (i) cmd += ' ';
        appendQuotedArg(args[i], cmd);
    }
    return cmd;
}

#ifdef _WIN32
std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_ACP, 0, s.data(),
                                      static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_ACP, 0, s.data(), static_cast<int>(s.size()),
                        w.data(), n);
    return w;
}
#endif

struct AgentProcess {
#ifdef _WIN32
    PROCESS_INFORMATION pi{};
#else
    pid_t pid = -1;
#endif
    bool running = false;

    bool start(const std::vector<std::string>& command, std::string& error) {
#ifdef _WIN32
        std::wstring cmdline = widen(buildCommandLine(command));
        std::vector<wchar_t> cmdbuf(cmdline.begin(), cmdline.end());
        cmdbuf.push_back(L'\0');

        STARTUPINFOW si{};
        si.cb = sizeof(si);
        if (!CreateProcessW(nullptr, cmdbuf.data(), nullptr, nullptr, TRUE, 0,
                            nullptr, nullptr, &si, &pi)) {
            error = "CreateProcess failed for gpufl-agent (err=" +
                    std::to_string(static_cast<unsigned long>(GetLastError())) + ")";
            return false;
        }
        running = true;
        return true;
#else
        pid = ::fork();
        if (pid < 0) {
            error = "fork failed while starting gpufl-agent";
            return false;
        }
        if (pid == 0) {
            std::vector<char*> argv;
            argv.reserve(command.size() + 1);
            for (const auto& s : command) {
                argv.push_back(const_cast<char*>(s.c_str()));
            }
            argv.push_back(nullptr);
            ::execvp(command.front().c_str(), argv.data());
            std::fprintf(stderr, "gpufl monitor: cannot exec %s\n",
                         command.front().c_str());
            std::_Exit(127);
        }
        running = true;
        return true;
#endif
    }

    void stop() {
        if (!running) return;
#ifdef _WIN32
        TerminateProcess(pi.hProcess, 0);
        WaitForSingleObject(pi.hProcess, 5000);
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
#else
        ::kill(pid, SIGTERM);
        int status = 0;
        for (int i = 0; i < 50; ++i) {
            const pid_t rc = ::waitpid(pid, &status, WNOHANG);
            if (rc == pid) {
                running = false;
                return;
            }
            usleep(100000);
        }
        ::kill(pid, SIGKILL);
        ::waitpid(pid, &status, 0);
#endif
        running = false;
    }
};

struct AgentStartPlan {
    std::vector<std::string> command;
    std::string description;
};

bool buildAgentStartPlan(const MonitorArgs& args, AgentStartPlan& plan,
                         std::string& error) {
    const std::string agent_jar = resolve(args.agent_jar, kAgentJarEnv);
    if (!agent_jar.empty()) {
        std::error_code ec;
        if (!fs::exists(agent_jar, ec)) {
            error = "agent jar not found: " + agent_jar;
            return false;
        }
        const fs::path java = findExecutableOnPath("java");
        if (java.empty()) {
            error = "java not found on PATH; install Java or put gpufl-agent on PATH";
            return false;
        }
        plan.command = {java.string(), "-jar", agent_jar};
        plan.description = "java -jar " + agent_jar;
        return true;
    }

    const fs::path agent = findExecutableOnPath("gpufl-agent");
    if (agent.empty()) {
        error = "gpufl-agent not found. Install gpufl-agent on PATH or pass "
                "--agent-jar <path> / GPUFL_AGENT_JAR.";
        return false;
    }
    plan.command = {agent.string()};
    plan.description = agent.string();
    return true;
}

}  // namespace

int runMonitor(const MonitorArgs& args) {
    fs::path output_dir = args.output_dir.empty()
                        ? defaultOutputDir()
                        : fs::path(args.output_dir);
    std::error_code ec;
    fs::create_directories(output_dir, ec);
    if (ec) {
        std::fprintf(stderr, "gpufl monitor: cannot create %s: %s\n",
                     output_dir.string().c_str(), ec.message().c_str());
        return 2;
    }
    output_dir = fs::weakly_canonical(output_dir, ec);

    AgentProcess agent;
    if (args.upload) {
        const std::string backend_url =
            resolve(args.backend_url, gpufl::env::kBackendUrl);
        const std::string api_key =
            resolve(args.api_key, gpufl::env::kApiKey);
        if (backend_url.empty()) {
            std::fprintf(stderr,
                "gpufl monitor --upload: --backend-url required "
                "(or set GPUFL_BACKEND_URL)\n");
            return 2;
        }
        if (api_key.empty()) {
            std::fprintf(stderr,
                "gpufl monitor --upload: --api-key required "
                "(or set GPUFL_API_KEY)\n");
            return 2;
        }

        fs::path cursor = args.agent_cursor.empty()
                        ? output_dir / "cursor.json"
                        : fs::path(args.agent_cursor);

        if (!setEnv("GPUFL_SOURCE_FOLDERS", output_dir.string()) ||
            !setEnv("GPUFL_LOG_TYPES", args.log_types) ||
            !setEnv("GPUFL_CURSOR_FILE", cursor.string()) ||
            !setEnv("GPUFL_PUBLISHER_TYPE", "http") ||
            !setEnv("GPUFL_HTTP_HOST", backend_url) ||
            !setEnv("GPUFL_HTTP_TOKEN", api_key) ||
            !setEnv("GPUFL_HTTP_API_VERSION", args.api_version) ||
            !setEnv("GPUFL_AGENT_UPLOAD_MODE", "stream")) {
            std::fprintf(stderr, "gpufl monitor: failed to configure agent environment\n");
            return 2;
        }

        AgentStartPlan plan;
        std::string error;
        if (!buildAgentStartPlan(args, plan, error)) {
            std::fprintf(stderr, "gpufl monitor --upload: %s\n", error.c_str());
            return 2;
        }
        if (!args.quiet) {
            std::fprintf(stderr, "[gpufl] starting agent: %s\n",
                         plan.description.c_str());
        }
        if (!agent.start(plan.command, error)) {
            std::fprintf(stderr, "gpufl monitor --upload: %s\n", error.c_str());
            return 3;
        }
    }

    gpufl::daemon::MonitorRunOptions opts;
    opts.app_name = args.name;
    opts.log_path = output_dir.string();
    opts.interval_ms = args.interval_ms;
    opts.quiet = args.quiet;

    const int rc = gpufl::daemon::runMonitorForeground(opts);
    agent.stop();
    return rc;
}

}  // namespace gpufl::launcher

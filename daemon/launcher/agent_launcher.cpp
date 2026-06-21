#include "agent_launcher.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <utility>
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

namespace fs = std::filesystem;

namespace gpufl::launcher {
namespace {

constexpr const char* kAgentJarEnv = "GPUFL_AGENT_JAR";

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

}  // namespace

AgentProcess::~AgentProcess() {
    stop();
}

bool AgentProcess::start(const std::vector<std::string>& command,
                         std::string& error) {
#ifdef _WIN32
    std::wstring cmdline = widen(buildCommandLine(command));
    std::vector<wchar_t> cmdbuf(cmdline.begin(), cmdline.end());
    cmdbuf.push_back(L'\0');

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(nullptr, cmdbuf.data(), nullptr, nullptr, TRUE, 0,
                        nullptr, nullptr, &si, &pi)) {
        error = "CreateProcess failed for gpufl-agent (err=" +
                std::to_string(static_cast<unsigned long>(GetLastError())) + ")";
        return false;
    }
    process_ = pi.hProcess;
    thread_ = pi.hThread;
    running_ = true;
    return true;
#else
    pid_ = ::fork();
    if (pid_ < 0) {
        error = "fork failed while starting gpufl-agent";
        return false;
    }
    if (pid_ == 0) {
        std::vector<char*> argv;
        argv.reserve(command.size() + 1);
        for (const auto& s : command) {
            argv.push_back(const_cast<char*>(s.c_str()));
        }
        argv.push_back(nullptr);
        ::execvp(command.front().c_str(), argv.data());
        std::fprintf(stderr, "gpufl: cannot exec %s\n", command.front().c_str());
        std::_Exit(127);
    }
    running_ = true;
    return true;
#endif
}

bool AgentProcess::waitForExit(int timeoutMs) {
    if (!running_) return true;
#ifdef _WIN32
    auto process = process_;
    const DWORD r = WaitForSingleObject(
        process, timeoutMs < 0 ? INFINITE : static_cast<DWORD>(timeoutMs));
    if (r == WAIT_OBJECT_0) {
        CloseHandle(thread_);
        CloseHandle(process);
        process_ = nullptr;
        thread_ = nullptr;
        running_ = false;
        return true;
    }
    return false;  // timeout/failure - caller falls back to stop()
#else
    constexpr int step_ms = 100;
    int waited = 0;
    int status = 0;
    while (timeoutMs < 0 || waited < timeoutMs) {
        const pid_t rc = ::waitpid(pid_, &status, WNOHANG);
        if (rc == pid_ || rc < 0) {  // exited, or already reaped / gone
            running_ = false;
            pid_ = -1;
            return true;
        }
        usleep(step_ms * 1000);
        waited += step_ms;
    }
    return false;
#endif
}

void AgentProcess::stop() {
    if (!running_) return;
#ifdef _WIN32
    const auto process = process_;
    const auto thread = thread_;
    TerminateProcess(process, 0);
    WaitForSingleObject(process, 5000);
    CloseHandle(thread);
    CloseHandle(process);
    process_ = nullptr;
    thread_ = nullptr;
#else
    ::kill(pid_, SIGTERM);
    int status = 0;
    for (int i = 0; i < 50; ++i) {
        const pid_t rc = ::waitpid(pid_, &status, WNOHANG);
        if (rc == pid_) {
            running_ = false;
            return;
        }
        usleep(100000);
    }
    ::kill(pid_, SIGKILL);
    ::waitpid(pid_, &status, 0);
    pid_ = -1;
#endif
    running_ = false;
}

std::string envOrEmpty(const char* name) {
    if (const char* v = std::getenv(name); v && *v) return v;
    return {};
}

std::string resolveOption(const std::string& flag, const char* env_name) {
    if (!flag.empty()) return flag;
    return envOrEmpty(env_name);
}

bool configureAgentEnvironment(const AgentOptions& opts, std::string& error) {
    if (opts.source_folders.empty()) {
        error = "agent source folder is empty";
        return false;
    }
    if (opts.log_types.empty()) {
        error = "agent log types are empty";
        return false;
    }
    if (opts.cursor_file.empty()) {
        error = "agent cursor file is empty";
        return false;
    }
    if (opts.backend_url.empty()) {
        error = "--backend-url required (or set GPUFL_BACKEND_URL)";
        return false;
    }
    if (opts.api_key.empty()) {
        error = "--api-key required (or set GPUFL_API_KEY)";
        return false;
    }
    if (!setEnv("GPUFL_SOURCE_FOLDER", opts.source_folders) ||
        !setEnv("GPUFL_SOURCE_FOLDERS", opts.source_folders) ||
        !setEnv("GPUFL_LOG_TYPES", opts.log_types) ||
        !setEnv("GPUFL_CURSOR_FILE", opts.cursor_file) ||
        !setEnv("GPUFL_PUBLISHER_TYPE", "http") ||
        !setEnv("GPUFL_HTTP_HOST", opts.backend_url) ||
        !setEnv("GPUFL_HTTP_TOKEN", opts.api_key) ||
        !setEnv("GPUFL_HTTP_API_VERSION", opts.api_version) ||
        !setEnv("GPUFL_AGENT_UPLOAD_MODE", "stream") ||
        // One-shot: the agent exits once this trace's session has fully uploaded, so the
        // launcher can wait for a clean drain (waitForExit) instead of hard-killing it.
        !setEnv("GPUFL_AGENT_EXIT_WHEN_DRAINED", "1")) {
        error = "failed to configure agent environment";
        return false;
    }
    return true;
}

bool buildAgentLaunchPlan(const AgentOptions& opts, AgentLaunchPlan& plan,
                          std::string& error) {
    const std::string agent_jar = resolveOption(opts.agent_jar, kAgentJarEnv);
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

}  // namespace gpufl::launcher

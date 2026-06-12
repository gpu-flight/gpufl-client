// Windows implementation of the tiny platform layer behind `gpufl trace`.
// The orchestration lives in trace_command_common.cpp.

#include "trace_command.hpp"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <string>
#include <vector>

#include "trace_command_common.hpp"

namespace gpufl::launcher {
namespace {

std::string makeTimestamp() {
    const auto t = std::time(nullptr);
    std::tm tm{};
    localtime_s(&tm, &t);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", &tm);
    return buf;
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

std::string buildCommandLine(const std::vector<std::string>& command) {
    std::string cmd;
    for (size_t i = 0; i < command.size(); ++i) {
        if (i) cmd += ' ';
        appendQuotedArg(command[i], cmd);
    }
    return cmd;
}

std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_ACP, 0, s.data(),
                                      static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_ACP, 0, s.data(), static_cast<int>(s.size()),
                        w.data(), n);
    return w;
}

class WindowsTracePlatform final : public TracePlatform {
   public:
    const char* platformName() const override { return "Windows"; }
    const char* injectLibraryName() const override { return "gpufl_inject.dll"; }

    fs::path selfExe() const override {
        wchar_t buf[32768];
        const DWORD n = GetModuleFileNameW(
                nullptr, buf, static_cast<DWORD>(sizeof(buf) / sizeof(buf[0])));
        if (n == 0 || n >= sizeof(buf) / sizeof(buf[0])) return {};
        return fs::path(std::wstring(buf, n));
    }

    std::vector<fs::path> injectLibCandidates(const fs::path& exe) const override {
        const fs::path dir = exe.parent_path();
        const auto kName = L"gpufl_inject.dll";
        return {
            dir / kName,                                           // colocated
            dir.parent_path() / kName,                             // one up
            dir.parent_path().parent_path() / kName,               // two up
            dir.parent_path().parent_path().parent_path() / kName, // VS cfg dir
            dir.parent_path() / "bin" / kName,                     // ../bin
            dir.parent_path().parent_path() / "bin" / kName,       // ../../bin
        };
    }

    fs::path defaultOutputDir(const std::string& tag) const override {
        fs::path root;
        if (const char* up = std::getenv("USERPROFILE"); up && *up) {
            root = fs::path(up);
        } else if (const char* hd = std::getenv("HOMEDRIVE")) {
            if (const char* hp = std::getenv("HOMEPATH")) {
                root = fs::path(std::string(hd) + hp);
            }
        }
        if (root.empty()) root = fs::temp_directory_path();
        return root / ".gpufl" / "traces" / (makeTimestamp() + "_" + tag);
    }

    std::string defaultAppName(const std::string& command0) const override {
        const auto pos = command0.find_last_of("/\\");
        std::string name = pos == std::string::npos ? command0 : command0.substr(pos + 1);
        if (name.size() > 4) {
            const std::string ext = name.substr(name.size() - 4);
            if (ext == ".exe" || ext == ".EXE") name.resize(name.size() - 4);
        }
        return name;
    }

    bool setEnv(const char* key, const std::string& value,
                std::string& error) const override {
        if (SetEnvironmentVariableA(key, value.c_str())) return true;
        error = "SetEnvironmentVariable " + std::string(key) + " failed (err=" +
                std::to_string(static_cast<unsigned long>(GetLastError())) + ")";
        return false;
    }

    bool prepareInjectionEnv(const fs::path& inject_lib,
                             std::string& error) const override {
        std::string path = inject_lib.parent_path().string();
        if (const char* prev = std::getenv("PATH"); prev && *prev) {
            path += ';';
            path += prev;
        }
        return setEnv("PATH", path, error);
    }

    TraceProcessResult runProcess(
            const std::vector<std::string>& command) const override {
        TraceProcessResult out;

        std::wstring cmdline = widen(buildCommandLine(command));
        std::vector<wchar_t> cmdbuf(cmdline.begin(), cmdline.end());
        cmdbuf.push_back(L'\0');

        STARTUPINFOW si{};
        si.cb = sizeof(si);
        PROCESS_INFORMATION pi{};
        HANDLE job = CreateJobObjectW(nullptr, nullptr);
        if (job) {
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
            limits.BasicLimitInformation.LimitFlags =
                    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if (!SetInformationJobObject(
                    job, JobObjectExtendedLimitInformation, &limits,
                    sizeof(limits))) {
                CloseHandle(job);
                job = nullptr;
            }
        }

        if (!CreateProcessW(/*lpApplicationName=*/nullptr,
                            cmdbuf.data(),
                            /*procAttrs=*/nullptr, /*threadAttrs=*/nullptr,
                            /*inheritHandles=*/TRUE,
                            /*creationFlags=*/CREATE_SUSPENDED,
                            /*lpEnvironment=*/nullptr,
                            /*lpCurrentDirectory=*/nullptr,
                            &si, &pi)) {
            if (job) CloseHandle(job);
            out.launcher_error = true;
            out.rc = 2;
            out.error = "cannot launch " + command.front() + " (CreateProcess err=" +
                        std::to_string(static_cast<unsigned long>(GetLastError())) + ")";
            return out;
        }

        if (job && !AssignProcessToJobObject(job, pi.hProcess)) {
            CloseHandle(job);
            job = nullptr;
        }
        ResumeThread(pi.hThread);

        WaitForSingleObject(pi.hProcess, INFINITE);

        DWORD exit_code = 1;
        GetExitCodeProcess(pi.hProcess, &exit_code);
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
        if (job) CloseHandle(job);

        out.rc = static_cast<int>(exit_code);
        return out;
    }
};

}  // namespace

int runTrace(const TraceArgs& args) {
    return runTraceCommon(args, WindowsTracePlatform{});
}

}  // namespace gpufl::launcher

// Windows implementation of `gpufl trace [opts] -- <command>...`.
//
//   1. Resolve gpufl_inject.dll adjacent to the launcher exe
//      (GetModuleFileNameW instead of /proc/self/exe).
//   2. Build the output dir (%USERPROFILE%\.gpufl\traces\{ts}_{sid}\ default).
//   3. Set GPUFL_* + CUDA_INJECTION64_PATH + NVTX_INJECTION64_PATH so the
//      driver loads the inject DLL and calls InitializeInjection.
//      NOTE: no LD_PRELOAD — Windows has no preload symbol interposition,
//      so the launch/sync interpose waits (the Linux early-kernel safety
//      net) do NOT exist here; we rely purely on the CUDA injection ABI,
//      which the driver invokes during cuInit, before the app's launches.
//   4. CreateProcessW the target; WaitForSingleObject; GetExitCodeProcess.
//   5. Print a one-line summary and return the target's exit code.
//
// Windows-only — selected by daemon/launcher/CMakeLists.txt on WIN32.

#include "trace_command.hpp"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

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

#include "gpufl/inject/inject_entry.hpp"

namespace fs = std::filesystem;

namespace gpufl::launcher {

namespace {

// Absolute path of the running gpufl.exe.
fs::path selfExe() {
    wchar_t buf[32768];
    const DWORD n = GetModuleFileNameW(nullptr, buf, static_cast<DWORD>(sizeof(buf) / sizeof(buf[0])));
    if (n == 0 || n >= sizeof(buf) / sizeof(buf[0])) return {};
    return fs::path(std::wstring(buf, n));
}

// Candidate paths for gpufl_inject.dll relative to the launcher binary,
// in search order. The not-found error echoes every path tried.
//
// Build tree (VS): build\daemon\launcher\<Cfg>\gpufl.exe + the DLL is copied
// next to it by the CMake POST_BUILD step, so "colocated" resolves first.
// Install: bin\gpufl.exe + bin\gpufl_inject.dll (DLLs live beside the exe
// on Windows, not in lib\).
std::vector<fs::path> injectLibCandidates(const fs::path& exe) {
    const fs::path dir = exe.parent_path();
    const auto kName = L"gpufl_inject.dll";
    return {
        dir / kName,                                          // colocated (primary on Windows)
        dir.parent_path() / kName,                            // one up
        dir.parent_path().parent_path() / kName,              // two up
        dir.parent_path().parent_path().parent_path() / kName,// three up (VS <Cfg> subdir)
        dir.parent_path() / "bin" / kName,                    // ../bin
        dir.parent_path().parent_path() / "bin" / kName,      // ../../bin
    };
}

fs::path findInjectLib(const fs::path& exe) {
    for (const auto& c : injectLibCandidates(exe)) {
        if (std::error_code ec; fs::exists(c, ec)) return fs::weakly_canonical(c, ec);
    }
    return {};
}

std::string makeSessionId() {
    static std::mt19937_64 rng{
        static_cast<uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count())};
    const uint64_t v = rng();
    std::ostringstream os;
    os << std::hex << std::setw(8) << std::setfill('0') << (v & 0xffffffffu);
    return os.str();
}

std::string makeTimestamp() {
    const auto t = std::time(nullptr);
    std::tm tm{};
    localtime_s(&tm, &t);  // Windows: (tm*, time_t*) — reversed from POSIX
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", &tm);
    return buf;
}

fs::path defaultOutputDir(const std::string& session_id) {
    fs::path root;
    if (const char* up = std::getenv("USERPROFILE"); up && *up) {
        root = fs::path(up);
    } else if (const char* hd = std::getenv("HOMEDRIVE")) {
        if (const char* hp = std::getenv("HOMEPATH")) root = fs::path(std::string(hd) + hp);
    }
    if (root.empty()) root = fs::temp_directory_path();
    return root / ".gpufl" / "traces" / (makeTimestamp() + "_" + session_id);
}

std::string baseName(const std::string& path) {
    const auto pos = path.find_last_of("/\\");  // accept either separator
    std::string name = pos == std::string::npos ? path : path.substr(pos + 1);
    // Drop a trailing .exe for a cleaner default session name.
    if (name.size() > 4) {
        const std::string ext = name.substr(name.size() - 4);
        if (ext == ".exe" || ext == ".EXE") name.resize(name.size() - 4);
    }
    return name;
}

void setEnvOrDie(const char* k, const std::string& v) {
    if (!SetEnvironmentVariableA(k, v.c_str())) {
        std::fprintf(stderr, "gpufl: SetEnvironmentVariable %s failed (err=%lu)\n",
                     k, static_cast<unsigned long>(GetLastError()));
        std::exit(2);
    }
}

// Quote one argument per the CommandLineToArgvW rules so CreateProcess
// reconstructs argv exactly (handles spaces, quotes, trailing backslashes).
// Algorithm per MSDN "Everyone quotes command line arguments the wrong way".
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
            cmd.append(backslashes * 2, '\\');  // escape all before closing quote
            break;
        }
        if (*it == '"') {
            cmd.append(backslashes * 2 + 1, '\\');  // escape backslashes + the quote
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
    const int n = MultiByteToWideChar(CP_ACP, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_ACP, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
    return w;
}

}  // namespace

int runTrace(const TraceArgs& args) {
    const fs::path exe = selfExe();
    if (exe.empty()) {
        std::fprintf(stderr, "gpufl: cannot resolve module path\n");
        return 3;
    }

    const fs::path inject_lib = findInjectLib(exe);
    if (inject_lib.empty()) {
        std::fprintf(stderr,
                     "gpufl: cannot find gpufl_inject.dll adjacent to %s\n",
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

    // When the driver loads gpufl_inject.dll into the target, the loader must
    // resolve the DLL's own dependencies (cupti64*.dll, nvperf_host.dll —
    // copied next to the DLL by the build). Prepend the inject DLL's directory
    // to the child's PATH so they're found regardless of whether the driver
    // uses LOAD_WITH_ALTERED_SEARCH_PATH. (OpenSSL is already on PATH; nvml is
    // in System32.)
    {
        std::string path = inject_lib.parent_path().string();
        if (const char* prev = std::getenv("PATH"); prev && *prev) {
            path += ';';
            path += prev;
        }
        setEnvOrDie("PATH", path);
    }

    // No LD_PRELOAD on Windows — the driver loads the DLL via the injection
    // path. We still set both CUDA + NVTX injection vars to the same DLL.
    setEnvOrDie("CUDA_INJECTION64_PATH", inject_lib.string());
    setEnvOrDie("NVTX_INJECTION64_PATH", inject_lib.string());
    setEnvOrDie(inject::kEnvSentinel, "1");
    setEnvOrDie(inject::kEnvAppName, app_name);
    setEnvOrDie(inject::kEnvLogDir, output_dir.string());
    setEnvOrDie(inject::kEnvProfile, args.profile);
    if (!args.engine.empty()) {
        setEnvOrDie(inject::kEnvProfilingEngine, args.engine);
    }

    // --upload: fail fast before launch if creds the inject DLL's post-run
    // uploadLogs() will need aren't in the environment.
    if (args.upload) {
        const char* api_key     = std::getenv("GPUFL_API_KEY");
        const char* backend_url = std::getenv("GPUFL_BACKEND_URL");
        if (!api_key || !*api_key || !backend_url || !*backend_url) {
            std::fprintf(stderr,
                "gpufl trace --upload: GPUFL_API_KEY and GPUFL_BACKEND_URL "
                "must both be set in the environment to upload.\n");
            return 2;
        }
        setEnvOrDie(inject::kEnvUpload, "1");
    }

    if (!args.quiet) {
        std::fprintf(stderr, "[gpufl] capturing -> %s\n", output_dir.string().c_str());
        if (args.verbose) {
            std::fprintf(stderr, "[gpufl] inject dll: %s\n", inject_lib.string().c_str());
            std::fprintf(stderr, "[gpufl] app_name: %s\n", app_name.c_str());
            std::fprintf(stderr, "[gpufl] profile: %s\n", args.profile.c_str());
        }
    }

    // CreateProcessW wants a mutable command-line buffer. lpApplicationName
    // = null so the program is resolved from the command line (incl. PATH),
    // mirroring execvp's PATH search.
    std::wstring cmdline = widen(buildCommandLine(args.command));
    std::vector<wchar_t> cmdbuf(cmdline.begin(), cmdline.end());
    cmdbuf.push_back(L'\0');

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};

    const auto t_start = std::chrono::steady_clock::now();

    // The env we just set via SetEnvironmentVariable is inherited because
    // lpEnvironment is null (child gets a copy of our environment block).
    if (!CreateProcessW(/*lpApplicationName=*/nullptr,
                        cmdbuf.data(),
                        /*procAttrs=*/nullptr, /*threadAttrs=*/nullptr,
                        /*inheritHandles=*/TRUE,
                        /*creationFlags=*/0,
                        /*lpEnvironment=*/nullptr,
                        /*lpCurrentDirectory=*/nullptr,
                        &si, &pi)) {
        std::fprintf(stderr, "gpufl: cannot launch %s (CreateProcess err=%lu)\n",
                     args.command.front().c_str(),
                     static_cast<unsigned long>(GetLastError()));
        return 2;
    }

    WaitForSingleObject(pi.hProcess, INFINITE);

    DWORD exit_code = 1;
    GetExitCodeProcess(pi.hProcess, &exit_code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);

    const auto t_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - t_start).count();

    if (!args.quiet) {
        std::fprintf(stderr, "[gpufl] target exited (rc=%lu) in %.2fs\n",
                     static_cast<unsigned long>(exit_code), t_elapsed / 1000.0);
        std::fprintf(stderr, "[gpufl] inspect: %s\n", output_dir.string().c_str());
    }

    return static_cast<int>(exit_code);
}

}  // namespace gpufl::launcher

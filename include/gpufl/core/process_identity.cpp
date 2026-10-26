// Process identity for launch grouping: which process of a multi-process
// launch a session came from. Its own TU so the per-OS includes
// (<windows.h> + <tlhelp32.h>, <mach-o/dyld.h>) stay out of everything else.

#include "gpufl/core/process_identity.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <mutex>
#include <utility>

#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/env_vars.hpp"

#if defined(_WIN32)
#include <windows.h>
#include <tlhelp32.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#include <stdlib.h>
#include <unistd.h>

#include <cstdint>
#include <vector>
#else
#include <unistd.h>

#include <fstream>
#endif

namespace gpufl {
namespace {

constexpr std::size_t kMaxProcessNameBytes = 64;
constexpr std::size_t kMaxArgv0Bytes = 128;

bool isSpace(const char c) {
    return std::isspace(static_cast<unsigned char>(c)) != 0;
}

// Cuts to at most max_bytes, then drops a trailing incomplete UTF-8 sequence,
// so neither this cap nor the kernel's own 15-byte comm cut leaves a broken
// character in the JSON.
std::string capUtf8(std::string value, const std::size_t max_bytes) {
    if (value.size() > max_bytes) value.resize(max_bytes);
    std::size_t lead = value.size();
    std::size_t continuation = 0;
    while (lead > 0 && continuation < 4 &&
           (static_cast<unsigned char>(value[lead - 1]) & 0xC0) == 0x80) {
        --lead;
        ++continuation;
    }
    if (lead == 0) return {};
    const auto first = static_cast<unsigned char>(value[lead - 1]);
    const std::size_t expected =
        first >= 0xF0 ? 4 : first >= 0xE0 ? 3 : first >= 0xC0 ? 2 : 1;
    if (continuation + 1 < expected) value.resize(lead - 1);
    return value;
}

#if defined(_WIN32)
std::string executableBaseName() {
    std::wstring path(MAX_PATH, L'\0');
    for (;;) {
        const DWORD n = GetModuleFileNameW(nullptr, path.data(),
                                           static_cast<DWORD>(path.size()));
        if (n == 0) return {};
        if (n < path.size()) {
            path.resize(n);
            break;
        }
        // Truncated: the path is longer than the buffer.
        if (path.size() >= 32768) return {};
        path.resize(path.size() * 2);
    }
    const std::size_t slash = path.find_last_of(L"\\/");
    const std::wstring base =
        slash == std::wstring::npos ? path : path.substr(slash + 1);
    const int bytes =
        WideCharToMultiByte(CP_UTF8, 0, base.data(),
                            static_cast<int>(base.size()), nullptr, 0,
                            nullptr, nullptr);
    if (bytes <= 0) return {};
    std::string out(static_cast<std::size_t>(bytes), '\0');
    WideCharToMultiByte(CP_UTF8, 0, base.data(), static_cast<int>(base.size()),
                        out.data(), bytes, nullptr, nullptr);
    return out;
}
#elif defined(__APPLE__)
std::string executableBaseName() {
    std::vector<char> buffer(1024);
    auto size = static_cast<uint32_t>(buffer.size());
    if (_NSGetExecutablePath(buffer.data(), &size) != 0) {
        // `size` now holds the length the path needs.
        buffer.resize(size);
        if (_NSGetExecutablePath(buffer.data(), &size) != 0) return {};
    }
    const std::string path(buffer.data());
    const std::size_t slash = path.rfind('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}
#endif

}  // namespace

bool isValidLaunchId(const std::string& value) {
    if (value.empty() || value.size() > kMaxLaunchIdLength) return false;
    return std::all_of(value.begin(), value.end(), [](const char c) {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
               (c >= '0' && c <= '9') || c == '_' || c == '-';
    });
}

std::string launchIdFromEnvironment() {
    const char* value = std::getenv(env::kLaunchId);
    if (!value || !*value) return {};
    if (isValidLaunchId(value)) return value;
    static std::once_flag warned;
    std::call_once(warned, [] {
        GFL_LOG_WARN(env::kLaunchId,
                     " is not 1-64 characters from [A-Za-z0-9_-]; this "
                     "session is recorded without a launch_id");
    });
    return {};
}

int getParentPid() {
#if defined(_WIN32)
    const HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snapshot == INVALID_HANDLE_VALUE) return 0;
    const DWORD self = GetCurrentProcessId();
    PROCESSENTRY32W entry{};
    entry.dwSize = sizeof(entry);
    int parent = 0;
    for (BOOL ok = Process32FirstW(snapshot, &entry); ok;
         ok = Process32NextW(snapshot, &entry)) {
        if (entry.th32ProcessID == self) {
            parent = static_cast<int>(entry.th32ParentProcessID);
            break;
        }
    }
    CloseHandle(snapshot);
    return parent;
#else
    return static_cast<int>(getppid());
#endif
}

std::string getProcessName() {
#if defined(_WIN32)
    return capUtf8(executableBaseName(), kMaxProcessNameBytes);
#elif defined(__APPLE__)
    const char* name = getprogname();
    return capUtf8(name ? name : "", kMaxProcessNameBytes);
#else
    // comm is one line: the name the kernel shows in ps/top, which follows
    // prctl(PR_SET_NAME) retitling.
    std::ifstream in("/proc/self/comm");
    std::string name;
    std::getline(in, name);
    const auto begin = std::find_if_not(name.begin(), name.end(), isSpace);
    const auto end = std::find_if_not(name.rbegin(), name.rend(), isSpace).base();
    return capUtf8(begin < end ? std::string(begin, end) : std::string(),
                   kMaxProcessNameBytes);
#endif
}

std::string getProcessArgv0() {
#if defined(_WIN32) || defined(__APPLE__)
    return capUtf8(executableBaseName(), kMaxArgv0Bytes);
#else
    // Only the first element is needed; the rest of the command line is never
    // read.
    std::ifstream in("/proc/self/cmdline", std::ios::binary);
    std::string first;
    std::getline(in, first, '\0');
    return detail::argv0FromCmdline(first);
#endif
}

namespace detail {

std::string argv0FromCmdline(const std::string& cmdline) {
    std::string first = cmdline.substr(0, cmdline.find('\0'));
    first.erase(std::find_if(first.begin(), first.end(), isSpace), first.end());
    const std::size_t slash = first.rfind('/');
    if (slash != std::string::npos) first.erase(0, slash + 1);
    return capUtf8(std::move(first), kMaxArgv0Bytes);
}

}  // namespace detail

}  // namespace gpufl

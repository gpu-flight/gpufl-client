#include "gpufl/core/stack_trace.hpp"
#include "gpufl/core/itanium_demangle.hpp"

#include <cstdlib>
#include <sstream>
#include <vector>

#ifdef _WIN32
// clang-format off
#include <windows.h>  // must precede dbghelp.h
#include <dbghelp.h>
// clang-format on
#pragma comment(lib, "dbghelp.lib")

namespace gpufl {
namespace core {
std::string DemangleName(const char* mangled) {
    if (!mangled || mangled[0] == '\0') return mangled ? mangled : "";
    // CUDA device-side kernel names use Itanium ABI mangling regardless
    // of host OS (NVCC always emits `_Z…`). MSVC's UnDecorateSymbolName
    // only understands its own `?…` mangling, so we delegate `_Z`-prefix
    // names to our portable Itanium demangler — without this, Windows
    // builds shipped raw mangled kernel names that broke the frontend's
    // regex-based op-catalog classification (no GEMM / ELEMENTWISE
    // chips on Windows traces).
    if (mangled[0] == '_' && mangled[1] == 'Z') {
        return DemangleItaniumName(mangled);
    }
    // Windows: MSVC-mangled names start with '?'; try UnDecorateSymbolName.
    if (mangled[0] == '?') {
        char buf[512];
        DWORD result = UnDecorateSymbolName(mangled, buf, sizeof(buf),
                                            UNDNAME_COMPLETE);
        if (result > 0) return std::string(buf);
    }
    return mangled;
}

std::string GetCallStack(int skipFrames) {
    HANDLE process = GetCurrentProcess();

    static bool symbolsInitialized = false;
    if (!symbolsInitialized) {
        SymInitialize(process, nullptr, TRUE);
        symbolsInitialized = true;
    }

    void* stack[62];
    unsigned short frames = CaptureStackBackTrace(0, 62, stack, nullptr);

    std::ostringstream oss;

    alignas(SYMBOL_INFO) char buffer[sizeof(SYMBOL_INFO) + 256];
    SYMBOL_INFO* symbol = reinterpret_cast<SYMBOL_INFO*>(buffer);
    symbol->MaxNameLen = 255;
    symbol->SizeOfStruct = sizeof(SYMBOL_INFO);

    bool first = true;
    for (int i = static_cast<int>(frames) - 1; i >= skipFrames; --i) {
        if (SymFromAddr(process, reinterpret_cast<DWORD64>(stack[i]), 0, symbol)) {
            std::string name = symbol->Name;

            if (name.empty()) continue;
            if (name.find("GetCallStack") != std::string::npos) continue;
            if (name.find("BaseThreadInitThunk") != std::string::npos) continue;
            if (name.find("RtlUserThreadStart") != std::string::npos) continue;

            if (!first) oss << "|";
            oss << name;
            first = false;
        }
    }

    return oss.str();
}
}  // namespace core
}  // namespace gpufl
#else
#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>

#include <cstring>
#include <memory>

namespace gpufl {
namespace core {
std::string DemangleName(const char* mangled) {
    if (!mangled || mangled[0] == '\0') return mangled ? mangled : "";
    int status = 0;
    char* demangled = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);
    if (status == 0 && demangled) {
        std::string result(demangled);
        free(demangled);
        return result;
    }
    return mangled;
}

std::string GetCallStack(int skipFrames) {
    void* callstack[64];
    int frames = backtrace(callstack, 64);
    char** strs = backtrace_symbols(callstack, frames);

    if (!strs) return "unknown";

    std::ostringstream oss;
    bool first = true;

    for (int i = frames - 1; i >= skipFrames; --i) {
        std::string line = strs[i];
        std::string name;

        size_t openParen = line.find('(');
        size_t plusSign = line.find('+');
        if (openParen != std::string::npos && plusSign != std::string::npos) {
            std::string raw =
                line.substr(openParen + 1, plusSign - openParen - 1);
            name = DemangleName(raw.c_str());
        }

        if (name.empty()) continue;
        if (name.find("GetCallStack") != std::string::npos) continue;
        if (name.find("clone") != std::string::npos) continue;
        if (name.find("_start") != std::string::npos) continue;
        if (name.find("start_thread") != std::string::npos) continue;

        if (!first) oss << "|";
        oss << name;
        first = false;
    }

    free(strs);
    return oss.str();
}
}  // namespace core
}  // namespace gpufl

#endif
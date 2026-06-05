#include "gpufl/core/stack_trace.hpp"

#include "gpufl/core/common.hpp"  // detail::SanitizeStackTrace
#include "gpufl/core/itanium_demangle.hpp"

#include <cstdlib>
#include <sstream>

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

RawStack CaptureCallStackRaw(int skipFrames) {
    RawStack raw;
    void* stack[RawStack::kMaxFrames];
    const unsigned short frames =
        CaptureStackBackTrace(0, RawStack::kMaxFrames, stack, nullptr);
    // Store outermost-first (matches the historical GetCallStack ordering),
    // dropping the top `skipFrames` frames (capture fn + immediate wrappers).
    for (int i = static_cast<int>(frames) - 1;
         i >= skipFrames && raw.count < RawStack::kMaxFrames; --i) {
        raw.frames[raw.count++] = stack[i];
    }
    return raw;
}

// Symbolize raw frames into an un-sanitized "outer|...|inner" string.
// Heavy: dbghelp's SymFromAddr takes a process-global lock. Off-hot-path only.
static std::string SymbolizeRaw(const RawStack& raw) {
    HANDLE process = GetCurrentProcess();

    static bool symbolsInitialized = false;
    if (!symbolsInitialized) {
        SymInitialize(process, nullptr, TRUE);
        symbolsInitialized = true;
    }

    alignas(SYMBOL_INFO) char buffer[sizeof(SYMBOL_INFO) + 256];
    SYMBOL_INFO* symbol = reinterpret_cast<SYMBOL_INFO*>(buffer);
    symbol->MaxNameLen = 255;
    symbol->SizeOfStruct = sizeof(SYMBOL_INFO);

    std::ostringstream oss;
    bool first = true;
    for (uint8_t i = 0; i < raw.count; ++i) {
        if (SymFromAddr(process, reinterpret_cast<DWORD64>(raw.frames[i]), 0,
                        symbol)) {
            std::string name = symbol->Name;

            if (name.empty()) continue;
            if (name.find("CaptureCallStackRaw") != std::string::npos) continue;
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

std::string ResolveCallStack(const RawStack& raw) {
    return detail::SanitizeStackTrace(SymbolizeRaw(raw));
}

std::string GetCallStack(int skipFrames) {
    // +1 compensates for CaptureCallStackRaw's own frame so the kept frame
    // set matches the historical inline-capture behavior of this function.
    return SymbolizeRaw(CaptureCallStackRaw(skipFrames + 1));
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
// Strip Itanium "[abi:...]" decorations that __cxa_demangle renders.
// Numba/NVVM CUDA kernels encode their lowered signature as a long hashed
// abi-tag (e.g. `__main__::matmul_kernel[abi:v1][abi:cw51cX…]`); for
// readable names + the frontend's regex op-categorization we drop them.
// This keeps the Linux output consistent with the Windows portable
// demangler (DemangleItaniumName), which discards abi-tags while parsing.
static std::string StripAbiTags(std::string s) {
    std::string::size_type pos;
    while ((pos = s.find("[abi:")) != std::string::npos) {
        const std::string::size_type close = s.find(']', pos);
        if (close == std::string::npos) break;
        s.erase(pos, close - pos + 1);
    }
    return s;
}

std::string DemangleName(const char* mangled) {
    if (!mangled || mangled[0] == '\0') return mangled ? mangled : "";
    int status = 0;
    char* demangled = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);
    if (status == 0 && demangled) {
        std::string result(demangled);
        free(demangled);
        return StripAbiTags(std::move(result));
    }
    return mangled;
}

RawStack CaptureCallStackRaw(int skipFrames) {
    RawStack raw;
    void* callstack[RawStack::kMaxFrames];
    const int frames = backtrace(callstack, RawStack::kMaxFrames);
    for (int i = frames - 1;
         i >= skipFrames && raw.count < RawStack::kMaxFrames; --i) {
        raw.frames[raw.count++] = callstack[i];
    }
    return raw;
}

// Symbolize raw frames into an un-sanitized "outer|...|inner" string.
// Heavy (backtrace_symbols allocates + parses). Off-hot-path only.
static std::string SymbolizeRaw(const RawStack& raw) {
    if (raw.count == 0) return "";
    char** strs = backtrace_symbols(raw.frames, raw.count);
    if (!strs) return "unknown";

    std::ostringstream oss;
    bool first = true;
    for (uint8_t i = 0; i < raw.count; ++i) {
        std::string line = strs[i];
        std::string name;

        size_t openParen = line.find('(');
        size_t plusSign = line.find('+');
        if (openParen != std::string::npos && plusSign != std::string::npos) {
            std::string rawname =
                line.substr(openParen + 1, plusSign - openParen - 1);
            name = DemangleName(rawname.c_str());
        }

        if (name.empty()) continue;
        if (name.find("CaptureCallStackRaw") != std::string::npos) continue;
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

std::string ResolveCallStack(const RawStack& raw) {
    return detail::SanitizeStackTrace(SymbolizeRaw(raw));
}

std::string GetCallStack(int skipFrames) {
    return SymbolizeRaw(CaptureCallStackRaw(skipFrames + 1));
}
}  // namespace core
}  // namespace gpufl

#endif

#include "gpufl/core/stack_trace.hpp"
#include <sstream>
#include <vector>
#include <cstdlib>

#ifdef _WIN32
    #include <windows.h>
    #include <dbghelp.h>
    #pragma comment(lib, "dbghelp.lib")

    namespace gpufl {
        namespace core {
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
                for (int i = frames - 1; i >= skipFrames; --i) {
                    if (SymFromAddr(process, (DWORD64)(stack[i]), 0, symbol)) {
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
        }
    }
#else
    #include <execinfo.h>
    #include <cxxabi.h>
    #include <dlfcn.h>
    #include <memory>
    #include <cstring>

    namespace gpufl {
        namespace core {
            std::string GetCallStack(int skipFrames) {
                void* callstack[64];
                int frames = backtrace(callstack, 64);
                char** strs = backtrace_symbols(callstack, frames);

                if (!strs) return "unknown";

                std::ostringstream oss;
                bool first = true;

                for (int i = frames - 1; i >= skipFrames; --i) {
                    std::string line = strs[i];
                    std::string name = line;

                    size_t openParen = line.find('(');
                    size_t plusSign = line.find('+');
                    if (openParen != std::string::npos && plusSign != std::string::npos) {
                        std::string raw = line.substr(openParen + 1, plusSign - openParen - 1);

                        int status;
                        char* demangled = abi::__cxa_demangle(raw.c_str(), nullptr, nullptr, &status);
                        if (status == 0 && demangled) {
                            name = demangled;
                            free(demangled);
                        } else {
                            name = raw;
                        }
                    }

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
        }
    }

#endif
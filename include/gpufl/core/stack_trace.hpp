#pragma once
#include <string>

namespace gpufl::core {
/**
 * Captures the current call stack and returns it as a string.
 * Format: "main::FunctionA::FunctionB"
 * * @param skipFrames Number of top frames to skip (default 1 to skip
 * GetCallStack itself)
 */
std::string GetCallStack(int skipFrames = 1);

/**
 * Demangles a C++ mangled symbol name (e.g. "_Z11vectorScalePiii" ->
 * "vectorScale(int*, int, int, int)"). Returns the original name if
 * demangling fails or the input is not a mangled symbol.
 */
std::string DemangleName(const char* mangled);
}  // namespace gpufl::core

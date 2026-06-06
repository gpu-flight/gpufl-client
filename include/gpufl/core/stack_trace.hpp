#pragma once
#include <cstdint>
#include <string>

namespace gpufl::core {

/**
 * A captured call stack as raw return addresses only — NO symbol resolution.
 * Capturing addresses (CaptureStackBackTrace / backtrace) is cheap and safe to
 * do on a hot path (e.g. inside a CUPTI launch callback). Turning addresses
 * into symbol names (SymFromAddr / backtrace_symbols) is expensive — dbghelp
 * serializes globally — so that step is deferred to ResolveCallStack(), run off
 * the hot path on the collector/worker thread.
 */
struct RawStack {
    static constexpr int kMaxFrames = 64;
    void* frames[kMaxFrames] = {};  // outermost-first (matches GetCallStack order)
    uint8_t count = 0;
};

/**
 * Cheap: walk the current call stack and store raw return addresses only.
 * `skipFrames` top frames (this function + immediate wrappers) are dropped.
 * No symbolization, no dbghelp lock — safe on the per-launch hot path.
 */
RawStack CaptureCallStackRaw(int skipFrames = 1);

/**
 * Expensive: resolve a RawStack to a sanitized "main|FunctionA|FunctionB"
 * string (SymFromAddr / backtrace_symbols + user-frame filtering). Call OFF
 * the hot path (collector/worker thread), e.g. from StackRegistry::get().
 */
std::string ResolveCallStack(const RawStack& raw);

/**
 * Captures the current call stack and returns it as a string.
 * Format: "main|FunctionA|FunctionB"
 * @param skipFrames Number of top frames to skip (default 1)
 *
 * NOTE: this symbolizes synchronously. Prefer CaptureCallStackRaw() +
 * deferred ResolveCallStack() on hot paths.
 */
std::string GetCallStack(int skipFrames = 1);

/**
 * Demangles a C++ mangled symbol name (e.g. "_Z11vectorScalePiii" ->
 * "vectorScale(int*, int, int, int)"). Returns the original name if
 * demangling fails or the input is not a mangled symbol.
 */
std::string DemangleName(const char* mangled);

/**
 * Demangle the kernel-name portion of a "name@source_file" function key.
 *
 * SASS and PC-sampling intern their per-symbol identity as
 * `function_name + "@" + source_file`, where `function_name` arrives
 * MANGLED from CUPTI (e.g. "_Z18distinct_kernel_26Pfi@/path/foo.cu").
 * Trace's kernel_dict names are already demangled (DemangleName on the
 * activity-record path), so without demangling here the very same kernel
 * carries two different identity strings across passes and the backend
 * multi-pass merge can never join them. Splits at the FIRST '@' (Itanium /
 * MSVC mangled names contain none), demangles the left part via
 * DemangleName, and re-appends the "@source_file" tail verbatim. A key with
 * no '@' is demangled whole.
 */
std::string DemangleFunctionKey(const std::string& function_key);
}  // namespace gpufl::core

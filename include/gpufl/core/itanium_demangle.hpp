#pragma once
#include <string>

namespace gpufl::core {

/**
 * Demangle an Itanium-ABI mangled C++ name (the format NVCC always emits
 * for CUDA device-side kernel names, regardless of host OS).
 *
 * Why this lives in the agent: Linux builds get full Itanium demangling
 * for free via `abi::__cxa_demangle` from libstdc++. Windows MSVC has
 * no equivalent — its `UnDecorateSymbolName` understands MSVC's
 * `?`-prefix mangling but not Itanium `_Z`-prefix. Without this fallback
 * Windows traces would ship raw `_ZN…` strings, breaking the frontend's
 * regex-based kernel categorization (GEMM / ELEMENTWISE / …) and making
 * kernel names unreadable in the UI.
 *
 * Scope: this is NOT a full Itanium ABI implementation. It handles the
 * subset of constructs that show up in real CUDA kernel mangling:
 *   - nested names         _ZN<segments>E
 *   - source names         <length><identifier>
 *   - template arguments   I<args>E
 *   - common type codes    v / i / f / d / j / l / m / b / etc.
 *   - cv-qualifiers        K (const), V (volatile)
 *   - pointer / reference  P / R
 *   - template params      T_, T0_, T1_, …
 *   - substitutions        S_, S0_, S1_, … (used for repeated types)
 *   - bare function types  return type + parameter list
 *
 * Constructs deliberately not handled (and the parser falls back to
 * returning the input unchanged when it hits them): operator names,
 * expressions inside template args (X…E), literals (L…E), argument
 * packs (J…E), local entities (Z…E), special names (T-mangled vtables,
 * etc.). These are vanishingly rare in CUDA device-side kernel names.
 *
 * On any parse error (unsupported construct, malformed input,
 * truncated string) the function returns the original mangled string
 * verbatim so callers can always display *something*. They won't get
 * a partial / wrong-looking name.
 */
std::string DemangleItaniumName(const char* mangled);

}  // namespace gpufl::core

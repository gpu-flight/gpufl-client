#pragma once

#include <string>

/**
 * Compile-time version + wire-format constants for the gpufl client.
 *
 * `GPUFL_CLIENT_VERSION` is injected by CMake from `project(... VERSION ...)`
 * via target_compile_definitions in the top-level CMakeLists.txt. The
 * fallback `"0.0.0-dev"` only kicks in when the library is consumed
 * outside the CMake build (header-only inclusion in a foreign build
 * system) — production builds always go through CMake and stamp the
 * real version.
 *
 * `kWireVersion` is the schema version of the JSON payloads this
 * client emits (the in-band `"version":<n>` field on every batch).
 * Bump only when the wire format makes a backwards-incompatible
 * change. Today's payloads are version "1".
 *
 * `kDefaultApiPath` is the URL prefix this client expects on the
 * backend. Hardcoded here because the client library version IS the
 * API version it speaks — letting users override the prefix is fine
 * (proxy/reverse-mount cases) but letting them pick a different
 * version would just produce parse errors. The override field is
 * `InitOptions.api_path`; this is the default applied when that's
 * left empty.
 */

namespace gpufl {

#ifndef GPUFL_CLIENT_VERSION
#define GPUFL_CLIENT_VERSION "0.0.0-dev"
#endif

inline constexpr const char* kClientVersion  = GPUFL_CLIENT_VERSION;
inline constexpr const char* kWireVersion    = "1";
inline constexpr const char* kDefaultApiPath = "/api/v1";

/**
 * Normalize a user-supplied API path so downstream URL composition is
 * always `(api_path) + "/events/<type>"` with exactly one slash join.
 *
 * Rules:
 *   - empty                        → kDefaultApiPath
 *   - missing leading slash        → prepend
 *   - one or more trailing slashes → strip
 *   - "/" (bare root after strip)  → kDefaultApiPath
 *
 * Result is guaranteed to start with `/` and never end with `/`.
 *
 * Defined inline here (rather than in a .cpp) so unit tests can call
 * it directly without linking against the full library, and so the
 * agent and any tooling reach the same canonical form.
 */
inline std::string normalizeApiPath(std::string p) {
    if (p.empty()) return kDefaultApiPath;
    if (p.front() != '/') p.insert(p.begin(), '/');
    while (p.size() > 1 && p.back() == '/') p.pop_back();
    if (p == "/") return kDefaultApiPath;
    return p;
}

}  // namespace gpufl

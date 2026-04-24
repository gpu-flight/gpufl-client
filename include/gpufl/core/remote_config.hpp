#pragma once

#include <string>

#include "gpufl/gpufl.hpp"

namespace gpufl {

/**
 * Synchronously fetch a named config from the GPUFlight backend and
 * apply whitelisted field overrides to {@code opts}.
 *
 * Performs {@code GET <base_url>/api/v1/config?config=<config_name>}
 * with the {@code X-API-Key} header set to {@code api_key}. On success,
 * parses the JSON response and merges a fixed set of InitOptions fields
 * in place (profiling_engine, sample rates, boolean flags — see the
 * allowlist in remote_config.cpp).
 *
 * Best-effort semantics: on any error (network, non-2xx, malformed
 * JSON, unparseable URL) the call logs a warning via GFL_LOG_ERROR and
 * leaves {@code opts} untouched. This is called synchronously from
 * {@code gpufl::init} before the monitor is started, so the 5-second
 * HTTP timeout caps the blocking period.
 *
 * This is declared in its own header (rather than inline in gpufl.cpp)
 * so the implementation TU (remote_config.cpp) can include
 * {@code <httplib.h>} without the windows.h / winsock2.h header
 * collision that plagued the previous in-place implementation. See
 * the plan file for details.
 */
void fetchRemoteConfig(const std::string& base_url,
                       const std::string& api_key,
                       const std::string& config_name,
                       InitOptions& opts);

}  // namespace gpufl

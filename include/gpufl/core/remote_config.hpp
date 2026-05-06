#pragma once

#include <string>

#include "gpufl/gpufl.hpp"

namespace gpufl {

/**
 * Synchronously fetch a named config from the GPUFlight backend and
 * apply whitelisted field overrides to {@code opts}.
 *
 * Performs {@code GET <base_url><api_path>/config?config=<config_name>}
 * with the {@code X-API-Key} header set to {@code api_key}. On success,
 * parses the JSON response and merges a fixed set of InitOptions fields
 * in place (profiling_engine, sample rates, boolean flags — see the
 * allowlist in remote_config.cpp).
 *
 * {@code api_path} must be pre-normalized by the caller: must start
 * with `/`, must not have a trailing slash. Empty string is treated
 * as `/api/v1` (the compiled-in default) so legacy callers that
 * passed only base_url + api_key still work.
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
                       const std::string& api_path,
                       const std::string& api_key,
                       const std::string& config_name,
                       InitOptions& opts);

/**
 * Fire-and-forget version discovery. GETs
 * {@code <base_url><api_path>/info/version} with 2-second connect/read
 * timeouts. If the response parses and the client's compiled-in
 * wire version (see {@code gpufl::kWireVersion}) is NOT in the
 * backend's {@code supported} list, emits a single warning via the
 * debug logger so the user sees a clear "client/server version drift"
 * message at init time.
 *
 * On any error (network timeout, non-2xx status, malformed JSON, the
 * endpoint not yet existing on an older backend) this function is a
 * silent no-op — it must NEVER block or fail the agent. Designed to
 * be called from a detached std::thread; the httplib timeouts are
 * the only bound on its lifetime.
 */
void probeBackendVersion(const std::string& base_url,
                         const std::string& api_path);

}  // namespace gpufl

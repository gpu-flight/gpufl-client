#pragma once

#include <string>

#include "gpufl/gpufl.hpp"

namespace gpufl {

// Historical note: this file previously also contained
// `fetchRemoteConfig`, which pulled a named profiling config from the
// backend's `/api/v1/config?config=<name>` endpoint and applied
// whitelisted overrides to InitOptions before the monitor started.
// That feature was removed alongside the backend's ConfigController
// and the SPA's ProfilingConfigPage. The version-discovery probe
// below survives because it's an orthogonal concern (client/server
// wire-version compatibility check, not config delivery).
//
// The filename is kept rather than renamed to `version_probe.{cpp,hpp}`
// purely to minimize churn for CMake and existing #include sites.

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

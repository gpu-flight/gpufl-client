#pragma once

#include <string>

namespace gpufl {

/**
 * Resolve the local machine's hostname (e.g. `gethostname()` /
 * `GetComputerName`). Result is cached on first call — these don't
 * change at runtime — and an empty string is returned on failure.
 *
 * Used to label session telemetry on its way to the backend so the
 * dashboard can group sessions by host. Both the file-tailing agent
 * path (reads NDJSON) and the direct HTTP-upload path (HttpLogSink
 * envelope) need this label, so we plumb it into both:
 *   * `job_start` event — top-level `hostname` field
 *   * `host_metric_batch` event — top-level `hostname` field
 *   * EventWrapper envelope (HttpLogSink only) — same field name
 */
std::string getLocalHostname();

/**
 * Best-effort IP address of the local machine. Currently always
 * returns an empty string (cross-platform IP enumeration is more
 * involved than hostname; the dashboard doesn't surface it yet).
 * Stub'd here so call sites can be future-proofed without refactoring.
 */
std::string getLocalIpAddr();

}  // namespace gpufl

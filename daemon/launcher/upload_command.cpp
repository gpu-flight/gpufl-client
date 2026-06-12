// Implements `gpufl upload <log_path> [opts]`.
//
// A thin front-end over gpufl::uploadLogs() - the same C++ core the
// Python `gpufl.upload_logs()` binding calls. Consolidating it here means
// one native `gpufl` binary owns trace + upload + version, replacing the
// separate Python `gpufl` console-script (which collided on the name).
//
// Portable C++ (no POSIX) - but it's compiled into the launcher binary,
// which is Linux-gated in CMake, so this only ships on Linux. Cross-
// platform callers use the gpufl.upload_logs() Python API instead.

#include "upload_command.hpp"

#include "gpufl/core/env_vars.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

#include "gpufl/upload/upload_logs.hpp"

namespace gpufl::launcher {

namespace {

// Flag value if set, else the env var, else empty.
std::string resolve(const std::string& flag, const char* env_name) {
    if (!flag.empty()) return flag;
    if (const char* v = std::getenv(env_name); v && *v) return v;
    return {};
}

}  // namespace

int runUpload(const UploadArgs& args) {
    const std::string backend_url = resolve(args.backend_url, gpufl::env::kBackendUrl);
    const std::string api_key     = resolve(args.api_key,     gpufl::env::kApiKey);

    if (backend_url.empty()) {
        std::fprintf(stderr,
            "gpufl upload: --backend-url required (or set GPUFL_BACKEND_URL)\n");
        return 2;
    }
    if (api_key.empty()) {
        std::fprintf(stderr,
            "gpufl upload: --api-key required (or set GPUFL_API_KEY)\n");
        return 2;
    }

    gpufl::UploadOptions opts;
    opts.log_path          = args.log_path;
    opts.backend_url       = backend_url;
    opts.api_key           = api_key;
    opts.api_path          = resolve(args.api_path, gpufl::env::kApiPath);
    opts.total_timeout_ms  = args.timeout_s * 1000;
    opts.max_retries       = args.retries;
    opts.report_progress   = !args.quiet;
    opts.session_id_filter = args.session_id;
    opts.all_sessions      = args.all_sessions;
    opts.force             = args.force;

    const gpufl::UploadResult r = gpufl::uploadLogs(opts);

    // Summary on stdout (machine-friendlier than the per-progress stderr).
    // Event counts are only known against legacy synchronous backends;
    // async-accept backends (202 + spool id) report none at POST time,
    // so bytes/files are the universal numbers.
    std::printf("Uploaded %.1f MB across %zu file(s) in %.1fs.\n",
                static_cast<double>(r.bytes_uploaded) / (1024.0 * 1024.0),
                r.files_processed,
                static_cast<double>(r.elapsed_ms) / 1000.0);
    if (r.events_uploaded > 0) {
        std::printf("Backend acknowledged %zu event(s) synchronously.\n",
                    r.events_uploaded);
    }
    if (r.files_skipped_by_cursor) {
        std::printf("Skipped %zu file(s) already uploaded (per cursor).\n",
                    r.files_skipped_by_cursor);
    }
    if (!r.warnings.empty()) {
        std::fprintf(stderr, "\n%zu warning(s):\n", r.warnings.size());
        for (const auto& w : r.warnings) {
            std::fprintf(stderr, "  - %s\n", w.c_str());
        }
    }

    if (r.success && r.warnings.empty()) return 0;  // clean
    if (r.success)                       return 1;  // uploaded, with warnings
    return 2;                                       // failed
}

}  // namespace gpufl::launcher

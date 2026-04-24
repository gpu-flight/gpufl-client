// Include httplib.h FIRST so its winsock2.h pull-in wins on Windows,
// before anything else could drag in <windows.h> (which would bring in
// legacy <winsock.h> and collide with winsock2). This TU intentionally
// does NOT include <windows.h> — any Windows-specific code stays in
// gpufl.cpp, which calls into this file via remote_config.hpp.
#include <httplib.h>

#include "gpufl/core/remote_config.hpp"

#include <memory>
#include <string>

#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/json/json.hpp"

namespace gpufl {

namespace {

/** URL-encode a string for use in a single query-parameter value. */
std::string urlEncode(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        const bool is_unreserved =
            (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
            (c >= '0' && c <= '9') ||
             c == '-' || c == '_' || c == '.' || c == '~';
        if (is_unreserved) {
            out.push_back(c);
        } else {
            constexpr char kHex[] = "0123456789ABCDEF";
            out.push_back('%');
            out.push_back(kHex[(static_cast<unsigned char>(c) >> 4) & 0xF]);
            out.push_back(kHex[ static_cast<unsigned char>(c)       & 0xF]);
        }
    }
    return out;
}

/**
 * Parse a backend base URL (e.g. "https://api.gpuflight.com" or
 * "http://localhost:8080") into (scheme, host, port). Mirrors the same
 * logic used by HttpLogSink — kept local because the duplication is
 * ~20 lines and sharing would require a public header that neither
 * consumer needs to export.
 */
bool parseBackendBaseUrl(const std::string& url,
                         std::string& scheme, std::string& host, int& port) {
    std::string u = url;
    while (!u.empty() && u.back() == '/') u.pop_back();
    const auto schemeEnd = u.find("://");
    if (schemeEnd != std::string::npos) {
        scheme = u.substr(0, schemeEnd);
        u = u.substr(schemeEnd + 3);
    } else {
        scheme = "http";
    }
    const auto pathStart = u.find('/');
    if (pathStart != std::string::npos) u = u.substr(0, pathStart);
    const auto colon = u.find(':');
    if (colon == std::string::npos) {
        host = u;
        port = -1;
    } else {
        host = u.substr(0, colon);
        try {
            port = std::stoi(u.substr(colon + 1));
        } catch (...) {
            return false;
        }
    }
    return !host.empty() && (scheme == "http" || scheme == "https");
}

/**
 * Apply a parsed remote-config JSON object to an InitOptions in place.
 *
 * Only a whitelist of fields is honored — matching what the Python
 * wrapper used to do — so the backend can't silently twist knobs we
 * haven't intentionally exposed. Unknown or wrong-type fields are
 * skipped.
 *
 * Precedence note: this path is lower priority than the caller's
 * kwargs — see the InitOptions comment block in gpufl.hpp. Because
 * this runs before env-var resolution and the caller's explicit
 * overrides in gpufl::init(), any explicit kwargs will still win.
 */
void applyRemoteConfigToOpts(const json::JsonValue& cfg, InitOptions& opts) {
    if (!cfg.is_object()) return;

    auto updateIfPresent = [&](const std::string& key, auto&& apply) {
        if (cfg.contains(key)) apply(cfg[key]);
    };

    updateIfPresent("profiling_engine", [&](const json::JsonValue& v) {
        if (!v.is_string()) return;
        const std::string& s = v.get_string();
        if      (s == "None")               opts.profiling_engine = ProfilingEngine::None;
        else if (s == "PcSampling")         opts.profiling_engine = ProfilingEngine::PcSampling;
        else if (s == "SassMetrics")        opts.profiling_engine = ProfilingEngine::SassMetrics;
        else if (s == "RangeProfiler")      opts.profiling_engine = ProfilingEngine::RangeProfiler;
        else if (s == "PcSamplingWithSass") opts.profiling_engine = ProfilingEngine::PcSamplingWithSass;
    });
    updateIfPresent("system_sample_rate_ms", [&](const json::JsonValue& v) {
        if (v.is_number()) opts.system_sample_rate_ms = static_cast<int>(v.as_int());
    });
    updateIfPresent("kernel_sample_rate_ms", [&](const json::JsonValue& v) {
        if (v.is_number()) opts.kernel_sample_rate_ms = static_cast<int>(v.as_int());
    });
    updateIfPresent("enable_stack_trace", [&](const json::JsonValue& v) {
        if (v.is_bool()) opts.enable_stack_trace = v.get_bool();
    });
    updateIfPresent("enable_kernel_details", [&](const json::JsonValue& v) {
        if (v.is_bool()) opts.enable_kernel_details = v.get_bool();
    });
    updateIfPresent("enable_source_collection", [&](const json::JsonValue& v) {
        if (v.is_bool()) opts.enable_source_collection = v.get_bool();
    });
    updateIfPresent("flush_logs_always", [&](const json::JsonValue& v) {
        if (v.is_bool()) opts.flush_logs_always = v.get_bool();
    });
}

}  // namespace

// Public entry point declared in remote_config.hpp — called from
// gpufl::init(). Best-effort: on any error (network, non-2xx, malformed
// JSON, unparseable URL) we log a warning and leave `opts` untouched.
//
// This is called before Monitor::Initialize() — the HttpLogSink worker
// thread isn't running yet, so the synchronous I/O here is the only
// blocking step in init(), and it's capped by a 5-second timeout.
void fetchRemoteConfig(const std::string& base_url,
                       const std::string& api_key,
                       const std::string& config_name,
                       InitOptions& opts) {
    std::string scheme, host;
    int port = -1;
    if (!parseBackendBaseUrl(base_url, scheme, host, port)) {
        GFL_LOG_ERROR("[fetchRemoteConfig] invalid backend_url: ",
                      base_url);
        return;
    }

    // Use the unified httplib::Client(scheme_host_port) constructor —
    // internally dispatches to SSLClient when scheme is https AND
    // CPPHTTPLIB_OPENSSL_SUPPORT is defined. Avoids the type mismatch
    // between unique_ptr<SSLClient> and unique_ptr<Client> (those are
    // sibling types in cpp-httplib, not parent/child).
#if !GPUFL_HTTPLIB_TLS
    if (scheme == "https") {
        GFL_LOG_ERROR(
            "[fetchRemoteConfig] https:// remote config requires OpenSSL "
            "(rebuild gpufl with OpenSSL available). Skipping fetch.");
        return;
    }
#endif
    std::string scheme_host_port = scheme + "://" + host;
    if (port > 0) scheme_host_port += ":" + std::to_string(port);
    auto cli = std::make_unique<httplib::Client>(scheme_host_port);
    cli->set_connection_timeout(5, 0);
    cli->set_read_timeout(5, 0);

    std::string path = "/api/v1/config";
    if (!config_name.empty()) {
        path += "?config=" + urlEncode(config_name);
    }
    httplib::Headers headers = {
        // Python used X-API-Key; match that here so the same backend
        // auth path (ConfigController :: X-API-Key resolver) is
        // exercised. If we later switch to Authorization: Bearer for
        // both config and upload, flip this in lockstep with the
        // backend's SecurityConfig.
        {"X-API-Key", api_key},
        {"Accept",    "application/json"},
    };

    auto res = cli->Get(path.c_str(), headers);
    if (!res) {
        GFL_LOG_ERROR(
            "[fetchRemoteConfig] transport error (", scheme, "://",
            host, path, ") — continuing with local config.");
        return;
    }
    if (res->status < 200 || res->status >= 300) {
        GFL_LOG_ERROR(
            "[fetchRemoteConfig] status=", res->status,
            " — continuing with local config.");
        return;
    }
    json::JsonValue cfg = json::parseJson(res->body);
    if (!cfg.is_object()) {
        GFL_LOG_ERROR(
            "[fetchRemoteConfig] response was not a JSON object — ignoring.");
        return;
    }
    applyRemoteConfigToOpts(cfg, opts);
    GFL_LOG_DEBUG(
        "[fetchRemoteConfig] applied remote config from ", base_url);
}

}  // namespace gpufl

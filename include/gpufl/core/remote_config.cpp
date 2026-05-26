// Include httplib.h FIRST so its winsock2.h pull-in wins on Windows,
// before anything else could drag in <windows.h> (which would bring in
// legacy <winsock.h> and collide with winsock2). This TU intentionally
// does NOT include <windows.h> — any Windows-specific code stays in
// gpufl.cpp, which calls into this file via remote_config.hpp.
#include <httplib.h>

#include "gpufl/core/remote_config.hpp"

#include <string>

#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/json/json.hpp"
#include "gpufl/core/version.hpp"

namespace gpufl {

namespace {

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

}  // namespace

// Best-effort version-discovery probe — see remote_config.hpp doc.
// Must never throw, never block beyond ~4s (2s connect + 2s read), and
// never disable the agent on failure. Called from a detached thread in
// gpufl::init().
void probeBackendVersion(const std::string& base_url,
                         const std::string& api_path) {
    std::string scheme, host;
    int port = -1;
    if (!parseBackendBaseUrl(base_url, scheme, host, port)) {
        return;  // bad URL — silent: the main config path will log it
    }
#if !GPUFL_HTTPLIB_TLS
    if (scheme == "https") return;  // can't probe TLS without OpenSSL
#endif
    std::string scheme_host_port = scheme + "://" + host;
    if (port > 0) scheme_host_port += ":" + std::to_string(port);
    httplib::Client cli(scheme_host_port);
    cli.set_connection_timeout(2, 0);
    cli.set_read_timeout(2, 0);

    const std::string ap = api_path.empty() ? std::string(kDefaultApiPath)
                                            : api_path;
    const std::string path = ap + "/info/version";
    // Per-call headers — cpp-httplib auto-injects its own User-Agent on
    // send and that beats set_default_headers in the merge order, so
    // per-call wins reliably across cpp-httplib versions.
    httplib::Headers headers = {
        {"User-Agent",                 std::string("gpufl/") + kClientVersion},
        {"X-GpuFlight-Client-Version", kClientVersion},
        {"X-GpuFlight-Wire-Version",   kWireVersion},
    };
    auto res = cli.Get(path.c_str(), headers);
    if (!res || res->status < 200 || res->status >= 300) {
        return;  // 404 / network error / older backend — silent
    }
    json::JsonValue resp = json::parseJson(res->body);
    if (!resp.is_object() || !resp.contains("supported")) return;
    const auto& sup = resp.at("supported");
    if (!sup.is_array()) return;

    bool found = false;
    std::string supported_joined;
    for (const auto& v : sup.get_array()) {
        if (!v.is_string()) continue;
        const auto& s = v.get_string();
        if (!supported_joined.empty()) supported_joined += ",";
        supported_joined += s;
        if (s == kWireVersion) { found = true; }
    }
    if (!found) {
        GFL_LOG_ERROR(
            "[gpufl] client wire-version=", kWireVersion,
            " not in backend supported=[", supported_joined,
            "] — uploads may fail. Upgrade the agent or the backend.");
    } else {
        GFL_LOG_DEBUG(
            "[probeBackendVersion] OK: wire=", kWireVersion,
            " supported=[", supported_joined, "]");
    }
}

}  // namespace gpufl

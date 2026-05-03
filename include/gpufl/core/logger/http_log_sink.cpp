#include "gpufl/core/logger/http_log_sink.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>

// cpp-httplib is a single-header library. With CPPHTTPLIB_OPENSSL_SUPPORT
// defined (set via GPUFL_HTTPLIB_TLS in CMake when OpenSSL is found) it
// also supports https:// endpoints via httplib::SSLClient.
#include <httplib.h>

#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/host_info.hpp"
#include "gpufl/core/json/json.hpp"

namespace gpufl {

namespace {

/**
 * Parse a backend base URL into (scheme, host, port).
 * Accepts:  "http://host", "https://host", "http://host:port", "host:port".
 * Strips a trailing slash. Returns false on obviously malformed input.
 */
bool parseBaseUrl(const std::string& url,
                  std::string& scheme, std::string& host, int& port) {
    std::string u = url;
    while (!u.empty() && u.back() == '/') u.pop_back();

    // Scheme
    const auto schemeEnd = u.find("://");
    if (schemeEnd != std::string::npos) {
        scheme = u.substr(0, schemeEnd);
        u = u.substr(schemeEnd + 3);
    } else {
        scheme = "http";
    }

    // Drop any path component — the sink appends its own path.
    const auto pathStart = u.find('/');
    if (pathStart != std::string::npos) u = u.substr(0, pathStart);

    // Split host:port
    const auto colon = u.find(':');
    if (colon == std::string::npos) {
        host = u;
        port = -1;  // use default for scheme
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

/** Build a connected httplib client for the given scheme/host/port.
 *
 * Uses the unified `httplib::Client(scheme_host_port)` constructor,
 * which internally delegates to SSLClient when the URL scheme is
 * `https://` (and CPPHTTPLIB_OPENSSL_SUPPORT was defined at build
 * time). Constructing the wrapper Client directly avoids the type
 * mismatch that bit us on Linux CI: `SSLClient` does NOT inherit
 * from `Client` in cpp-httplib — they're siblings that both wrap
 * `ClientImpl` — so `unique_ptr<SSLClient>` can't be assigned into
 * `unique_ptr<Client>`.
 */
std::unique_ptr<httplib::Client> makeClient(
        const std::string& scheme, const std::string& host, int port,
        int connect_timeout_ms, int read_timeout_ms) {
#if !GPUFL_HTTPLIB_TLS
    if (scheme == "https") {
        GFL_LOG_ERROR(
            "[HttpLogSink] https:// requested but OpenSSL was not linked "
            "at build time. Falling back to http:// — TLS uploads will "
            "fail until gpufl is rebuilt with OpenSSL available.");
    }
#endif
    std::string scheme_host_port = scheme + "://" + host;
    if (port > 0) scheme_host_port += ":" + std::to_string(port);
    auto cli = std::make_unique<httplib::Client>(scheme_host_port);
    cli->set_connection_timeout(0, connect_timeout_ms * 1000);
    cli->set_read_timeout(0, read_timeout_ms * 1000);
    cli->set_keep_alive(true);
    return cli;
}

}  // namespace

// --- HttpLogSink ---

HttpLogSink::HttpLogSink(Options opts)
    : opts_(std::move(opts)) {
    GFL_LOG_DEBUG("[HttpLogSink] ctor base_url='", opts_.base_url,
                  "' api_key_len=", opts_.api_key.size(),
                  " queue_capacity=", opts_.queue_capacity);
    if (opts_.base_url.empty() || opts_.api_key.empty()) {
        GFL_LOG_ERROR(
            "[HttpLogSink] base_url or api_key empty — upload disabled.");
        return;
    }
    if (!parseBaseUrl(opts_.base_url, scheme_, host_, port_)) {
        GFL_LOG_ERROR(
            "[HttpLogSink] Invalid base_url '", opts_.base_url,
            "' — upload disabled.");
        return;
    }
#if GPUFL_HTTPLIB_TLS
    ssl_supported_ = true;
#endif
    GFL_LOG_DEBUG("[HttpLogSink] parsed URL: scheme=", scheme_,
                  " host=", host_, " port=", port_,
                  " → worker thread starting");
    running_.store(true, std::memory_order_release);
    worker_ = std::thread([this] { workerLoop(); });
}

HttpLogSink::~HttpLogSink() { close(); }

void HttpLogSink::write(Channel /*ch*/, std::string_view json) {
    if (!running_.load(std::memory_order_acquire)) {
        GFL_LOG_DEBUG("[HttpLogSink] write skipped: running_=false "
                      "(ctor failed or close() already ran)");
        return;
    }
    if (auth_failed_.load(std::memory_order_acquire)) {
        GFL_LOG_DEBUG("[HttpLogSink] write skipped: auth_failed_=true "
                      "(backend returned 401/403 earlier)");
        return;
    }

    {
        std::lock_guard<std::mutex> lk(queue_mu_);
        if (queue_.size() >= opts_.queue_capacity) {
            queue_.pop_front();  // drop oldest — we prefer fresh data
            dropped_.fetch_add(1, std::memory_order_relaxed);
            if (!overflow_warned_.exchange(true)) {
                GFL_LOG_ERROR(
                    "[HttpLogSink] queue full (", opts_.queue_capacity,
                    " entries); dropping oldest. Backend may be slow or "
                    "unreachable. File sink still persists everything — "
                    "use the agent to back-fill later.");
            }
        }
        queue_.emplace_back(json.data(), json.size());
        enqueued_.fetch_add(1, std::memory_order_relaxed);
    }
    queue_cv_.notify_one();
    GFL_LOG_DEBUG("[HttpLogSink] write enqueued ", json.size(),
                  " bytes; queue size now=", enqueued_.load() -
                  uploaded_.load() - dropped_.load() - failed_.load());
}

void HttpLogSink::close() {
    // Signal shutdown, then give the worker a bounded grace period to
    // drain. This call is idempotent.
    if (!running_.exchange(false, std::memory_order_acq_rel)) {
        if (worker_.joinable()) worker_.join();
        return;
    }
    closing_.store(true, std::memory_order_release);
    queue_cv_.notify_all();

    const auto deadline =
        std::chrono::steady_clock::now() +
        std::chrono::milliseconds(opts_.shutdown_drain_ms);
    // Poll-join: we want to time-box the join rather than wait
    // indefinitely for a stuck backend.
    while (worker_.joinable()
           && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        std::lock_guard<std::mutex> lk(queue_mu_);
        if (queue_.empty()) break;
    }
    if (worker_.joinable()) worker_.join();

    // End-of-session bandwidth + delivery summary. Logged once at
    // close() so users can quickly see how much data this session
    // actually shipped to the backend without grep-scanning the
    // per-request DEBUG lines. Uses DEBUG level (matches the rest
    // of this file) so it shows up only when DebugLogger is enabled
    // — typical for users investigating bandwidth/cost concerns.
    const std::size_t enq    = enqueued_.load(std::memory_order_relaxed);
    const std::size_t up     = uploaded_.load(std::memory_order_relaxed);
    const std::size_t drp    = dropped_.load(std::memory_order_relaxed);
    const std::size_t fl     = failed_.load(std::memory_order_relaxed);
    const std::size_t bytes  = bytes_uploaded_.load(std::memory_order_relaxed);
    GFL_LOG_DEBUG(
        "[HttpLogSink] session totals: ",
        bytes, " bytes sent over HTTP across ",
        up, " uploaded line(s) "
        "(enqueued=", enq, ", dropped=", drp, ", failed=", fl, ")");
}

// --- Worker ---

void HttpLogSink::workerLoop() {
    GFL_LOG_DEBUG("[HttpLogSink] worker thread running (scheme=", scheme_,
                  " host=", host_, " port=", port_, ")");
    auto client = makeClient(scheme_, host_, port_,
                             opts_.connect_timeout_ms, opts_.read_timeout_ms);
    if (!client) {
        GFL_LOG_ERROR(
            "[HttpLogSink] makeClient returned null — worker exiting");
        return;
    }
    // Per-POST gzip compression was removed — see header note above
    // Options. Live event POSTs go uncompressed; bandwidth-conscious
    // users run gpufl-agent against the FileLogSink NDJSON for
    // batch-compressed uploads (which achieve a much better ratio).

    while (true) {
        std::string line;
        {
            std::unique_lock<std::mutex> lk(queue_mu_);
            queue_cv_.wait_for(lk, std::chrono::milliseconds(200), [this] {
                return !queue_.empty()
                    || !running_.load(std::memory_order_acquire);
            });
            if (queue_.empty()) {
                if (!running_.load(std::memory_order_acquire)) {
                    GFL_LOG_DEBUG(
                        "[HttpLogSink] worker exiting: running_=false + "
                        "queue empty (uploaded=", uploaded_.load(),
                        " failed=", failed_.load(),
                        " dropped=", dropped_.load(),
                        " enqueued=", enqueued_.load(), ")");
                    return;
                }
                continue;
            }
            line = std::move(queue_.front());
            queue_.pop_front();
        }

        const std::string_view type = extractType(line);
        if (type.empty()) {
            failed_.fetch_add(1, std::memory_order_relaxed);
            GFL_LOG_DEBUG(
                "[HttpLogSink] dropping line with no parseable 'type' field: ",
                line.substr(0, 120));
            continue;
        }
        if (auth_failed_.load(std::memory_order_acquire)) continue;

        // Wrap the raw NDJSON line in the EventWrapper envelope the
        // backend's EventIngestionController expects:
        //
        //   {"data":"<NDJSON string>","agentSendingTime":<ms>,
        //    "hostname":"<localhost>","ipAddr":"<local-ip>"}
        //
        // `data` is a STRING (the NDJSON line JSON-escaped), NOT nested
        // JSON — matches com.gpuflight.gpuflbackend.model.EventWrapper
        // which has `String data`. Sending the raw NDJSON here makes
        // Jackson bind it to EventWrapper with `data=null`, which then
        // crashes in SystemEventServiceImpl.addSystemEvent() trying to
        // readValue(null, ...). This wrapping matches what the monitor
        // daemon / agent does for production file-tailing ingestion.
        //
        // hostname + ipAddr are populated via getLocalHostname() /
        // getLocalIpAddr(). InitEventServiceImpl reads these off
        // EventWrapper and writes them onto SessionEntity, so the
        // dashboard's Sessions page can label each session by host.
        // Field names use camelCase (ipAddr, not ip_addr) to match the
        // EventWrapper Java record's default Jackson binding.
        const long long nowMs =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
        const std::string& hostName = getLocalHostname();
        const std::string& ipAddr   = getLocalIpAddr();
        std::string wrapped;
        wrapped.reserve(line.size() + 128 + hostName.size() + ipAddr.size());
        wrapped.append("{\"data\":\"");
        wrapped.append(json::escape(line));
        wrapped.append("\",\"agentSendingTime\":");
        wrapped.append(std::to_string(nowMs));
        wrapped.append(",\"hostname\":\"");
        wrapped.append(json::escape(hostName));
        wrapped.append("\",\"ipAddr\":\"");
        wrapped.append(json::escape(ipAddr));
        wrapped.append("\"}");

        // Retry loop with exponential backoff: 50ms, 200ms, 1s. Budget
        // comes from Options; a 401/403 aborts immediately (retry won't
        // help) and disables the sink for the rest of the session.
        int delay_ms = 50;
        bool ok = false;
        for (int attempt = 0; attempt <= opts_.max_retries; ++attempt) {
            const std::string path =
                "/api/v1/events/" + std::string(type);
            GFL_LOG_DEBUG(
                "[HttpLogSink] POST attempt=", attempt,
                " ", scheme_, "://", host_, ":", port_, path,
                " body_bytes=", wrapped.size(),
                " (inner_ndjson_bytes=", line.size(), ")");
            httplib::Headers headers = {
                {"Authorization", "Bearer " + opts_.api_key},
                {"Content-Type",  "application/json"},
            };
            auto res = client->Post(path.c_str(), headers, wrapped,
                                    "application/json");
            if (!res) {
                const auto err = res.error();
                GFL_LOG_ERROR(
                    "[HttpLogSink] POST ", path,
                    " transport error (httplib::Error=", static_cast<int>(err),
                    ") attempt=", attempt, "/", opts_.max_retries);
                // Transport-level failure (conn refused, TLS error, etc.)
                if (attempt < opts_.max_retries) {
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(delay_ms));
                    delay_ms *= 4;
                }
                continue;
            }
            const int status = res->status;
            GFL_LOG_DEBUG(
                "[HttpLogSink] POST ", path, " → status=", status,
                " body_bytes=", res->body.size());
            if (status >= 200 && status < 300) {
                ok = true;
                break;
            }
            if (status == 401 || status == 403) {
                if (!auth_warned_.exchange(true)) {
                    GFL_LOG_ERROR(
                        "[HttpLogSink] backend returned ", status,
                        " — API key is missing or invalid. Disabling "
                        "upload for this session; File sink continues "
                        "normally. Response body: ",
                        res->body.substr(0, 200));
                }
                auth_failed_.store(true, std::memory_order_release);
                break;
            }
            if (status >= 400 && status < 500) {
                // Client error (malformed body, wrong endpoint) — retry
                // won't fix it.
                GFL_LOG_ERROR(
                    "[HttpLogSink] 4xx response (", status,
                    ") for ", path, " — dropping line. Body: ",
                    res->body.substr(0, 200));
                break;
            }
            // 5xx: worth retrying.
            GFL_LOG_DEBUG(
                "[HttpLogSink] 5xx response (", status,
                ") — retrying after ", delay_ms, "ms");
            if (attempt < opts_.max_retries) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(delay_ms));
                delay_ms *= 4;
            }
        }

        if (ok) {
            uploaded_.fetch_add(1, std::memory_order_relaxed);
            // Tally the request body bytes that actually went over
            // the wire on this successful POST. `wrapped` is the
            // post-envelope payload (data + agentSendingTime +
            // hostname + ipAddr) — what the network and the backend
            // actually saw, not the raw event size.
            bytes_uploaded_.fetch_add(wrapped.size(),
                                      std::memory_order_relaxed);
            GFL_LOG_DEBUG(
                "[HttpLogSink] upload OK (total uploaded=",
                uploaded_.load() + 1, ")");
        } else {
            failed_.fetch_add(1, std::memory_order_relaxed);
            GFL_LOG_DEBUG(
                "[HttpLogSink] upload FAILED after ",
                opts_.max_retries + 1, " attempts (total failed=",
                failed_.load() + 1, ")");
        }
    }
}

// --- Helpers ---

// Extract the "type" field value from a JSON line without doing a full
// parse. We expect every event to have "type":"<string>" near the top.
// This is O(n) and allocates nothing — worth it because extractType()
// runs on every uploaded line.
std::string_view HttpLogSink::extractType(std::string_view json) {
    constexpr std::string_view kKey = "\"type\"";
    auto pos = json.find(kKey);
    if (pos == std::string_view::npos) return {};
    pos += kKey.size();
    // Skip whitespace + ':'
    while (pos < json.size() &&
           (json[pos] == ' ' || json[pos] == ':')) ++pos;
    if (pos >= json.size() || json[pos] != '"') return {};
    const auto valueStart = pos + 1;
    const auto valueEnd = json.find('"', valueStart);
    if (valueEnd == std::string_view::npos) return {};
    return json.substr(valueStart, valueEnd - valueStart);
}

}  // namespace gpufl

// Implementation of gpufl::uploadLogs — see upload_logs.hpp for the
// API contract and ~/.claude/plans/deferred-upload-only.md for design.
//
// High-level structure:
//   1. parse log_path → (directory, file-prefix)
//   2. discover .log / .log.gz files matching the prefix; sort by
//      (channel, rotation_index DESC, active-file last)
//   3. read cursor file → set of filenames already uploaded
//   4. for each file in order:
//        - if in cursor (and not the active file): skip
//        - open (decompress if .gz), stream NDJSON line by line
//        - for each line: extract `"type"`, route to POST
//          * `shutdown` → hold in a deferred-tail buffer, POST after all files
//          * everything else (including `job_start`, which appears first
//            naturally because it lives in the oldest file's first line)
//            → POST immediately
//      after a successful pass over a non-active file: append to cursor
//   5. POST the held `shutdown` events last so the backend's
//      session-lifecycle bookkeeping sees a clean job_start → … → shutdown
//      sequence in arrival order
//   6. enforce total_timeout_ms across the whole loop — abort + return
//      success=false on overrun rather than blocking the host process

#include "gpufl/upload/upload_logs.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <regex>
#include <sstream>
#include <thread>
#include <unordered_set>
#include <utility>

#include <httplib.h>
#include <zlib.h>

#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/host_info.hpp"
#include "gpufl/core/json/json.hpp"
#include "gpufl/core/version.hpp"

namespace gpufl {
namespace fs = std::filesystem;
namespace {

// ─────────────────────────────────────────────────────────────────────
// File discovery + classification
// ─────────────────────────────────────────────────────────────────────

struct DiscoveredFile {
    fs::path     path;
    std::string  channel;         // e.g. "device", "scope", "system"
    int          rotation_index;  // 0 = active .log file; ≥1 = rotated
    bool         compressed;      // true if .log.gz
};

/// Split `log_path` ("/tmp/foo/bar" or "/tmp/foo/bar.log") into the
/// directory and the file-prefix used by LogFileRotator::basePrefix().
struct PathParts {
    fs::path    directory;
    std::string prefix;  // filename component, with any .log suffix stripped
};

PathParts splitLogPath(const std::string& log_path) {
    fs::path p(log_path);
    PathParts out;
    // Mirror LogFileRotator::basePrefix(): strip a trailing .log so
    // callers can pass either "/dir/app" or "/dir/app.log" and get the
    // same files discovered.
    if (p.extension() == ".log") p.replace_extension();
    out.directory = p.has_parent_path() ? p.parent_path() : fs::current_path();
    out.prefix    = p.filename().string();
    return out;
}

/// Parse a filename like "app.device.log" or "app.device.5.log.gz" into
/// its components. Returns false if the name doesn't match either shape
/// or doesn't start with `prefix + "."`.
bool parseLogFilename(const std::string& filename,
                      const std::string& prefix,
                      std::string& channel_out,
                      int& rotation_index_out,
                      bool& compressed_out) {
    // Expected: {prefix}.{channel}[.{index}].log[.gz]
    const std::string head = prefix + ".";
    if (filename.size() <= head.size() ||
        filename.compare(0, head.size(), head) != 0) {
        return false;
    }
    std::string rest = filename.substr(head.size());

    // Strip optional .gz
    compressed_out = false;
    if (rest.size() > 3 && rest.compare(rest.size() - 3, 3, ".gz") == 0) {
        compressed_out = true;
        rest.erase(rest.size() - 3);
    }
    // Must end in .log
    if (rest.size() <= 4 || rest.compare(rest.size() - 4, 4, ".log") != 0) {
        return false;
    }
    rest.erase(rest.size() - 4);  // strip .log

    // Now rest is either "{channel}" or "{channel}.{index}"
    const auto last_dot = rest.find_last_of('.');
    if (last_dot == std::string::npos) {
        // active file: no rotation index
        channel_out = rest;
        rotation_index_out = 0;
        return !channel_out.empty();
    }
    // Try to parse trailing component as an integer
    const std::string tail = rest.substr(last_dot + 1);
    try {
        const int idx = std::stoi(tail);
        if (idx <= 0) return false;
        rotation_index_out = idx;
        channel_out = rest.substr(0, last_dot);
        return !channel_out.empty();
    } catch (...) {
        // No trailing integer → treat whole thing as channel name
        // (e.g. a channel name that happens to contain a dot — unusual
        // but not technically forbidden). Active file.
        channel_out = rest;
        rotation_index_out = 0;
        return !channel_out.empty();
    }
}

std::vector<DiscoveredFile> discoverFiles(const PathParts& parts) {
    std::vector<DiscoveredFile> out;
    std::error_code ec;
    if (!fs::exists(parts.directory, ec) || !fs::is_directory(parts.directory, ec)) {
        return out;
    }
    for (const auto& entry : fs::directory_iterator(parts.directory, ec)) {
        if (!entry.is_regular_file(ec)) continue;
        const std::string fname = entry.path().filename().string();
        DiscoveredFile df;
        if (!parseLogFilename(fname, parts.prefix,
                              df.channel, df.rotation_index, df.compressed)) {
            continue;
        }
        df.path = entry.path();
        out.push_back(std::move(df));
    }

    // Sort: by channel (stable lexicographic), then by upload order
    // within a channel — oldest first. Rotation index N=max_files is
    // the oldest (rotated longest ago); active (N=0) is newest. So
    // within a channel: descending rotation_index, with active (0) last.
    std::sort(out.begin(), out.end(),
              [](const DiscoveredFile& a, const DiscoveredFile& b) {
                  if (a.channel != b.channel) return a.channel < b.channel;
                  // Within channel: rotated files first (in DESCENDING
                  // index order, since high index = oldest), then the
                  // active file (index 0) last.
                  const bool a_active = a.rotation_index == 0;
                  const bool b_active = b.rotation_index == 0;
                  if (a_active != b_active) return !a_active;  // active goes last
                  return a.rotation_index > b.rotation_index;  // higher = older = first
              });
    return out;
}

// ─────────────────────────────────────────────────────────────────────
// NDJSON line reader (transparent gzip support)
// ─────────────────────────────────────────────────────────────────────

class NdjsonReader {
   public:
    explicit NdjsonReader(const fs::path& p, const bool gz) : path_(p), gz_(gz) {
        if (gz_) {
            gzfile_ = gzopen(p.string().c_str(), "rb");
        } else {
            stream_.open(p);
        }
    }
    ~NdjsonReader() {
        if (gzfile_) gzclose(gzfile_);
    }

    bool ok() const { return gz_ ? (gzfile_ != nullptr) : stream_.is_open(); }

    /// Read one line into `out`. Returns false at EOF or on error.
    bool readLine(std::string& out) {
        out.clear();
        if (gz_) {
            char buf[4096];
            // gzgets reads until newline or buffer-full. We loop in case
            // a single NDJSON line is larger than the buffer (some
            // kernel_event_batch payloads are several KB).
            while (gzgets(gzfile_, buf, sizeof(buf)) != nullptr) {
                out.append(buf);
                if (!out.empty() && out.back() == '\n') {
                    out.pop_back();
                    return true;
                }
            }
            return !out.empty();
        }
        if (!std::getline(stream_, out)) return false;
        // Handle CRLF on Windows: getline strips '\n' but leaves '\r'
        if (!out.empty() && out.back() == '\r') out.pop_back();
        return true;
    }

   private:
    fs::path      path_;
    bool          gz_;
    std::ifstream stream_;
    gzFile        gzfile_ = nullptr;
};

// ─────────────────────────────────────────────────────────────────────
// Cursor file (.gpufl-upload-cursor.json)
// ─────────────────────────────────────────────────────────────────────
//
// Schema v2:
//   {
//     "schema_version": 2,
//     "uploaded_files":      ["app.device.1.log.gz", ...],
//     "completed_sessions": {
//       "<session_id>": {"completed_at": "2026-05-26T15:30:00Z",
//                        "events": 1234}
//     }
//   }
//
// v1 (only `uploaded_files`, no `completed_sessions`) is read
// transparently — missing keys default to empty collections. The next
// write upgrades the file to v2.

struct CompletedSession {
    std::string completed_at_iso8601;
    std::size_t events = 0;
};

struct Cursor {
    std::unordered_set<std::string> uploaded_files;     // basename only
    std::map<std::string, CompletedSession> completed_sessions;  // session_id → details
};

Cursor readCursor(const fs::path& dir, const std::string& cursor_filename) {
    Cursor c;
    const fs::path cursor_path = dir / cursor_filename;
    std::error_code ec;
    if (!fs::exists(cursor_path, ec)) return c;

    const json::JsonValue v = json::loadFile(cursor_path.string());
    if (!v.is_object()) return c;

    // uploaded_files (both v1 and v2 have this)
    if (v.contains("uploaded_files")) {
        const json::JsonValue& list = v.at("uploaded_files");
        if (list.is_array()) {
            for (size_t i = 0; i < list.size(); ++i) {
                const json::JsonValue& e = list[i];
                if (e.is_string()) c.uploaded_files.insert(e.get_string());
            }
        }
    }

    // completed_sessions (v2+). Missing in v1 → empty map → upgrade
    // on next write. No error path needed.
    if (v.contains("completed_sessions")) {
        const json::JsonValue& obj = v.at("completed_sessions");
        if (obj.is_object()) {
            for (const auto& [sid, detail] : obj) {
                if (!detail.is_object()) continue;
                CompletedSession cs;
                if (detail.contains("completed_at") &&
                    detail.at("completed_at").is_string()) {
                    cs.completed_at_iso8601 = detail.at("completed_at").get_string();
                }
                if (detail.contains("events")) {
                    cs.events = static_cast<std::size_t>(detail.at("events").as_int(0));
                }
                c.completed_sessions[sid] = cs;
            }
        }
    }
    return c;
}

/// Format current UTC time as ISO-8601 (`2026-05-26T15:30:00Z`).
/// Used purely for the cursor's `completed_at` field; humans glance at
/// this when debugging "when did this session last upload."
std::string nowIso8601Utc() {
    const std::time_t now = std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now());
    std::tm tm_utc{};
#if defined(_MSC_VER)
    gmtime_s(&tm_utc, &now);
#else
    gmtime_r(&now, &tm_utc);
#endif
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02dZ",
                  tm_utc.tm_year + 1900, tm_utc.tm_mon + 1, tm_utc.tm_mday,
                  tm_utc.tm_hour, tm_utc.tm_min, tm_utc.tm_sec);
    return buf;
}

/// Writes cursor atomically: write to .tmp, then rename. Avoids leaving
/// a half-written cursor file if the process dies mid-write. Always
/// writes v2 schema regardless of the input cursor's origin version.
bool writeCursor(const fs::path& dir, const std::string& cursor_filename,
                 const Cursor& c) {
    const fs::path cursor_path = dir / cursor_filename;
    const fs::path tmp_path    = dir / (cursor_filename + ".tmp");

    // Build JSON manually — schema is tiny and we don't have a JSON
    // serializer in this project (parser only).
    std::ostringstream oss;
    oss << "{\"schema_version\":2,\"uploaded_files\":[";
    bool first = true;
    for (const auto& f : c.uploaded_files) {
        if (!first) oss << ",";
        oss << "\"" << json::escape(f) << "\"";
        first = false;
    }
    oss << "],\"completed_sessions\":{";
    first = true;
    for (const auto& [sid, detail] : c.completed_sessions) {
        if (!first) oss << ",";
        oss << "\"" << json::escape(sid) << "\":{"
            << "\"completed_at\":\"" << json::escape(detail.completed_at_iso8601)
            << "\",\"events\":" << detail.events << "}";
        first = false;
    }
    oss << "}}";

    {
        std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
        if (!out) return false;
        out << oss.str();
        out.flush();
        if (!out) return false;
    }

    std::error_code ec;
    fs::rename(tmp_path, cursor_path, ec);
    if (ec) {
        fs::remove(tmp_path, ec);
        return false;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────
// Event type extraction
// ─────────────────────────────────────────────────────────────────────

/// Extract just the `"type":"<value>"` substring from an NDJSON line
/// without parsing the whole JSON. Much faster on big batch events,
/// where a full parse would walk every row.
///
/// Returns empty string if the type field can't be found cheaply — the
/// caller falls back to a real parse in that case.
std::string fastExtractType(const std::string& line) {
    static const std::string kKey = "\"type\":\"";
    const auto pos = line.find(kKey);
    if (pos == std::string::npos) return {};
    const auto start = pos + kKey.size();
    const auto end = line.find('"', start);
    if (end == std::string::npos) return {};
    return line.substr(start, end - start);
}

std::string extractType(const std::string& line) {
    std::string t = fastExtractType(line);
    if (!t.empty()) return t;
    // Fallback: structured parse. Slow but reliable for unusual line shapes.
    const json::JsonValue v = json::parseJson(line);
    if (v.is_object() && v.contains("type") && v.at("type").is_string()) {
        return v.at("type").get_string();
    }
    return {};
}

/// Same idea for the `"session_id":"<uuid>"` field. Every event the
/// client emits carries one at the top level, so a single substring
/// search per line is both correct and ~100× faster than a full JSON
/// parse.
std::string fastExtractSessionId(const std::string& line) {
    static const std::string kKey = "\"session_id\":\"";
    const auto pos = line.find(kKey);
    if (pos == std::string::npos) return {};
    const auto start = pos + kKey.size();
    const auto end = line.find('"', start);
    if (end == std::string::npos) return {};
    return line.substr(start, end - start);
}

std::string extractSessionId(const std::string& line) {
    std::string s = fastExtractSessionId(line);
    if (!s.empty()) return s;
    const json::JsonValue v = json::parseJson(line);
    if (v.is_object() && v.contains("session_id") &&
        v.at("session_id").is_string()) {
        return v.at("session_id").get_string();
    }
    return {};
}

/// Pull a numeric `ts_ns` out of an NDJSON line. Used only on
/// `job_start` events to pick the "latest" session by timestamp.
/// Returns 0 if the field is missing or unparseable — sorts to the
/// oldest, which is the safe fallback.
int64_t fastExtractTsNs(const std::string& line) {
    static const std::string kKey = "\"ts_ns\":";
    const auto pos = line.find(kKey);
    if (pos == std::string::npos) return 0;
    const auto start = pos + kKey.size();
    // Find the end of the number (next non-digit, allowing leading minus).
    std::size_t i = start;
    if (i < line.size() && line[i] == '-') ++i;
    while (i < line.size() && std::isdigit(static_cast<unsigned char>(line[i]))) {
        ++i;
    }
    if (i == start) return 0;
    try {
        return std::stoll(line.substr(start, i - start));
    } catch (...) {
        return 0;
    }
}

// ─────────────────────────────────────────────────────────────────────
// HTTP client + POST helpers
// ─────────────────────────────────────────────────────────────────────

struct UrlParts {
    std::string scheme;     // "http" or "https"
    std::string host;
    int         port = -1;  // -1 = use scheme default
};

bool parseUrl(const std::string& url, UrlParts& out) {
    std::string u = url;
    while (!u.empty() && u.back() == '/') u.pop_back();
    const auto sep = u.find("://");
    if (sep == std::string::npos) {
        out.scheme = "http";
    } else {
        out.scheme = u.substr(0, sep);
        u = u.substr(sep + 3);
    }
    // Strip path component if any (we use only the host part).
    const auto pathstart = u.find('/');
    if (pathstart != std::string::npos) u = u.substr(0, pathstart);
    const auto colon = u.find(':');
    if (colon == std::string::npos) {
        out.host = u;
    } else {
        out.host = u.substr(0, colon);
        try { out.port = std::stoi(u.substr(colon + 1)); }
        catch (...) { return false; }
    }
    return !out.host.empty() && (out.scheme == "http" || out.scheme == "https");
}

std::unique_ptr<httplib::Client> makeClient(const UrlParts& url,
                                            const UploadOptions& opts) {
    std::string scheme_host_port = url.scheme + "://" + url.host;
    if (url.port > 0) scheme_host_port += ":" + std::to_string(url.port);
    try {
        auto cli = std::make_unique<httplib::Client>(scheme_host_port);
        cli->set_connection_timeout(0, opts.connect_timeout_ms * 1000);
        cli->set_read_timeout(0, opts.read_timeout_ms * 1000);
        cli->set_keep_alive(true);
        return cli;
    } catch (const std::exception& e) {
        GFL_LOG_ERROR("[uploadLogs] httplib client construction failed for '",
                      scheme_host_port, "': ", e.what());
        return nullptr;
    }
}

/// Wrap a raw NDJSON event line in the `EventWrapper` envelope the
/// backend expects on every POST to /api/v1/events/{type}:
///
///   {
///     "data": "<NDJSON line, JSON-string-encoded>",
///     "agentSendingTime": <ms-since-epoch>,
///     "hostname": "<host>",
///     "ipAddr":   "<ip-or-empty>"
///   }
///
/// CRITICAL: `EventIngestionController` deserializes the request body
/// as `EventWrapper`, then `BatchIngestionServiceImpl` re-deserializes
/// `wrapper.data()` into the per-event-type class. If we POST the bare
/// event JSON instead, Spring leaves every field of EventWrapper null,
/// the inner readValue(null, ...) throws, the exception is swallowed
/// in a catch block, and the controller returns 200 anyway — silent
/// data loss. The wrapper MUST be present.
std::string wrapEventBody(const std::string& ndjson_line,
                          const std::string& hostname,
                          const std::string& ip_addr) {
    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    std::ostringstream oss;
    oss << "{\"data\":\""        << json::escape(ndjson_line)
        << "\",\"agentSendingTime\":" << now_ms
        << ",\"hostname\":\""    << json::escape(hostname)
        << "\",\"ipAddr\":\""    << json::escape(ip_addr)
        << "\"}";
    return oss.str();
}

/// Outcome of a single POST attempt. Drives the retry / abort logic.
enum class PostOutcome {
    Ok,
    TransientFailure,   // network error or 5xx — retry candidate
    AuthFailure,        // 401 / 403 — abort the whole upload
    ClientError,        // other 4xx — log and skip the event
};

PostOutcome doPost(httplib::Client& client, const std::string& path,
                   const httplib::Headers& headers, const std::string& body,
                   std::string& failure_reason) {
    auto res = client.Post(path.c_str(), headers, body, "application/json");
    if (!res) {
        std::ostringstream os;
        os << "transport error httplib::Error=" << static_cast<int>(res.error());
        failure_reason = os.str();
        return PostOutcome::TransientFailure;
    }
    const int status = res->status;
    if (status >= 200 && status < 300) return PostOutcome::Ok;
    if (status == 401 || status == 403) {
        failure_reason = "auth failure (HTTP " + std::to_string(status) + ")";
        return PostOutcome::AuthFailure;
    }
    if (status >= 500) {
        failure_reason = "server error (HTTP " + std::to_string(status) + ")";
        return PostOutcome::TransientFailure;
    }
    failure_reason = "client error (HTTP " + std::to_string(status) + ")";
    return PostOutcome::ClientError;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────
// Main entry point
// ─────────────────────────────────────────────────────────────────────

// Lightweight info about one session we found in the logs. Built by
// the discovery pass before any HTTP work happens.
struct SessionInfo {
    std::string session_id;
    int64_t     job_start_ts_ns = 0;  // 0 if no ts_ns parsed (sorts oldest)
};

/// Single-pass scan over every discovered file looking for `job_start`
/// events. Records {session_id, ts_ns} for each unique session. Stops
/// reading each file at the first `job_start` it finds — there's only
/// ever one per session, and they tend to live at the head of the
/// oldest file in their channel, so this is cheap in practice.
std::vector<SessionInfo> discoverSessions(const std::vector<DiscoveredFile>& files) {
    std::map<std::string, int64_t> seen;  // sid → earliest ts_ns observed
    for (const auto& f : files) {
        NdjsonReader reader(f.path, f.compressed);
        if (!reader.ok()) continue;
        std::string line;
        while (reader.readLine(line)) {
            if (line.empty()) continue;
            const std::string type = fastExtractType(line);
            if (type != "job_start") continue;
            const std::string sid = fastExtractSessionId(line);
            if (sid.empty()) continue;
            const int64_t ts = fastExtractTsNs(line);
            auto it = seen.find(sid);
            if (it == seen.end()) {
                seen[sid] = ts;
            } else if (ts < it->second) {
                it->second = ts;  // pick earliest (defensive — there's usually only one)
            }
        }
    }
    std::vector<SessionInfo> out;
    out.reserve(seen.size());
    for (const auto& [sid, ts] : seen) out.push_back({sid, ts});
    // Oldest first (ascending ts_ns) → "latest" = back(), and all_sessions
    // iterates in natural chronological order.
    std::sort(out.begin(), out.end(),
              [](const SessionInfo& a, const SessionInfo& b) {
                  if (a.job_start_ts_ns != b.job_start_ts_ns) {
                      return a.job_start_ts_ns < b.job_start_ts_ns;
                  }
                  return a.session_id < b.session_id;  // tie-break for determinism
              });
    return out;
}

UploadResult uploadLogs(const UploadOptions& opts) {
    UploadResult result;
    const auto upload_start = std::chrono::steady_clock::now();
    auto elapsedMs = [&]() -> long long {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - upload_start).count();
    };

    // ── Input validation ─────────────────────────────────────────────
    if (opts.log_path.empty()) {
        result.warnings.emplace_back("uploadLogs: log_path is empty");
        return result;
    }
    if (opts.backend_url.empty()) {
        result.warnings.emplace_back("uploadLogs: backend_url is empty");
        return result;
    }
    if (opts.api_key.empty()) {
        result.warnings.emplace_back("uploadLogs: api_key is empty");
        return result;
    }
    if (!opts.session_id_filter.empty() && opts.all_sessions) {
        result.warnings.emplace_back(
            "uploadLogs: session_id_filter and all_sessions are mutually "
            "exclusive — pass only one.");
        return result;
    }

    const PathParts parts = splitLogPath(opts.log_path);
    std::error_code ec;
    if (!fs::exists(parts.directory, ec) || !fs::is_directory(parts.directory, ec)) {
        result.warnings.emplace_back(
            "uploadLogs: log directory does not exist: " + parts.directory.string());
        return result;
    }

    // ── Discover files + sessions ────────────────────────────────────
    auto files = discoverFiles(parts);
    if (files.empty()) {
        GFL_LOG_DEBUG("[uploadLogs] no log files matched prefix '",
                      parts.prefix, "' in ", parts.directory.string());
        result.success = true;
        result.elapsed_ms = elapsedMs();
        return result;
    }

    const auto all_sessions = discoverSessions(files);
    if (all_sessions.empty()) {
        result.warnings.emplace_back(
            "uploadLogs: no job_start events found in " + parts.directory.string() +
            " — the directory has files matching the prefix but none carry "
            "a session header. Was the session aborted before init?");
        result.success = true;  // nothing to upload → not a failure, just a no-op
        result.elapsed_ms = elapsedMs();
        return result;
    }

    // ── Pick target session(s) ───────────────────────────────────────
    Cursor cursor = readCursor(parts.directory, opts.cursor_filename);

    std::vector<SessionInfo> targets;
    if (!opts.session_id_filter.empty()) {
        for (const auto& s : all_sessions) {
            if (s.session_id == opts.session_id_filter) {
                targets.push_back(s);
                break;
            }
        }
        if (targets.empty()) {
            result.warnings.emplace_back(
                "session_id '" + opts.session_id_filter +
                "' not found in any job_start event under " +
                parts.directory.string() + ". Sessions present: " +
                std::to_string(all_sessions.size()) +
                " (run with --all-sessions to upload all of them).");
            result.elapsed_ms = elapsedMs();
            return result;
        }
    } else if (opts.all_sessions) {
        targets = all_sessions;
    } else {
        // Default: latest session (highest job_start.ts_ns). The sort
        // in discoverSessions puts oldest first, so back() is newest.
        targets.push_back(all_sessions.back());
    }

    // ── Cursor pre-flight (refuse / skip already-completed) ─────────
    if (!opts.force) {
        if (opts.all_sessions) {
            // Batch mode: silently filter out completed sessions.
            std::vector<SessionInfo> remaining;
            for (const auto& s : targets) {
                if (cursor.completed_sessions.count(s.session_id) == 0) {
                    remaining.push_back(s);
                } else {
                    result.files_skipped_by_cursor++;  // reusing field for "sessions skipped"
                    GFL_LOG_DEBUG("[uploadLogs] skipping completed session ",
                                  s.session_id, " (completed at ",
                                  cursor.completed_sessions.at(s.session_id).completed_at_iso8601,
                                  ")");
                }
            }
            targets = std::move(remaining);
            if (targets.empty()) {
                // Every session in the dir is already in the cursor.
                // No work to do — that's a success, not a failure.
                result.success = true;
                result.elapsed_ms = elapsedMs();
                GFL_LOG_DEBUG("[uploadLogs] all sessions already in cursor — no-op.");
                return result;
            }
        } else {
            // Single-session mode (default or session_id_filter): refuse
            // if the target is already completed.
            const auto& tsid = targets.front().session_id;
            auto it = cursor.completed_sessions.find(tsid);
            if (it != cursor.completed_sessions.end()) {
                result.warnings.emplace_back(
                    "Session '" + tsid + "' was already uploaded on " +
                    it->second.completed_at_iso8601 + " (" +
                    std::to_string(it->second.events) +
                    " events). Pass force=true (CLI: --force) to re-upload.");
                result.elapsed_ms = elapsedMs();
                return result;  // success stays false
            }
        }
    }

    // ── HTTP client setup ────────────────────────────────────────────
    UrlParts url;
    if (!parseUrl(opts.backend_url, url)) {
        result.warnings.emplace_back(
            "uploadLogs: malformed backend_url: " + opts.backend_url);
        return result;
    }
    auto client = makeClient(url, opts);
    if (!client) {
        result.warnings.emplace_back(
            "uploadLogs: failed to construct HTTP client for " + opts.backend_url);
        return result;
    }
    const std::string api_path = normalizeApiPath(opts.api_path);
    const std::string ua_header =
        std::string("gpufl-client/") + kClientVersion + " (deferred-upload)";
    const httplib::Headers headers = {
        {"Authorization",              "Bearer " + opts.api_key},
        {"User-Agent",                 ua_header},
        {"X-GpuFlight-Client-Version", kClientVersion},
        {"X-GpuFlight-Wire-Version",   kWireVersion},
    };

    // Stamped on every POST envelope so the backend can attribute
    // events to the originating host. Resolved once per upload — both
    // helpers cache after first call (and IP currently returns empty).
    const std::string envelope_hostname = getLocalHostname();
    const std::string envelope_ip       = getLocalIpAddr();

    // Progress reporting state.
    auto last_progress_time = upload_start;
    std::size_t bytes_since_last_progress = 0;
    auto maybeLogProgress = [&](bool force) {
        if (!opts.report_progress) return;
        const auto now = std::chrono::steady_clock::now();
        const auto elapsed_since = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_progress_time).count();
        const bool by_time = elapsed_since >= opts.progress_log_interval_ms;
        const bool by_bytes = bytes_since_last_progress >= opts.progress_log_interval_bytes;
        if (!force && !by_time && !by_bytes) return;
        const auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - upload_start).count();
        std::fprintf(stderr,
                     "[gpufl::upload] %zu events uploaded (%zu MB), "
                     "%zu/%zu session(s), %llds elapsed\n",
                     result.events_uploaded,
                     result.bytes_uploaded / (1024 * 1024),
                     result.files_processed,
                     targets.size(),
                     static_cast<long long>(total_elapsed));
        last_progress_time = now;
        bytes_since_last_progress = 0;
    };

    auto budgetExpired = [&]() {
        return elapsedMs() >= opts.total_timeout_ms;
    };

    bool auth_failed = false;
    bool budget_aborted = false;

    // Per-event POST with retry. Returns true if the event was either
    // successfully POSTed OR skipped with a recorded warning (caller
    // should continue). Returns false only when the whole upload must
    // abort (auth failure or budget exhausted).
    //
    // `ndjson_line` is the raw event JSON we read from disk. We wrap it
    // in the EventWrapper envelope here (see wrapEventBody for why) and
    // POST that. bytes_uploaded counts the wrapped envelope size — that
    // matches what actually went over the wire.
    auto postEvent = [&](const std::string& type,
                         const std::string& ndjson_line) -> bool {
        if (budgetExpired()) {
            budget_aborted = true;
            return false;
        }
        const std::string body = wrapEventBody(ndjson_line, envelope_hostname,
                                               envelope_ip);
        const std::string path = api_path + "/events/" + type;
        std::string fail_reason;
        for (int attempt = 0; attempt <= opts.max_retries; ++attempt) {
            const PostOutcome outcome = doPost(*client, path, headers, body, fail_reason);
            if (outcome == PostOutcome::Ok) {
                result.events_uploaded++;
                result.bytes_uploaded   += body.size();
                bytes_since_last_progress += body.size();
                return true;
            }
            if (outcome == PostOutcome::AuthFailure) {
                result.warnings.push_back(
                    "POST /events/" + type + " failed: " + fail_reason +
                    " — aborting remaining uploads");
                auth_failed = true;
                return false;
            }
            if (outcome == PostOutcome::ClientError) {
                result.warnings.push_back(
                    "POST /events/" + type + " failed: " + fail_reason +
                    " — skipping event");
                return true;
            }
            if (attempt < opts.max_retries && !budgetExpired()) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(opts.retry_delay_ms));
                continue;
            }
            result.warnings.push_back(
                "POST /events/" + type + " failed after " +
                std::to_string(attempt + 1) + " attempt(s): " + fail_reason);
            return true;
        }
        return true;
    };

    GFL_LOG_DEBUG("[uploadLogs] uploading ", targets.size(),
                  " session(s) from ", files.size(), " file(s) to ",
                  opts.backend_url);

    // Track which underlying files we've at-least-opened so the
    // `files_processed` count matches its name (unique files visited,
    // not files × sessions).
    std::unordered_set<std::string> files_visited;

    // ── Per-session upload loop ──────────────────────────────────────
    //
    // For each target session: stream every discovered file, filter
    // events by session_id, defer that session's shutdown events to
    // the end of THAT session (not the end of the whole upload).
    // Result: backend sees a clean job_start → batches → shutdown
    // arrival order per session.
    for (const auto& target : targets) {
        if (auth_failed || budget_aborted) break;
        const std::string& current_sid = target.session_id;

        std::vector<std::pair<std::string, std::string>> deferred_shutdowns;
        std::size_t events_from_session = 0;
        bool session_ok = true;

        GFL_LOG_DEBUG("[uploadLogs] session ", current_sid, " starting");

        for (auto& f : files) {
            if (auth_failed || budget_aborted) {
                session_ok = false;
                break;
            }
            const std::string basename = f.path.filename().string();

            NdjsonReader reader(f.path, f.compressed);
            if (!reader.ok()) {
                result.warnings.push_back("Could not open " + basename + " — skipping");
                continue;
            }
            files_visited.insert(basename);

            std::string line;
            while (reader.readLine(line)) {
                if (line.empty()) continue;

                // Cheap session filter. Three cases:
                //   1. session_id field missing entirely → the line is
                //      malformed (every gpufl event carries session_id).
                //      Warn + skip; don't silently lose visibility into
                //      bad data.
                //   2. session_id present but doesn't match this target →
                //      another session's event. Silently skip — that's
                //      the whole point of the filter.
                //   3. session_id matches → process below.
                const std::string line_sid = fastExtractSessionId(line);
                if (line_sid.empty()) {
                    result.warnings.push_back(
                        "Unparseable NDJSON line in " + basename +
                        " (no session_id field) — skipping");
                    continue;
                }
                if (line_sid != current_sid) continue;

                const std::string type = extractType(line);
                if (type.empty()) {
                    result.warnings.push_back(
                        "Unparseable NDJSON line in " + basename +
                        " (no type field) — skipping");
                    continue;
                }
                if (type == "shutdown") {
                    deferred_shutdowns.emplace_back(type, line);
                    continue;
                }
                if (!postEvent(type, line)) {
                    session_ok = false;
                    break;
                }
                events_from_session++;
                maybeLogProgress(/*force=*/false);
            }
            if (!session_ok) break;
        }

        // POST this session's deferred shutdowns BEFORE moving on to
        // the next session so lifecycle ordering stays per-session.
        if (session_ok) {
            for (auto& [type, body] : deferred_shutdowns) {
                if (!postEvent(type, body)) {
                    session_ok = false;
                    break;
                }
            }
        }

        // Mark this session as completed in the cursor if it shipped
        // cleanly. We persist after each session — a mid-run crash
        // still lets re-runs skip the sessions that did complete.
        if (session_ok && events_from_session > 0) {
            CompletedSession cs;
            cs.completed_at_iso8601 = nowIso8601Utc();
            cs.events = events_from_session;
            cursor.completed_sessions[current_sid] = cs;
            if (!writeCursor(parts.directory, opts.cursor_filename, cursor)) {
                result.warnings.emplace_back(
                    "Could not write cursor file " + opts.cursor_filename);
            }
            GFL_LOG_DEBUG("[uploadLogs] session ", current_sid,
                          " complete (", events_from_session, " events)");
        }
    }

    result.files_processed = files_visited.size();

    // ── Wrap up ──────────────────────────────────────────────────────
    if (budget_aborted) {
        result.warnings.emplace_back(
            "Total timeout (" + std::to_string(opts.total_timeout_ms) +
            " ms) exceeded — upload aborted");
    }
    // Success = no auth or budget failure AND every target session
    // either shipped events or was a deliberately-empty target.
    result.success = !auth_failed && !budget_aborted;
    result.elapsed_ms = elapsedMs();
    maybeLogProgress(/*force=*/true);

    GFL_LOG_DEBUG("[uploadLogs] complete: success=",
                  (result.success ? "true" : "false"),
                  " events=", result.events_uploaded,
                  " bytes=", result.bytes_uploaded,
                  " sessions=", targets.size(),
                  " warnings=", result.warnings.size(),
                  " elapsed_ms=", result.elapsed_ms);

    return result;
}

}  // namespace gpufl

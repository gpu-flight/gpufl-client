// High-level structure (U1: file-unit upload — one POST per rotated file):
//   1. resolve log_path as a v1.2 session-output root, with legacy prefix fallback
//   2. discover .log / .log.gz files; sort by
//      (session_id, channel, rotation_index DESC, active-file last)
//   3. read cursor file → uploaded files + completed sessions
//   4. for each target session, for each of its files in order:
//        - rotated file already in cursor: skip (crash-resume)
//        - POST the file body in ONE request: .log.gz bytes go on the
//          wire AS-IS (Content-Encoding: gzip, zero re-encode), plain
//          .log is gzipped first
//        - rotated file shipped → append to cursor
//      The old 5000-line/5 MB client-side chunking is gone: the rotator
//      already caps files (Logger::kDefaultRotateBytes = 64 MiB raw,
//      ≈ 4–8 MB gz), the backend accept path just streams the body to
//      spool/lake storage, and the worker ingest is line-streamed — so
//      the file IS the natural unit, ~13× fewer round trips / GCS
//      objects / ingestion jobs per file.
//   5. lifecycle ordering: job_start lives in the oldest file (POSTed
//      first) and shutdown in the active file (POSTed last), so arrival
//      order is preserved without the old extract-and-defer pass. Note
//      arrival ≠ ingest-completion order under the async queue — the
//      ordering chain (upload-plan U3) is the real guarantee.
//   6. enforce total_timeout_ms across the whole loop — abort + return
//      success=false on overrun rather than blocking the host process

#include "gpufl/upload/upload_logs.hpp"

#include <algorithm>
#include <array>
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
#include "gpufl/core/logger/logger.hpp"
#include "gpufl/core/version.hpp"

namespace gpufl {
namespace fs = std::filesystem;
namespace {

// ─────────────────────────────────────────────────────────────────────
// File discovery + classification
// ─────────────────────────────────────────────────────────────────────

struct DiscoveredFile {
    fs::path     path;
    std::string  session_id;      // v1.2+: derived from parent subdir name
    std::string  channel;         // e.g. "device", "scope", "system"
    int          rotation_index;  // 0 = active .log file; ≥1 = rotated
    bool         compressed;      // true if .log.gz
};

/// v1.2 disk layout — `log_path` is the directory under which each
/// session lives in its own subdirectory:
///
///   <log_path>/
///     <session_id_A>/
///       device.log
///       device.1.log.gz
///       scope.log
///       system.log
///     <session_id_B>/
///       ...
///
/// `splitLogPath` strips a legacy trailing `.log` so a caller passing
/// the pre-v1.2 "/dir/app" or "/dir/app.log" form still gets a
/// reasonable directory to look in.
struct PathParts {
    fs::path directory;
};

PathParts splitLogPath(const std::string& log_path) {
    fs::path p(log_path);
    if (p.extension() == ".log") p.replace_extension();
    PathParts out;
    out.directory = p;
    return out;
}

/// Parse a filename like "device.log" or "device.5.log.gz" — the
/// per-channel files inside a session subdirectory. Returns false if
/// the name doesn't match.
///
/// Note: this is the v1.2 form (no prefix in the filename — the
/// session_id is the parent directory name). The pre-v1.2 form
/// `<prefix>.<channel>.log` is handled separately by the legacy-
/// detection path.
bool parseLogFilename(const std::string& filename,
                      std::string& channel_out,
                      int& rotation_index_out,
                      bool& compressed_out) {
    std::string rest = filename;

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

    // Now rest is either "{channel}" (active) or "{channel}.{index}" (rotated)
    const auto last_dot = rest.find_last_of('.');
    if (last_dot == std::string::npos) {
        channel_out = rest;
        rotation_index_out = 0;
        return !channel_out.empty();
    }
    const std::string tail = rest.substr(last_dot + 1);
    try {
        const int idx = std::stoi(tail);
        if (idx <= 0) return false;
        rotation_index_out = idx;
        channel_out = rest.substr(0, last_dot);
        return !channel_out.empty();
    } catch (...) {
        channel_out = rest;
        rotation_index_out = 0;
        return !channel_out.empty();
    }
}

/// Detect pre-v1.2 flat layout — files like `<anything>.<device|scope|
/// system>.log[.gz]` directly inside `dir`, instead of inside a
/// session subdirectory. The check is intentionally narrow (only the
/// three known channel names) to avoid false positives on user files.
///
/// Returns true on detection; caller bails with a migration-hint
/// warning rather than attempting to upload (the on-disk shape no
/// longer matches what the uploader's discovery walks).
bool detectLegacyLayoutAt(const fs::path& dir) {
    std::error_code ec;
    if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec)) return false;
    static const std::regex kLegacyRe(
        R"(.+\.(device|scope|system)\.(?:\d+\.)?log(?:\.gz)?)");
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (!entry.is_regular_file(ec)) continue;
        if (std::regex_match(entry.path().filename().string(), kLegacyRe)) {
            return true;
        }
    }
    return false;
}

/// Outcome of `repairOrphanLogIfNeeded`. Three states:
///   - Keep (path): use the returned path as the file to read. Either
///       the original .log (repair was unnecessary or failed) or the
///       newly-created .log.gz (repair succeeded).
///   - Skip:        do NOT add this entry to discovery. Happens when a
///       .log AND its .log.gz counterpart both exist — we've removed
///       the stale .log here; the .gz will be discovered separately by
///       the directory iterator and added then. Adding the .log would
///       duplicate the .gz's events on upload.
struct RepairResult {
    enum Kind { Keep, Skip };
    Kind     kind = Keep;
    fs::path path;
};

/// Lazy crash repair — gzip an orphan `.log` file in place if the
/// session's clean-shutdown compress-on-close never ran.
///
/// Cases:
///   1. `.log` exists, `.log.gz` does NOT  → repair (gzip + remove .log)
///       → returns Keep(<.gz path>).
///   2. `.log` exists, `.log.gz` ALSO exists → the .log is a stale
///       leftover from a failed compress-on-shutdown (Windows file-lock
///       edge case, etc.). Remove the .log; signal Skip so the caller
///       doesn't add a duplicate entry for the .gz (the directory
///       iterator will hit the .gz separately).
///   3. `.log` exists but is empty → remove (no data to preserve),
///       signal Skip.
///   4. `.log` doesn't exist → returns Keep(<.log path>) unchanged.
RepairResult repairOrphanLogIfNeeded(const fs::path& log_path) {
    std::error_code ec;
    if (!fs::exists(log_path, ec)) return {RepairResult::Keep, log_path};
    const fs::path gz_path = fs::path(log_path.string() + ".gz");
    if (fs::exists(gz_path, ec)) {
        // Both files exist — .log is the stale duplicate from a
        // failed compress-on-shutdown. Remove it and skip; the .gz
        // entry will be added by the iterator's other pass.
        std::error_code rm_ec;
        fs::remove(log_path, rm_ec);
        return {RepairResult::Skip, log_path};
    }

    // Empty file → just remove. No data to preserve.
    if (fs::file_size(log_path, ec) == 0) {
        fs::remove(log_path, ec);
        return {RepairResult::Skip, log_path};
    }

    // Inline gzip using zlib. Read source, compress, write dest, then
    // remove source on success.
    std::ifstream in(log_path, std::ios::binary);
    if (!in) return {RepairResult::Keep, log_path};
    gzFile out = gzopen(gz_path.string().c_str(), "wb");
    if (!out) return {RepairResult::Keep, log_path};
    char buf[64 * 1024];
    bool ok = true;
    while (in) {
        in.read(buf, sizeof(buf));
        const auto n = in.gcount();
        if (n > 0) {
            if (gzwrite(out, buf, static_cast<unsigned>(n)) != static_cast<int>(n)) {
                ok = false;
                break;
            }
        }
    }
    gzclose(out);
    in.close();
    if (!ok) {
        fs::remove(gz_path, ec);
        return {RepairResult::Keep, log_path};
    }
    fs::remove(log_path, ec);
    return {RepairResult::Keep, gz_path};
}

std::vector<DiscoveredFile> discoverFiles(const PathParts& parts) {
    std::vector<DiscoveredFile> out;
    std::error_code ec;
    if (!fs::exists(parts.directory, ec) || !fs::is_directory(parts.directory, ec)) {
        return out;
    }
    // v1.2: walk one level of subdirectories — each is a session_id.
    // Inside each subdir, parse channel files. Lazy crash repair runs
    // here so any orphan `.log` files get gzipped on first discovery.
    for (const auto& session_entry : fs::directory_iterator(parts.directory, ec)) {
        if (!session_entry.is_directory(ec)) continue;
        const std::string sid = session_entry.path().filename().string();
        if (sid.empty() || sid.front() == '.') continue;  // skip dotfiles like .gpufl-upload-cursor.json

        for (const auto& entry : fs::directory_iterator(session_entry.path(), ec)) {
            if (!entry.is_regular_file(ec)) continue;
            const std::string fname = entry.path().filename().string();
            DiscoveredFile df;
            if (!parseLogFilename(fname, df.channel, df.rotation_index, df.compressed)) {
                continue;
            }
            // Lazy repair / dedup for `.log` entries:
            //   - Orphan .log (no .log.gz): gzip in place, treat as .gz.
            //   - .log + .log.gz both present: the .log is a stale
            //     leftover from a failed compress-on-shutdown. Drop the
            //     .log; the iterator will hit the .gz separately and
            //     add the (single) correct entry then. This is the fix
            //     for the "duplicate records after uploadLogs" symptom
            //     that surfaces when compress() succeeded but couldn't
            //     remove the source file (e.g., Windows file-lock edge).
            //   - Empty .log: just removed; skip.
            fs::path final_path = entry.path();
            if (!df.compressed) {
                const auto r = repairOrphanLogIfNeeded(entry.path());
                if (r.kind == RepairResult::Skip) continue;
                final_path = r.path;
                if (final_path.extension() == ".gz") {
                    df.compressed = true;
                }
            }
            df.path = final_path;
            df.session_id = sid;
            out.push_back(std::move(df));
        }
    }

    // Sort: by (session_id, channel) lexicographic, then by upload
    // order within a channel — oldest first. Rotation index N=max_files
    // is the oldest; active (N=0) is newest. So within (sid, channel):
    // descending rotation_index, with active (0) last.
    std::sort(out.begin(), out.end(),
              [](const DiscoveredFile& a, const DiscoveredFile& b) {
                  if (a.session_id != b.session_id) return a.session_id < b.session_id;
                  if (a.channel != b.channel) return a.channel < b.channel;
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
//     "uploaded_files":      ["<session_id>/device.1.log.gz", ...],
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

// ─────────────────────────────────────────────────────────────────────
// Chunked NDJSON upload (POST /api/v1/events/stream)
// ─────────────────────────────────────────────────────────────────────
//
// Replaces the pre-v1.2 per-event POST loop. Each chunk is a bundle of
// NDJSON lines (one event per line, all events the same session_id),
// gzipped and POSTed in a single HTTP request. The wire format is
// described in EventIngestionController#receiveEventStream on the
// backend; in short:
//
//   POST /api/v1/events/stream
//   Authorization: Bearer <api_key>
//   X-GpuFlight-Session-Id: <uuid>          (required; backend validates
//                                            every line's session_id matches)
//   X-GpuFlight-Hostname: <host>            (optional, stamped on every event)
//   Content-Type: application/x-ndjson
//   Content-Encoding: gzip                  (we always gzip)
//
//   <gzipped NDJSON body, one event per line>
//
// Response shape on 2xx (parsed by parseStreamResponse below):
//
//   {"accepted": N, "rejected": M, "errors": [{"line": L, "type": "T",
//                                              "reason": "..."}]}

/// Gzip-compress `input` and return the compressed bytes. Uses zlib
/// directly (deflate with windowBits +16 = gzip wrapper). Throws
/// nothing — on failure, returns an empty string and the caller
/// treats that as a transient failure for the chunk (retry).
///
/// Target compression level 6 is the same default cpp-httplib uses
/// internally — a reasonable balance of CPU vs ratio for log NDJSON,
/// which compresses to roughly 10× smaller on real workloads.
std::string gzipString(const std::string& input) {
    if (input.empty()) return {};
    z_stream zs{};
    // 15 = max window, +16 = gzip wrapper. memLevel 8 is the zlib
    // default. Level 6 matches httplib::detail::compress's default.
    if (deflateInit2(&zs, /*level=*/6, Z_DEFLATED,
                     /*windowBits=*/15 + 16, /*memLevel=*/8,
                     Z_DEFAULT_STRATEGY) != Z_OK) {
        return {};
    }
    zs.next_in  = reinterpret_cast<Bytef*>(const_cast<char*>(input.data()));
    zs.avail_in = static_cast<uInt>(input.size());

    // Worst-case compressed size ≈ source + 64 KB; allocate generously.
    std::string out;
    out.resize(deflateBound(&zs, static_cast<uLong>(input.size())));
    zs.next_out  = reinterpret_cast<Bytef*>(out.data());
    zs.avail_out = static_cast<uInt>(out.size());

    const int rc = deflate(&zs, Z_FINISH);
    const uLong produced = zs.total_out;
    deflateEnd(&zs);
    if (rc != Z_STREAM_END) return {};
    out.resize(produced);
    return out;
}

/// Slurp a file's raw bytes. Empty string on open/read failure (caller
/// warns + skips the file). Rotated logs are ≤ ~10 MB gzipped, so whole-
/// file buffering is fine; this is the same memory order the old chunk
/// assembly used.
std::string readFileBytes(const fs::path& p) {
    const std::ifstream in(p, std::ios::binary);
    if (!in) return {};
    std::ostringstream oss;
    oss << in.rdbuf();
    if (in.bad()) return {};
    return oss.str();
}

/// Decompressed size of a gzip stream from its ISIZE trailer (last 4
/// bytes, little-endian, mod 2^32). Exact for our single-member files
/// (the rotator caps them at Logger::kDefaultRotateBytes = 64 MiB, far
/// under 4 GB). 0 = unknown/empty.
std::size_t gzipDecompressedSizeHint(const std::string& gz) {
    if (gz.size() < 18) return 0;   // 10-byte header + 8-byte trailer
    const auto* p = reinterpret_cast<const unsigned char*>(gz.data()) + gz.size() - 4;
    return static_cast<std::size_t>(p[0]) |
           (static_cast<std::size_t>(p[1]) << 8) |
           (static_cast<std::size_t>(p[2]) << 16) |
           (static_cast<std::size_t>(p[3]) << 24);
}

/// Outcome of a single stream-chunk POST. Drives the retry / abort
/// logic in the main loop. Mirrors the per-event PostOutcome that came
/// before but adds OldBackend404 — a sentinel for "the /stream endpoint
/// is missing, the backend predates v1.2." We don't retry on that; we
/// abort the whole upload with a migration-hint warning.
enum class StreamPostOutcome {
    Ok,
    OldBackend404,
    TransientFailure,
    AuthFailure,
    ClientError,
};

/// One per-line error parsed out of a 200-OK chunk response. Lets us
/// surface partial failures as `UploadResult.warnings` entries without
/// failing the whole upload.
struct StreamLineError {
    int         line = 0;
    std::string type;
    std::string reason;
};

/// Outcome of a chunk POST plus the parsed response counters.
///
/// <p>Two wire shapes are recognised, distinguished by `async_accepted`:
///
/// <ul>
///   <li><b>Legacy synchronous (pre-Phase 3a backends).</b> HTTP 200
///       with `{"accepted": N, "rejected": M, "errors": [...]}`. The
///       per-line dispatch ran on the request thread; `accepted +
///       rejected == chunk_lines` invariant holds; `line_errors`
///       carries the explanations.</li>
///   <li><b>Async-accept (Phase 3a+ backends).</b> HTTP 202 with
///       `{"accepted_for_processing": true, "spool_id": "..."}`. The
///       body was streamed to a spool (local disk or GCS) and a
///       background worker drains it asynchronously. There are no
///       per-line counts to return — the caller trusts the local
///       `chunk_lines` as the accepted count.</li>
/// </ul>
struct StreamPostResult {
    StreamPostOutcome           outcome = StreamPostOutcome::TransientFailure;
    std::string                 failure_reason;  // populated when outcome != Ok

    // ── Legacy synchronous-accept fields ──
    std::size_t                 accepted = 0;
    std::size_t                 rejected = 0;
    std::vector<StreamLineError> line_errors;

    // ── Async-accept fields (Phase 3a+) ──
    bool                        async_accepted = false;
    std::string                 spool_id;
};

/// Parse the backend's stream-endpoint response. Two wire shapes are
/// recognised — see {@link StreamPostResult} for the full description.
/// Tolerant: missing or unrecognised fields leave the corresponding
/// `out` field at its default; the caller falls back to "all-sent-
/// assumed-accepted" via the chunk_lines it tracked locally.
void parseStreamResponse(const std::string& body, StreamPostResult& out) {
    try {
        const json::JsonValue v = json::parseJson(body);
        if (!v.is_object()) return;

        // ── Phase 3a+ async-accept shape ─────────────────────────────
        // The backend streamed our body into a spool (local disk or
        // GCS) and queued it for the SpoolWorker to drain. There's no
        // per-line accept/reject count to return — completion happens
        // out-of-band. We record the spool id so the caller can stash
        // it for operator debugging, and flag async_accepted so the
        // upload loop knows to use chunk_lines as the accepted count
        // instead of looking for an `accepted` field.
        if (v.contains("accepted_for_processing")) {
            out.async_accepted = v.at("accepted_for_processing").get_bool(false);
            if (v.contains("spool_id") && v.at("spool_id").is_string()) {
                out.spool_id = v.at("spool_id").get_string();
            }
            return;
        }

        // ── Legacy synchronous-accept shape (pre-Phase 3a) ──────────
        // Per-line dispatch ran on the request thread. accepted +
        // rejected == chunk_lines; line_errors carries the per-line
        // explanations bounded server-side at MAX_REPORTED_ERRORS.
        if (v.contains("accepted")) {
            out.accepted = static_cast<std::size_t>(v.at("accepted").as_int(0));
        }
        if (v.contains("rejected")) {
            out.rejected = static_cast<std::size_t>(v.at("rejected").as_int(0));
        }
        if (v.contains("errors")) {
            const json::JsonValue& errs = v.at("errors");
            if (errs.is_array()) {
                for (std::size_t i = 0; i < errs.size(); ++i) {
                    const json::JsonValue& e = errs[i];
                    if (!e.is_object()) continue;
                    StreamLineError le;
                    if (e.contains("line"))   le.line   = static_cast<int>(e.at("line").as_int(0));
                    if (e.contains("type")   && e.at("type").is_string())
                        le.type   = e.at("type").get_string();
                    if (e.contains("reason") && e.at("reason").is_string())
                        le.reason = e.at("reason").get_string();
                    out.line_errors.push_back(std::move(le));
                }
            }
        }
    } catch (...) {
        // Body wasn't JSON or was unparseable — leave counters at 0,
        // caller falls back to "all-sent-assumed-accepted" via the
        // line_count it tracked locally. Better than failing the chunk
        // for a malformed response from an otherwise-2xx backend.
    }
}

/// POST one NDJSON chunk to `/api/v1/events/stream`. The body has
/// already been gzipped by the caller — we set Content-Encoding here
/// to match.
///
/// Returns the parsed outcome. Caller is responsible for retry /
/// abort decisions based on `outcome`.
StreamPostResult postStreamChunk(httplib::Client&         client,
                                 const std::string&       api_path,
                                 const std::string&       session_id,
                                 const std::string&       hostname,
                                 const std::string&       api_key,
                                 const std::string&       ua_header,
                                 const std::string&       gzipped_body) {
    StreamPostResult out;
    const std::string path = api_path + "/events/stream";

    // Per-chunk headers. Authorization etc. are repeated here (rather
    // than relying on client.set_default_headers) so a future move to
    // multiple concurrent clients doesn't silently drop auth.
    httplib::Headers headers = {
        {"Authorization",              "Bearer " + api_key},
        {"User-Agent",                 ua_header},
        {"X-GpuFlight-Client-Version", kClientVersion},
        {"X-GpuFlight-Wire-Version",   kWireVersion},
        {"X-GpuFlight-Session-Id",     session_id},
        {"Content-Encoding",           "gzip"},
    };
    // Hostname is optional from the backend's perspective; only send
    // the header when we have something meaningful. Avoids advertising
    // a blank "" hostname for hosts with no resolvable name.
    if (!hostname.empty()) {
        headers.emplace("X-GpuFlight-Hostname", hostname);
    }

    auto res = client.Post(path.c_str(), headers, gzipped_body, "application/x-ndjson");
    if (!res) {
        std::ostringstream os;
        os << "transport error httplib::Error=" << static_cast<int>(res.error());
        out.failure_reason = os.str();
        out.outcome = StreamPostOutcome::TransientFailure;
        return out;
    }
    const int status = res->status;
    if (status >= 200 && status < 300) {
        parseStreamResponse(res->body, out);
        out.outcome = StreamPostOutcome::Ok;
        return out;
    }
    // 404 is reserved for "endpoint doesn't exist." We treat it as the
    // sentinel for an old backend that pre-dates v1.2's /stream route.
    // No retry — every subsequent chunk will hit the same 404.
    if (status == 404) {
        out.failure_reason =
            "gpufl client v1.2 requires backend v1.2+. The "
            "/api/v1/events/stream endpoint is missing (HTTP 404) — "
            "upgrade your backend or downgrade gpufl-client to v1.1.x.";
        out.outcome = StreamPostOutcome::OldBackend404;
        return out;
    }
    if (status == 401 || status == 403) {
        out.failure_reason = "auth failure (HTTP " + std::to_string(status) + ")";
        out.outcome = StreamPostOutcome::AuthFailure;
        return out;
    }
    if (status == 413) {
        // Body exceeded the backend's decompressed cap (128 MB) — we
        // pre-check file sizes against it, so this means the check and
        // the server disagree. Treat as client error (don't retry) so
        // the caller logs + skips the file.
        out.failure_reason =
            "file exceeded backend body limit (HTTP 413) — rotated log "
            "larger than the backend's decompressed cap?";
        out.outcome = StreamPostOutcome::ClientError;
        return out;
    }
    if (status >= 500) {
        out.failure_reason = "server error (HTTP " + std::to_string(status) + ")";
        out.outcome = StreamPostOutcome::TransientFailure;
        return out;
    }
    out.failure_reason = "client error (HTTP " + std::to_string(status) + ")";
    out.outcome = StreamPostOutcome::ClientError;
    return out;
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

/// v1.2: session_id comes from the parent subdirectory name (already
/// populated on every DiscoveredFile). We still parse the first
/// `job_start` event to get ts_ns, which drives "default = latest
/// session" ordering. Sessions with no job_start (interrupted before
/// init?) get ts_ns=0 — they sort to the oldest position.
std::vector<SessionInfo> discoverSessions(const std::vector<DiscoveredFile>& files) {
    // Group by session_id (from subdir name), pick earliest ts_ns.
    std::map<std::string, int64_t> seen;  // sid → earliest ts_ns observed
    for (const auto& f : files) {
        if (f.session_id.empty()) continue;
        if (seen.find(f.session_id) == seen.end()) {
            seen[f.session_id] = 0;  // default — overwritten if we find a job_start below
        }
        NdjsonReader reader(f.path, f.compressed);
        if (!reader.ok()) continue;
        std::string line;
        while (reader.readLine(line)) {
            if (line.empty()) continue;
            const std::string type = fastExtractType(line);
            if (type != "job_start") continue;
            const int64_t ts = fastExtractTsNs(line);
            const auto it = seen.find(f.session_id);
            if (it->second == 0 || ts < it->second) {
                it->second = ts;
            }
            break;  // one job_start per file is enough
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
                  return a.session_id < b.session_id;
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

    // v1.2 legacy-format detection. Fires when the user's logs are in
    // the pre-v1.2 flat layout (`<base>.<channel>.log[.gz]`) instead
    // of the v1.2 per-session subdirectory layout
    // (`<base>/<session_id>/<channel>.log[.gz]`). Two places to check:
    //
    //   1. Inside parts.directory itself — the user passed the parent
    //      dir of a pre-v1.2 install. Files like `myapp.device.log`
    //      sit at the top level alongside session subdirs.
    //   2. At parts.directory's parent, matching the basename — the
    //      user passed the pre-v1.2 prefix path `/tmp/myapp` while
    //      files are at `/tmp/myapp.device.log` etc.
    //
    // Either case → emit a single warning that points at the migration
    // path. We DON'T attempt to upload — the on-disk shape doesn't
    // match what discovery now walks, and silently uploading nothing
    // is worse than a loud error.
    auto emitLegacyWarning = [&](const fs::path& legacy_at) {
        result.warnings.emplace_back(
            "uploadLogs: detected pre-v1.2 flat log layout at " +
            legacy_at.string() + ". v1.2 expects per-session "
            "subdirectories (<log_path>/<session_id>/<channel>.log). "
            "Delete the old files (gpufl.clean_logs() / `rm "
            "<log_path>*.{device,scope,system}.log*`) and re-run the "
            "session to generate logs in the new layout.");
    };
    if (detectLegacyLayoutAt(parts.directory)) {
        emitLegacyWarning(parts.directory);
        return result;
    }
    if (parts.directory.has_parent_path()) {
        const fs::path parent = parts.directory.parent_path();
        const std::string basename = parts.directory.filename().string();
        if (!basename.empty()) {
            // Look for files matching `<basename>.<channel>.log[.gz]`
            // directly in the parent dir.
            static const std::array<const char*, 3> kChannels = {
                "device", "scope", "system"};
            for (const char* ch : kChannels) {
                for (const std::string& suffix : {".log", ".log.gz"}) {
                    const fs::path p = parent / (basename + "." + ch + suffix);
                    if (fs::exists(p, ec)) {
                        emitLegacyWarning(p);
                        return result;
                    }
                }
            }
        }
    }

    if (!fs::exists(parts.directory, ec) || !fs::is_directory(parts.directory, ec)) {
        result.warnings.emplace_back(
            "uploadLogs: log directory does not exist: " + parts.directory.string());
        return result;
    }

    // ── Discover files + sessions ────────────────────────────────────
    auto files = discoverFiles(parts);
    if (files.empty()) {
        GFL_LOG_DEBUG("[uploadLogs] no session subdirs found in ",
                      parts.directory.string());
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
                // The events count is only known when the upload ran
                // against a legacy synchronous backend; omit it at 0.
                const std::string events_note = it->second.events > 0
                    ? " (" + std::to_string(it->second.events) + " events)"
                    : "";
                result.warnings.emplace_back(
                    "Session '" + tsid + "' was already uploaded on " +
                    it->second.completed_at_iso8601 + events_note +
                    ". Pass force=true (CLI: --force) to re-upload.");
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

    // Stamped on every chunk so the backend can attribute events to
    // the originating host. Resolved once per upload — getLocalHostname
    // caches after first call.
    const std::string envelope_hostname = getLocalHostname();

    // File-size guards. The upload unit is one rotated file; these only
    // reject pathological inputs the backend (or its fronting infra)
    // would refuse anyway:
    //   - compressed: Cloud Run hard-caps a request body at 32 MiB —
    //     leave headroom. (Infra bound — unrelated to the rotate size.)
    //   - decompressed: derived from the rotator's threshold (the file
    //     size IS the rotate setting): 2× covers the one-line overshoot
    //     the rotator allows plus any future modest bump, and matches
    //     the backend's 128 MB streaming cap
    //     (StreamEventIngestionConstants.MAX_DECOMPRESSED_BODY_BYTES).
    //     Pre-checking the gzip ISIZE trailer avoids shipping megabytes
    //     just to be 413'd.
    // Real files sit at the 64 MiB rotate default (≈ 4–8 MB gz), far
    // under both limits.
    static constexpr std::size_t kMaxCompressedPostBytes   = 30 * 1024 * 1024;
    static constexpr std::size_t kMaxDecompressedFileBytes = 2 * Logger::kDefaultRotateBytes;

    // Progress reporting state.
    auto last_progress_time = upload_start;
    std::size_t bytes_since_last_progress = 0;
    auto maybeLogProgress = [&](const bool force) {
        if (!opts.report_progress) return;
        const auto now = std::chrono::steady_clock::now();
        const auto elapsed_since = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_progress_time).count();
        const bool by_time = elapsed_since >= opts.progress_log_interval_ms;
        const bool by_bytes = bytes_since_last_progress >= opts.progress_log_interval_bytes;
        if (!force && !by_time && !by_bytes) return;
        const auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - upload_start).count();
        // Two distinct numbers: files visited so far, and target
        // sessions for this upload. The pre-v1.2 form was
        // "%zu/%zu session(s)" which read like "F of S sessions
        // complete" but was actually mixing files (numerator) and
        // sessions (denominator) — fixed by labeling both axes.
        std::fprintf(stderr,
                     "[gpufl::upload] %zu events uploaded (%zu MB), "
                     "%zu file(s), %zu session(s), %llds elapsed\n",
                     result.events_uploaded,
                     result.bytes_uploaded / (1024 * 1024),
                     result.files_processed,
                     targets.size(),
                     static_cast<long long>(total_elapsed));
        last_progress_time = now;
        bytes_since_last_progress = 0;
    };

    auto budgetExpired = [&] {
        return elapsedMs() >= opts.total_timeout_ms;
    };

    bool auth_failed     = false;
    bool budget_aborted  = false;
    bool old_backend_404 = false;   // /events/stream missing → abort all sessions

    // Per-file POST with retry. The body is the file's gzipped bytes
    // (either the .log.gz content verbatim or a just-gzipped .log);
    // `decompressed_bytes` drives the byte accounting and progress.
    //
    // Returns true if the file was either accepted (possibly with
    // per-line warnings from a legacy sync backend) OR refused with a
    // recorded warning AND the upload should continue. Returns false
    // only when the WHOLE upload must abort (auth failure, budget
    // exhausted, or 404 from an old backend that doesn't have /stream).
    //
    // `session_events` accumulates the per-session accepted count —
    // only legacy synchronous backends report one; the async-accept
    // path (202 + spool_id) has no per-line counts at POST time.
    //
    // `skipped_out` is set when the file was given up on (4xx, retries
    // exhausted) but the upload continues — the caller must then NOT
    // mark the session completed, so a re-run retries the hole.
    auto postFile = [&](const std::string& session_id,
                        const std::string& display_name,
                        const std::string& gz_body,
                        const std::size_t  decompressed_bytes,
                        std::size_t&       session_events,
                        bool&              skipped_out) -> bool {
        skipped_out = false;
        if (gz_body.empty()) return true;
        if (budgetExpired()) {
            budget_aborted = true;
            return false;
        }

        for (int attempt = 0; attempt <= opts.max_retries; ++attempt) {
            const StreamPostResult r = postStreamChunk(
                *client, api_path, session_id, envelope_hostname,
                opts.api_key, ua_header, gz_body);

            if (r.outcome == StreamPostOutcome::Ok) {
                result.bytes_uploaded     += decompressed_bytes;
                bytes_since_last_progress += decompressed_bytes;
                // Two backend shapes:
                //   1. async-accept (Phase 3a+): 202 + spool_id, no
                //      per-line counts — the worker ingests out-of-band.
                //   2. legacy synchronous: explicit accepted/rejected
                //      counts + per-line errors. Surface partials as
                //      warnings, trust the server's accepted count.
                if (r.async_accepted) {
                    if (!r.spool_id.empty()) {
                        result.spool_ids.push_back(r.spool_id);
                    }
                    return true;
                }
                result.events_uploaded += r.accepted;
                session_events         += r.accepted;
                if (r.rejected > 0) {
                    result.warnings.push_back(
                        display_name + ": backend rejected " +
                        std::to_string(r.rejected) + " line(s)");
                }
                for (const auto& le : r.line_errors) {
                    result.warnings.push_back(
                        display_name + " line " + std::to_string(le.line) +
                        " (type=" + le.type + ") rejected: " + le.reason);
                }
                return true;
            }
            if (r.outcome == StreamPostOutcome::OldBackend404) {
                // Backend predates the /stream endpoint. No retry —
                // every file would hit the same 404. Abort the whole
                // upload with a migration-hint warning.
                result.warnings.push_back(r.failure_reason);
                old_backend_404 = true;
                return false;
            }
            if (r.outcome == StreamPostOutcome::AuthFailure) {
                result.warnings.push_back(
                    "POST /events/stream failed: " + r.failure_reason +
                    " — aborting remaining uploads");
                auth_failed = true;
                return false;
            }
            if (r.outcome == StreamPostOutcome::ClientError) {
                // 4xx that isn't auth or 404 (e.g. 413/400). Log and
                // skip the file — retrying won't help.
                result.warnings.push_back(
                    "POST /events/stream failed: " + r.failure_reason +
                    " — skipping " + display_name);
                skipped_out = true;
                return true;
            }
            // TransientFailure: retry on the loop's next iteration if
            // we have budget for both more retries AND wall time.
            std::string fail_reason = r.failure_reason;
            if (attempt < opts.max_retries && !budgetExpired()) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(opts.retry_delay_ms));
                continue;
            }
            result.warnings.push_back(
                "POST /events/stream failed after " +
                std::to_string(attempt + 1) + " attempt(s): " + fail_reason +
                " — skipping " + display_name);
            skipped_out = true;
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
    // For each target session: POST each of its files whole, in the
    // discovery order (oldest rotation first, active file last). A
    // file lives in its session's subdirectory, so a file maps to
    // exactly one session — no per-line filtering needed; the backend
    // validates every line's session_id against the header anyway.
    //
    // Lifecycle ordering rides on the file order: job_start is the
    // first line of the oldest file (POSTed first) and shutdown the
    // last line of the active file (POSTed last). Arrival order is
    // thus preserved; ingest-COMPLETION order under the async queue is
    // the upload-plan U3 ordering chain's job, same as before.
    for (const auto& target : targets) {
        if (auth_failed || budget_aborted || old_backend_404) break;
        const std::string& current_sid = target.session_id;

        std::size_t events_from_session = 0;   // known only on legacy sync backends
        bool session_ok = true;
        bool session_had_skips = false;
        bool posted_anything = false;

        GFL_LOG_DEBUG("[uploadLogs] session ", current_sid, " starting");

        for (auto& f : files) {
            if (auth_failed || budget_aborted || old_backend_404) {
                session_ok = false;
                break;
            }
            if (f.session_id != current_sid) continue;

            const std::string basename   = f.path.filename().string();
            const std::string cursor_key = current_sid + "/" + basename;

            // Rotated files are immutable once written — skip the ones
            // a previous run already shipped (crash-resume). The active
            // file (rotation 0) is never cursor-skipped; completed
            // sessions were already filtered out above.
            if (!opts.force && f.rotation_index >= 1 &&
                cursor.uploaded_files.count(cursor_key) > 0) {
                result.files_skipped_by_cursor++;
                continue;
            }

            std::string body = readFileBytes(f.path);
            if (body.empty()) {
                result.warnings.push_back("Could not read " + basename + " — skipping");
                session_had_skips = true;
                continue;
            }

            // .log.gz goes on the wire AS-IS; a plain .log (rare — the
            // discovery pass gzips orphans in place) is gzipped here.
            std::size_t decompressed_bytes;
            if (f.compressed) {
                decompressed_bytes = gzipDecompressedSizeHint(body);
            } else {
                decompressed_bytes = body.size();
                body = gzipString(body);
                if (body.empty()) {
                    result.warnings.push_back(
                        "gzip compression failed for " + basename + " — skipping");
                    session_had_skips = true;
                    continue;
                }
            }
            if (decompressed_bytes == 0) continue;   // empty file — nothing to send
            if (body.size() > kMaxCompressedPostBytes ||
                decompressed_bytes > kMaxDecompressedFileBytes) {
                result.warnings.push_back(
                    basename + " is larger than the per-request limit (" +
                    std::to_string(body.size()) + " B compressed / " +
                    std::to_string(decompressed_bytes) + " B raw) — skipping");
                session_had_skips = true;
                continue;
            }

            bool skipped = false;
            if (!postFile(current_sid, basename, body, decompressed_bytes,
                          events_from_session, skipped)) {
                session_ok = false;
                break;
            }
            if (skipped) {
                session_had_skips = true;
                continue;
            }
            posted_anything = true;
            files_visited.insert(basename);

            // Persist per-file progress for rotated files so a crash
            // mid-session resumes after the last shipped file instead
            // of starting over.
            if (f.rotation_index >= 1) {
                cursor.uploaded_files.insert(cursor_key);
                if (!writeCursor(parts.directory, opts.cursor_filename, cursor)) {
                    result.warnings.emplace_back(
                        "Could not write cursor file " + opts.cursor_filename);
                }
            }
            maybeLogProgress(/*force=*/false);
        }

        // Mark this session as completed in the cursor only when every
        // file shipped (a skipped file is a hole — leave the session
        // incomplete so a re-run retries it). Persisted after each
        // session so a mid-run crash keeps the finished ones skipped.
        if (session_ok && posted_anything && !session_had_skips) {
            CompletedSession cs;
            cs.completed_at_iso8601 = nowIso8601Utc();
            cs.events = events_from_session;
            cursor.completed_sessions[current_sid] = cs;
            if (!writeCursor(parts.directory, opts.cursor_filename, cursor)) {
                result.warnings.emplace_back(
                    "Could not write cursor file " + opts.cursor_filename);
            }
            GFL_LOG_DEBUG("[uploadLogs] session ", current_sid, " complete");
        }
    }

    result.files_processed = files_visited.size();

    // ── Wrap up ──────────────────────────────────────────────────────
    if (budget_aborted) {
        result.warnings.emplace_back(
            "Total timeout (" + std::to_string(opts.total_timeout_ms) +
            " ms) exceeded — upload aborted");
    }
    // Success = no auth failure, no budget exhaustion, and the backend
    // wasn't an ancient pre-v1.2 one missing the /stream endpoint. The
    // 404 case has already pushed a clear migration-hint warning into
    // result.warnings via flushChunk's OldBackend404 branch.
    result.success = !auth_failed && !budget_aborted && !old_backend_404;
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

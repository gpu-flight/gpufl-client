// End-to-end tests for gpufl::uploadLogs.
//
// U1 wire model: ONE gzipped NDJSON POST per rotated log file to
// /api/v1/events/stream, X-GpuFlight-Session-Id in the header, the
// file's bytes shipped as-is (no client-side chunking or per-line
// filtering - the backend validates per line). The capture server here
// listens on that single route, decompresses the body, and exposes
// both per-request and per-line views so tests can assert on either
// granularity. ("Chunk" in the helper names = one captured request.)
//
// Earlier revisions tested the per-event EventWrapper format (removed
// in v1.2) and the 5000-line client-side chunking (removed in U1).

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <httplib.h>
#include <zlib.h>

#include "gpufl/upload/upload_logs.hpp"
#include "gpufl/core/version.hpp"

namespace fs = std::filesystem;

namespace {

// ─────────────────────────────────────────────────────────────────────
// Test helpers
// ─────────────────────────────────────────────────────────────────────

/// Gunzip a body the server received with Content-Encoding: gzip.
std::string gunzipString(const std::string& gzipped) {
    if (gzipped.empty()) return {};
    z_stream zs{};
    zs.next_in  = reinterpret_cast<Bytef*>(const_cast<char*>(gzipped.data()));
    zs.avail_in = static_cast<uInt>(gzipped.size());
    // +16 = gzip wrapper (matches what the client uses in upload_logs.cpp)
    if (inflateInit2(&zs, 15 + 16) != Z_OK) return {};
    std::string out;
    std::vector<char> buf(64 * 1024);
    while (true) {
        zs.next_out  = reinterpret_cast<Bytef*>(buf.data());
        zs.avail_out = static_cast<uInt>(buf.size());
        int rc = inflate(&zs, Z_NO_FLUSH);
        if (rc == Z_STREAM_END || rc == Z_OK) {
            out.append(buf.data(), buf.size() - zs.avail_out);
            if (rc == Z_STREAM_END) break;
            if (zs.avail_in == 0 && zs.avail_out > 0) break;
            continue;
        }
        inflateEnd(&zs);
        return {};
    }
    inflateEnd(&zs);
    return out;
}

/// Split an NDJSON body on '\n', dropping blank lines.
std::vector<std::string> splitLines(const std::string& body) {
    std::vector<std::string> out;
    std::stringstream ss(body);
    std::string line;
    while (std::getline(ss, line)) {
        if (line.empty()) continue;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) out.push_back(std::move(line));
    }
    return out;
}

/// Pull `"type":"<value>"` out of a raw NDJSON line. Mirrors the
/// fastExtractType helper used inside upload_logs.cpp.
std::string extractType(const std::string& line) {
    static const std::string kKey = "\"type\":\"";
    auto pos = line.find(kKey);
    if (pos == std::string::npos) return {};
    auto start = pos + kKey.size();
    auto end   = line.find('"', start);
    if (end == std::string::npos) return {};
    return line.substr(start, end - start);
}

std::string extractSessionId(const std::string& line) {
    static const std::string kKey = "\"session_id\":\"";
    auto pos = line.find(kKey);
    if (pos == std::string::npos) return {};
    auto start = pos + kKey.size();
    auto end   = line.find('"', start);
    if (end == std::string::npos) return {};
    return line.substr(start, end - start);
}

// ─────────────────────────────────────────────────────────────────────
// Capture server - listens on /api/v1/events/stream
// ─────────────────────────────────────────────────────────────────────

/// Localhost server that captures every chunk POSTed by uploadLogs.
/// Each chunk records: the decompressed NDJSON body, the per-chunk
/// headers, and the individual lines parsed out of the body. Tests can
/// then assert at chunk-granularity (e.g. "exactly N chunks"), or
/// flatten via `allLines()` for per-event semantics matching the
/// pre-v1.2 test surface.
class CaptureServer {
   public:
    struct Chunk {
        std::string session_id;       // X-GpuFlight-Session-Id header
        std::string hostname;         // X-GpuFlight-Hostname header (may be empty)
        std::string user_agent;
        std::string authorization;
        std::string client_version;
        std::string wire_version;
        std::string content_encoding;
        std::string body;             // decompressed NDJSON body
        std::vector<std::string> lines;
        // For convenience in tests: vector of (event_type, session_id)
        // extracted from each line.
        std::vector<std::pair<std::string, std::string>> events;
    };

    /// Response shape selector. Lets a single test fixture exercise
    /// both backend lineages:
    ///   - LegacySync - pre-Phase 3a backend: HTTP 200 with
    ///     `{accepted, rejected, errors}`.
    ///   - AsyncSpool - Phase 3a+ backend: HTTP 202 with
    ///     `{accepted_for_processing, spool_id}`.
    /// The body-recording logic is identical for both; only the
    /// response status code + body differ.
    enum class ResponseShape { LegacySync, AsyncSpool };

    /// Construct with optional HTTP status override. force_status=200
    /// is the happy path; 401/403 simulates auth failure; 404 the old-
    /// backend case; 500 a transient failure. `shape` picks which wire
    /// shape the 2xx body takes.
    explicit CaptureServer(int force_status = 200,
                           std::size_t fail_first_n_with_5xx = 0,
                           ResponseShape shape = ResponseShape::LegacySync)
        : force_status_(force_status),
          fail_first_n_(fail_first_n_with_5xx),
          shape_(shape) {
        server_.Post("/api/v1/events/stream",
            [this](const httplib::Request& req, httplib::Response& res) {
                std::lock_guard<std::mutex> lk(mu_);

                Chunk c;
                c.session_id       = req.get_header_value("X-GpuFlight-Session-Id");
                c.hostname         = req.get_header_value("X-GpuFlight-Hostname");
                c.user_agent       = req.get_header_value("User-Agent");
                c.authorization    = req.get_header_value("Authorization");
                c.client_version   = req.get_header_value("X-GpuFlight-Client-Version");
                c.wire_version     = req.get_header_value("X-GpuFlight-Wire-Version");
                c.content_encoding = req.get_header_value("Content-Encoding");

                // With CPPHTTPLIB_ZLIB_SUPPORT enabled (see top-level
                // CMakeLists.txt), cpp-httplib's server auto-decompresses
                // incoming bodies that carry Content-Encoding: gzip
                // BEFORE the handler runs - req.body is already the raw
                // NDJSON. (Without ZLIB support, the server would have
                // returned 415 before reaching us.)
                c.body = req.body;
                c.lines = splitLines(c.body);
                for (const auto& l : c.lines) {
                    c.events.emplace_back(extractType(l), extractSessionId(l));
                }
                captured_.push_back(std::move(c));

                // Decide status. The "fail first N then succeed" mode
                // lets tests exercise the retry path: the first N
                // chunks return 500, subsequent ones return 200.
                int status = force_status_;
                if (fail_first_n_ > 0 && captured_.size() <= fail_first_n_) {
                    status = 500;
                }
                // In async mode the natural success code is 202; map
                // any 2xx the test asked for to 202 so the wire shape
                // matches what Phase 3a backends actually send.
                if (shape_ == ResponseShape::AsyncSpool &&
                    status >= 200 && status < 300) {
                    status = 202;
                }

                res.status = status;
                if (status >= 200 && status < 300) {
                    if (shape_ == ResponseShape::AsyncSpool) {
                        // Phase 3a+ wire shape. spool_id is a synthetic
                        // sequence - real GcsSpoolBackend uses UUIDs
                        // but tests don't need the entropy.
                        std::ostringstream rsp;
                        rsp << "{\"accepted_for_processing\":true,"
                            << "\"spool_id\":\"spool-"
                            << captured_.size() << "\"}";
                        res.set_content(rsp.str(), "application/json");
                    } else {
                        // Legacy sync wire shape: accepted = line
                        // count, rejected = 0, no errors.
                        std::ostringstream rsp;
                        rsp << "{\"accepted\":" << captured_.back().lines.size()
                            << ",\"rejected\":0,\"errors\":[]}";
                        res.set_content(rsp.str(), "application/json");
                    }
                } else {
                    res.set_content("{\"error\":\"forced\"}", "application/json");
                }
            });

        port_ = server_.bind_to_any_port("127.0.0.1");
        thread_ = std::thread([this] { server_.listen_after_bind(); });
        for (int i = 0; i < 200 && !server_.is_running(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    ~CaptureServer() {
        server_.stop();
        if (thread_.joinable()) thread_.join();
    }

    std::string base_url() const {
        return "http://127.0.0.1:" + std::to_string(port_);
    }

    std::vector<Chunk> snapshot() {
        std::lock_guard<std::mutex> lk(mu_);
        return captured_;
    }

    /// Flatten across all chunks: returns every NDJSON line POSTed, in
    /// arrival order. The pre-v1.2 tests called this "captures" - keep
    /// the same idea so semantically-equivalent assertions read the
    /// same.
    std::vector<std::string> allLines() {
        std::lock_guard<std::mutex> lk(mu_);
        std::vector<std::string> out;
        for (const auto& c : captured_) {
            for (const auto& l : c.lines) out.push_back(l);
        }
        return out;
    }

    /// Same idea but returns (event_type, session_id) pairs.
    std::vector<std::pair<std::string, std::string>> allEvents() {
        std::lock_guard<std::mutex> lk(mu_);
        std::vector<std::pair<std::string, std::string>> out;
        for (const auto& c : captured_) {
            for (const auto& e : c.events) out.push_back(e);
        }
        return out;
    }

   private:
    httplib::Server      server_;
    std::thread          thread_;
    int                  port_{0};
    int                  force_status_;
    std::size_t          fail_first_n_;
    ResponseShape        shape_;
    std::mutex           mu_;
    std::vector<Chunk>   captured_;
};

/// Write `lines` (one NDJSON event per line) to `path`, optionally gzipped.
void writeLog(const fs::path& path, const std::vector<std::string>& lines,
              bool gzip = false) {
    if (gzip) {
        gzFile gz = gzopen(path.string().c_str(), "wb");
        ASSERT_NE(gz, nullptr) << "Could not open " << path << " for gz write";
        for (const auto& l : lines) {
            const std::string with_nl = l + "\n";
            gzwrite(gz, with_nl.data(), static_cast<unsigned>(with_nl.size()));
        }
        gzclose(gz);
    } else {
        std::ofstream out(path);
        ASSERT_TRUE(out.is_open()) << "Could not open " << path << " for write";
        for (const auto& l : lines) out << l << "\n";
    }
}

/// Build a minimal session under the v1.2 per-session-subdirectory
/// layout. Files live at `<root>/<sid>/<channel>[.N].log[.gz]`. The
/// returned string is the `log_path` to pass to uploadLogs - in v1.2
/// that's just the parent directory (root), not a prefix.
std::string makeMinimalSession(const fs::path& root,
                               const std::string& sid = "s1") {
    const fs::path session_dir = root / sid;
    fs::create_directories(session_dir);
    // Rotated (older) - device channel only, gzipped. job_start is
    // Channel::All in the client, so it heads every channel's oldest
    // file; one copy is enough for the device chain here.
    writeLog(session_dir / "device.1.log.gz", {
        R"({"type":"job_start","session_id":")" + sid + R"(","app":"test","pid":1,"ts_ns":100})",
        R"({"type":"kernel_event_batch","session_id":")" + sid + R"(","batch_id":1,"rows":[[0,1,0,1000,101,0,32,1]]})",
    }, /*gzip=*/true);
    // Active device .log - newer events. shutdown is Channel::All in
    // the client: every channel's active file ends with a copy.
    writeLog(session_dir / "device.log", {
        R"({"type":"kernel_event_batch","session_id":")" + sid + R"(","batch_id":2,"rows":[[2000,2,0,1500,102,0,64,1]]})",
        R"({"type":"shutdown","session_id":")" + sid + R"(","app":"test","pid":1,"ts_ns":9999})",
    });
    // Scope channel, active only - carries its own shutdown copy, like
    // the real logger writes.
    writeLog(session_dir / "scope.log", {
        R"({"type":"scope_event_batch","session_id":")" + sid + R"(","batch_id":1,"rows":[[0,1,1,0,0],[5000,1,1,1,0]]})",
        R"({"type":"shutdown","session_id":")" + sid + R"(","app":"test","pid":1,"ts_ns":9999})",
    });
    return root.string();
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────
// Happy path
// ─────────────────────────────────────────────────────────────────────

TEST(UploadLogs, UploadsAllEventsAcrossChannelsAndRotation) {
    const fs::path tmp = fs::temp_directory_path() / "gpufl_upload_test_happy";
    fs::remove_all(tmp);
    const std::string log_path = makeMinimalSession(tmp);

    CaptureServer srv;

    gpufl::UploadOptions opts;
    opts.log_path        = log_path;
    opts.backend_url     = srv.base_url();
    opts.api_key         = "gpfl_test";
    opts.report_progress = false;

    const auto result = gpufl::uploadLogs(opts);

    EXPECT_TRUE(result.success) << "warnings: "
        << (result.warnings.empty() ? "<none>" : result.warnings.front());
    EXPECT_EQ(result.events_uploaded, 6u)
        << "Expected 6 NDJSON events across all files (job_start, two "
        << "kernel batches, one scope batch, two shutdown copies).";
    EXPECT_EQ(result.files_processed, 3u);  // device.1.log.gz + device.log + scope.log
    EXPECT_TRUE(result.warnings.empty()) << "Unexpected warnings on happy path";

    // One POST per file - the U1 contract.
    EXPECT_EQ(srv.snapshot().size(), 3u);

    const auto events = srv.allEvents();
    ASSERT_EQ(events.size(), 6u);

    // job_start must arrive first (first line of the oldest file of the
    // first channel). The last file's last line is its shutdown copy -
    // files go (channel asc, rotation desc, active last).
    EXPECT_EQ(events.front().first, "job_start");
    EXPECT_EQ(events.back().first,  "shutdown");

    fs::remove_all(tmp);
}

TEST(UploadLogs, AsyncAccept202_NoEventCounts_SpoolIdsRecorded) {
    // Phase 3a+ backends return HTTP 202 with
    // `{accepted_for_processing, spool_id}` and do NOT carry per-line
    // accepted/rejected counts. The client must:
    //   - treat the 202 as success (no warning),
    //   - leave events_uploaded at 0 (nothing was dispatched yet -
    //     bytes/files are the progress numbers on this path),
    //   - stash spool_id into UploadResult.spool_ids so operators can
    //     correlate with backend logs (one per file POST).
    const fs::path tmp = fs::temp_directory_path() / "gpufl_upload_test_async";
    fs::remove_all(tmp);
    const std::string log_path = makeMinimalSession(tmp);

    CaptureServer srv(/*force_status=*/200, /*fail_first_n=*/0,
                      CaptureServer::ResponseShape::AsyncSpool);
    gpufl::UploadOptions opts;
    opts.log_path        = log_path;
    opts.backend_url     = srv.base_url();
    opts.api_key         = "gpfl_test";
    opts.report_progress = false;

    const auto result = gpufl::uploadLogs(opts);

    EXPECT_TRUE(result.success) << "warnings: "
        << (result.warnings.empty() ? "<none>" : result.warnings.front());
    EXPECT_EQ(result.events_uploaded, 0u)
        << "Async-accept responses carry no per-line counts - "
        << "events_uploaded must stay 0 rather than guess.";
    EXPECT_GT(result.bytes_uploaded, 0u);
    EXPECT_EQ(result.files_processed, 3u);
    EXPECT_TRUE(result.warnings.empty())
        << "Async path must not emit warnings: "
        << (result.warnings.empty() ? "<none>" : result.warnings.front());
    EXPECT_EQ(result.spool_ids.size(), 3u)
        << "Each 202 response should add one spool_id (one per file).";
    for (const auto& sid : result.spool_ids) {
        EXPECT_NE(sid.find("spool-"), std::string::npos);
    }
    fs::remove_all(tmp);
}

TEST(UploadLogs, ChunkBodyIsGzippedNdjsonNotEventWrapper) {
    // Regression guard against accidentally falling back to the v1.1
    // per-event EventWrapper format. The new wire format is raw NDJSON
    // (one event per line), gzipped, with the session-id in a header.
    // No "data" field, no "agentSendingTime" field - just the events
    // themselves, byte-identical to what's on disk.
    const fs::path tmp = fs::temp_directory_path() / "gpufl_upload_test_ndjson";
    fs::remove_all(tmp);
    const std::string log_path = makeMinimalSession(tmp);

    CaptureServer srv;
    gpufl::UploadOptions opts;
    opts.log_path        = log_path;
    opts.backend_url     = srv.base_url();
    opts.api_key         = "x";
    opts.report_progress = false;

    const auto result = gpufl::uploadLogs(opts);
    ASSERT_TRUE(result.success);

    const auto chunks = srv.snapshot();
    ASSERT_FALSE(chunks.empty());
    for (const auto& c : chunks) {
        EXPECT_EQ(c.content_encoding, "gzip")
            << "Every chunk should be gzipped on the wire.";
        EXPECT_FALSE(c.session_id.empty())
            << "Every chunk must carry X-GpuFlight-Session-Id.";
        // Body should NOT look like an EventWrapper envelope. The
        // legacy format started with `{"data":` - the new format
        // starts directly with the first event's `{"type":`.
        EXPECT_EQ(c.body.find("\"data\":\""), std::string::npos)
            << "Chunk body should not be EventWrapper-wrapped: " << c.body;
        EXPECT_EQ(c.body.find("\"agentSendingTime\":"), std::string::npos)
            << "Chunk body should not be EventWrapper-wrapped: " << c.body;
        // Every parsed line should be a real event.
        for (const auto& l : c.lines) {
            EXPECT_NE(l.find("\"type\":\""), std::string::npos);
            EXPECT_NE(l.find("\"session_id\":\""), std::string::npos);
        }
    }

    fs::remove_all(tmp);
}

TEST(UploadLogs, AuthHeaderAndVersionHeadersPresentOnEveryChunk) {
    const fs::path tmp = fs::temp_directory_path() / "gpufl_upload_test_headers";
    fs::remove_all(tmp);
    const std::string log_path = makeMinimalSession(tmp);

    CaptureServer srv;
    gpufl::UploadOptions opts;
    opts.log_path        = log_path;
    opts.backend_url     = srv.base_url();
    opts.api_key         = "secret_token";
    opts.report_progress = false;

    const auto result = gpufl::uploadLogs(opts);
    ASSERT_TRUE(result.success);

    const auto chunks = srv.snapshot();
    ASSERT_FALSE(chunks.empty());
    for (const auto& c : chunks) {
        EXPECT_EQ(c.authorization,  "Bearer secret_token");
        EXPECT_EQ(c.client_version, gpufl::kClientVersion);
        EXPECT_EQ(c.wire_version,   gpufl::kWireVersion);
        EXPECT_EQ(c.session_id,     "s1");
        EXPECT_FALSE(c.user_agent.empty());
    }

    fs::remove_all(tmp);
}

TEST(UploadLogs, FileBodyShipsAsIs_SessionValidationIsServerSide) {
    // U1 ships the file's bytes verbatim - there is no per-line
    // client-side filtering anymore. A corrupted file containing
    // another session's lines therefore reaches the wire as-is; the
    // BACKEND rejects mismatched lines (it has always validated every
    // line's session_id against the X-GpuFlight-Session-Id header).
    // Normal operation never produces such files - the rotator writes
    // only the active session_id into <sid>/<channel>.log.
    const fs::path tmp = fs::temp_directory_path() / "gpufl_upload_test_filter";
    fs::remove_all(tmp);
    fs::create_directories(tmp);

    fs::create_directories(tmp / "s1");
    writeLog(tmp / "s1" / "device.log", {
        R"({"type":"job_start","session_id":"s1","ts_ns":1})",
        R"({"type":"kernel_event_batch","session_id":"s2","rows":[]})",   // wrong session - server's problem
        R"({"type":"kernel_event_batch","session_id":"s1","rows":[]})",
        R"({"type":"shutdown","session_id":"s1","ts_ns":99})",
    });

    CaptureServer srv;
    gpufl::UploadOptions opts;
    opts.log_path          = tmp.string();
    opts.backend_url       = srv.base_url();
    opts.api_key           = "x";
    opts.session_id_filter = "s1";
    opts.report_progress   = false;

    const auto result = gpufl::uploadLogs(opts);
    ASSERT_TRUE(result.success);

    const auto chunks = srv.snapshot();
    ASSERT_EQ(chunks.size(), 1u) << "One file → one POST.";
    EXPECT_EQ(chunks.front().session_id, "s1");
    EXPECT_EQ(chunks.front().lines.size(), 4u)
        << "The body is the file verbatim - including the s2 line the "
        << "backend will reject per-line.";

    fs::remove_all(tmp);
}

// ─────────────────────────────────────────────────────────────────────
// Cursor file
// ─────────────────────────────────────────────────────────────────────

TEST(UploadLogs, CursorRefusesRepeatUploadOfSameSession) {
    const fs::path tmp = fs::temp_directory_path() / "gpufl_upload_test_cursor_refuse";
    fs::remove_all(tmp);
    const std::string log_path = makeMinimalSession(tmp);

    CaptureServer srv;
    gpufl::UploadOptions opts;
    opts.log_path        = log_path;
    opts.backend_url     = srv.base_url();
    opts.api_key         = "x";
    opts.report_progress = false;

    const auto r1 = gpufl::uploadLogs(opts);
    ASSERT_TRUE(r1.success);
    const auto chunks_after_first = srv.snapshot().size();
    EXPECT_GE(chunks_after_first, 1u);
    EXPECT_TRUE(fs::exists(tmp / ".gpufl-upload-cursor.json"));

    // Second run: same session_id is in cursor.completed_sessions, so
    // the default mode refuses and emits a warning suggesting --force.
    const auto r2 = gpufl::uploadLogs(opts);
    EXPECT_FALSE(r2.success);
    EXPECT_EQ(r2.events_uploaded, 0u);
    ASSERT_FALSE(r2.warnings.empty());
    EXPECT_NE(r2.warnings.front().find("already uploaded"), std::string::npos);
    EXPECT_NE(r2.warnings.front().find("force"), std::string::npos);
    EXPECT_EQ(srv.snapshot().size(), chunks_after_first)
        << "Server should see no further chunks on refusal.";

    fs::remove_all(tmp);
}

TEST(UploadLogs, ForceOverridesCursorRefusal) {
    const fs::path tmp = fs::temp_directory_path() / "gpufl_upload_test_force";
    fs::remove_all(tmp);
    const std::string log_path = makeMinimalSession(tmp);

    CaptureServer srv;
    gpufl::UploadOptions opts;
    opts.log_path        = log_path;
    opts.backend_url     = srv.base_url();
    opts.api_key         = "x";
    opts.report_progress = false;

    const auto r1 = gpufl::uploadLogs(opts);
    ASSERT_TRUE(r1.success);
    const auto first_lines = srv.allLines().size();

    opts.force = true;
    const auto r2 = gpufl::uploadLogs(opts);
    EXPECT_TRUE(r2.success);
    EXPECT_EQ(r2.events_uploaded, r1.events_uploaded)
        << "force=true should re-upload every event from the session.";
    EXPECT_EQ(srv.allLines().size(), first_lines * 2)
        << "Server should see double the lines after a forced re-upload.";

    fs::remove_all(tmp);
}

// ─────────────────────────────────────────────────────────────────────
// Failure modes
// ─────────────────────────────────────────────────────────────────────

TEST(UploadLogs, AuthFailureShortCircuits) {
    const fs::path tmp = fs::temp_directory_path() / "gpufl_upload_test_auth";
    fs::remove_all(tmp);
    const std::string log_path = makeMinimalSession(tmp);

    CaptureServer srv(/*force_status=*/401);
    gpufl::UploadOptions opts;
    opts.log_path        = log_path;
    opts.backend_url     = srv.base_url();
    opts.api_key         = "bad_token";
    opts.max_retries     = 0;
    opts.report_progress = false;

    const auto result = gpufl::uploadLogs(opts);

    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.warnings.empty());
    // First chunk returns 401 → abort. No subsequent chunks.
    EXPECT_EQ(srv.snapshot().size(), 1u)
        << "Auth failure should short-circuit - no further chunks expected";

    fs::remove_all(tmp);
}

TEST(UploadLogs, OldBackend404SurfacesMigrationWarning) {
    // Backend predates v1.2 - /events/stream doesn't exist. We expect
    // a single 404, no retries, and a clear migration-hint warning.
    const fs::path tmp = fs::temp_directory_path() / "gpufl_upload_test_404";
    fs::remove_all(tmp);
    const std::string log_path = makeMinimalSession(tmp);

    CaptureServer srv(/*force_status=*/404);
    gpufl::UploadOptions opts;
    opts.log_path        = log_path;
    opts.backend_url     = srv.base_url();
    opts.api_key         = "x";
    opts.max_retries     = 3;       // confirm 404 doesn't trigger retries
    opts.report_progress = false;

    const auto result = gpufl::uploadLogs(opts);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.events_uploaded, 0u);
    ASSERT_FALSE(result.warnings.empty());
    // Look for the migration hint anywhere in the warnings - exact
    // wording can drift but the marker terms shouldn't.
    bool found_hint = false;
    for (const auto& w : result.warnings) {
        if (w.find("/api/v1/events/stream") != std::string::npos &&
            w.find("backend") != std::string::npos) {
            found_hint = true;
            break;
        }
    }
    EXPECT_TRUE(found_hint)
        << "Expected a 404 warning mentioning /api/v1/events/stream and backend";
    EXPECT_EQ(srv.snapshot().size(), 1u)
        << "404 must not trigger retries - every subsequent chunk would 404 too.";

    fs::remove_all(tmp);
}

TEST(UploadLogs, TransientFailureRetriesThenSucceeds) {
    // First chunk POST gets a 500; retry succeeds. No data loss.
    const fs::path tmp = fs::temp_directory_path() / "gpufl_upload_test_retry";
    fs::remove_all(tmp);
    const std::string log_path = makeMinimalSession(tmp);

    CaptureServer srv(/*force_status=*/200, /*fail_first_n=*/1);
    gpufl::UploadOptions opts;
    opts.log_path        = log_path;
    opts.backend_url     = srv.base_url();
    opts.api_key         = "x";
    opts.max_retries     = 2;
    opts.retry_delay_ms  = 1;
    opts.report_progress = false;

    const auto result = gpufl::uploadLogs(opts);
    EXPECT_TRUE(result.success) << "Should retry past the transient 500.";
    EXPECT_EQ(result.events_uploaded, 6u);
}

TEST(UploadLogs, MissingLogDirReturnsFailureNotThrow) {
    gpufl::UploadOptions opts;
    opts.log_path    = "/nonexistent/path/that/should/not/exist";
    opts.backend_url = "http://127.0.0.1:1";
    opts.api_key     = "x";
    opts.report_progress = false;

    const auto result = gpufl::uploadLogs(opts);
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.warnings.empty());
    EXPECT_EQ(result.events_uploaded, 0u);
}

TEST(UploadLogs, EmptyLogDirIsSuccessNoOp) {
    const fs::path tmp = fs::temp_directory_path() / "gpufl_upload_test_empty";
    fs::create_directories(tmp);
    fs::remove_all(tmp);
    fs::create_directories(tmp);

    CaptureServer srv;
    gpufl::UploadOptions opts;
    opts.log_path        = tmp.string();    // tmp exists but has no session subdirs
    opts.backend_url     = srv.base_url();
    opts.api_key         = "x";
    opts.report_progress = false;

    const auto result = gpufl::uploadLogs(opts);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.events_uploaded, 0u);
    EXPECT_EQ(result.files_processed, 0u);
    EXPECT_TRUE(srv.snapshot().empty());

    fs::remove_all(tmp);
}

TEST(UploadLogs, MalformedLinesShipAsIs_ServerJudgesPerLine) {
    // No client-side line parsing in U1 - a malformed line ships with
    // its file and the backend records it as a per-line rejection
    // (skip-and-continue; it never reaches SQL). The client neither
    // warns nor drops anything locally.
    const fs::path tmp = fs::temp_directory_path() / "gpufl_upload_test_bad_line";
    fs::remove_all(tmp);
    fs::create_directories(tmp);

    fs::create_directories(tmp / "s1");
    writeLog(tmp / "s1" / "device.log", {
        R"({"type":"job_start","session_id":"s1"})",
        "this is not json at all - the backend rejects it per-line",
        R"({"type":"kernel_event_batch","session_id":"s1","rows":[]})",
        R"({"type":"shutdown","session_id":"s1"})",
    });

    CaptureServer srv;
    gpufl::UploadOptions opts;
    opts.log_path        = tmp.string();
    opts.backend_url     = srv.base_url();
    opts.api_key         = "x";
    opts.report_progress = false;

    const auto result = gpufl::uploadLogs(opts);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(srv.snapshot().size(), 1u);
    EXPECT_EQ(srv.allLines().size(), 4u)
        << "All 4 lines (including the malformed one) ship verbatim.";

    fs::remove_all(tmp);
}

TEST(UploadLogs, EmptyBackendUrlReturnsFailure) {
    gpufl::UploadOptions opts;
    opts.log_path = "/tmp";
    opts.api_key  = "x";
    const auto result = gpufl::uploadLogs(opts);
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.warnings.empty());
}

TEST(UploadLogs, EmptyApiKeyReturnsFailure) {
    gpufl::UploadOptions opts;
    opts.log_path    = "/tmp";
    opts.backend_url = "http://example.com";
    const auto result = gpufl::uploadLogs(opts);
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.warnings.empty());
}

// ─────────────────────────────────────────────────────────────────────
// Session selection (default=latest, --session-id, --all-sessions)
// ─────────────────────────────────────────────────────────────────────

namespace {

std::string makeTwoSessions(const fs::path& root) {
    // v1.2 layout: each session in its own subdirectory under root.
    // Pre-v1.2 fixtures interleaved sessions in one flat file; that
    // was a worst case for per-line filtering. Per-session-dir makes
    // session isolation a property of the on-disk layout itself.
    fs::create_directories(root / "s_old");
    fs::create_directories(root / "s_new");
    writeLog(root / "s_old" / "device.log", {
        R"({"type":"job_start","session_id":"s_old","app":"x","pid":1,"ts_ns":100})",
        R"({"type":"kernel_event_batch","session_id":"s_old","batch_id":1,"rows":[[0,1,0,1,1,0,0,0]]})",
        R"({"type":"shutdown","session_id":"s_old","app":"x","pid":1,"ts_ns":200})",
    });
    writeLog(root / "s_new" / "device.log", {
        R"({"type":"job_start","session_id":"s_new","app":"x","pid":2,"ts_ns":1000})",
        R"({"type":"kernel_event_batch","session_id":"s_new","batch_id":1,"rows":[[0,1,0,1,1,0,0,0]]})",
        R"({"type":"shutdown","session_id":"s_new","app":"x","pid":2,"ts_ns":2000})",
    });
    return root.string();
}

}  // namespace

TEST(UploadLogs, DefaultSelectsLatestSessionOnly) {
    const fs::path tmp = fs::temp_directory_path() / "gpufl_upload_test_latest";
    fs::remove_all(tmp);
    const std::string log_path = makeTwoSessions(tmp);

    CaptureServer srv;
    gpufl::UploadOptions opts;
    opts.log_path        = log_path;
    opts.backend_url     = srv.base_url();
    opts.api_key         = "x";
    opts.report_progress = false;

    const auto result = gpufl::uploadLogs(opts);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.events_uploaded, 3u)
        << "Default mode should upload only the latest session (s_new), not s_old.";

    const auto events = srv.allEvents();
    ASSERT_EQ(events.size(), 3u);
    for (const auto& [type, sid] : events) {
        EXPECT_EQ(sid, "s_new")
            << "Expected only s_new events; got session " << sid;
    }
    // Header on every chunk must also be s_new.
    for (const auto& c : srv.snapshot()) {
        EXPECT_EQ(c.session_id, "s_new");
    }

    fs::remove_all(tmp);
}

TEST(UploadLogs, SessionIdFilterUploadsOnlyMatchingSession) {
    const fs::path tmp = fs::temp_directory_path() / "gpufl_upload_test_filter_select";
    fs::remove_all(tmp);
    const std::string log_path = makeTwoSessions(tmp);

    CaptureServer srv;
    gpufl::UploadOptions opts;
    opts.log_path          = log_path;
    opts.backend_url       = srv.base_url();
    opts.api_key           = "x";
    opts.session_id_filter = "s_old";
    opts.report_progress   = false;

    const auto result = gpufl::uploadLogs(opts);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.events_uploaded, 3u);

    for (const auto& c : srv.snapshot()) {
        EXPECT_EQ(c.session_id, "s_old");
    }

    fs::remove_all(tmp);
}

TEST(UploadLogs, SessionIdFilterUnknownReturnsFailure) {
    const fs::path tmp = fs::temp_directory_path() / "gpufl_upload_test_filter_miss";
    fs::remove_all(tmp);
    const std::string log_path = makeTwoSessions(tmp);

    CaptureServer srv;
    gpufl::UploadOptions opts;
    opts.log_path          = log_path;
    opts.backend_url       = srv.base_url();
    opts.api_key           = "x";
    opts.session_id_filter = "no_such_session";
    opts.report_progress   = false;

    const auto result = gpufl::uploadLogs(opts);
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.events_uploaded, 0u);
    ASSERT_FALSE(result.warnings.empty());
    EXPECT_NE(result.warnings.front().find("not found"), std::string::npos);

    fs::remove_all(tmp);
}

TEST(UploadLogs, AllSessionsUploadsEveryDistinctSession) {
    const fs::path tmp = fs::temp_directory_path() / "gpufl_upload_test_all";
    fs::remove_all(tmp);
    const std::string log_path = makeTwoSessions(tmp);

    CaptureServer srv;
    gpufl::UploadOptions opts;
    opts.log_path        = log_path;
    opts.backend_url     = srv.base_url();
    opts.api_key         = "x";
    opts.all_sessions    = true;
    opts.report_progress = false;

    const auto result = gpufl::uploadLogs(opts);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.events_uploaded, 6u)
        << "Expected 3 events from s_old + 3 from s_new.";

    // Per-session lifecycle ordering: each session's chunks form a
    // contiguous block (no interleaving) so the backend sees a clean
    // job_start → ... → shutdown sequence per session.
    const auto chunks = srv.snapshot();
    ASSERT_FALSE(chunks.empty());
    int s_old_first = -1, s_old_last = -1, s_new_first = -1, s_new_last = -1;
    for (int i = 0; i < static_cast<int>(chunks.size()); ++i) {
        if (chunks[i].session_id == "s_old") {
            if (s_old_first < 0) s_old_first = i;
            s_old_last = i;
        }
        if (chunks[i].session_id == "s_new") {
            if (s_new_first < 0) s_new_first = i;
            s_new_last = i;
        }
    }
    const bool no_interleave =
        (s_old_last < s_new_first) || (s_new_last < s_old_first);
    EXPECT_TRUE(no_interleave)
        << "Sessions should not interleave at the chunk level - each "
        << "session's chunks must be contiguous.";

    fs::remove_all(tmp);
}

TEST(UploadLogs, AllSessionsSilentlySkipsAlreadyCompleted) {
    const fs::path tmp = fs::temp_directory_path() / "gpufl_upload_test_all_skip";
    fs::remove_all(tmp);
    const std::string log_path = makeTwoSessions(tmp);

    CaptureServer srv;
    gpufl::UploadOptions opts;
    opts.log_path        = log_path;
    opts.backend_url     = srv.base_url();
    opts.api_key         = "x";
    opts.all_sessions    = true;
    opts.report_progress = false;

    const auto r1 = gpufl::uploadLogs(opts);
    ASSERT_TRUE(r1.success);
    EXPECT_EQ(r1.events_uploaded, 6u);
    const auto chunks_first = srv.snapshot().size();

    const auto r2 = gpufl::uploadLogs(opts);
    EXPECT_TRUE(r2.success);
    EXPECT_EQ(r2.events_uploaded, 0u);
    EXPECT_EQ(srv.snapshot().size(), chunks_first)
        << "Server should see no additional chunks.";

    fs::remove_all(tmp);
}

TEST(UploadLogs, SessionIdFilterAndAllSessionsRejected) {
    gpufl::UploadOptions opts;
    opts.log_path          = "/tmp";
    opts.backend_url       = "http://127.0.0.1:1";
    opts.api_key           = "x";
    opts.session_id_filter = "s_old";
    opts.all_sessions      = true;
    const auto result = gpufl::uploadLogs(opts);
    EXPECT_FALSE(result.success);
    ASSERT_FALSE(result.warnings.empty());
    EXPECT_NE(result.warnings.front().find("mutually exclusive"),
              std::string::npos);
}

// ─────────────────────────────────────────────────────────────────────
// Lifecycle ordering
// ─────────────────────────────────────────────────────────────────────

TEST(UploadLogs, JobStartFirstShutdownLast) {
    const fs::path tmp = fs::temp_directory_path() / "gpufl_upload_test_order";
    fs::remove_all(tmp);
    fs::create_directories(tmp);

    fs::create_directories(tmp / "s");
    writeLog(tmp / "s" / "device.1.log.gz", {
        R"({"type":"job_start","session_id":"s","ts_ns":1})",
        R"({"type":"kernel_event_batch","session_id":"s","batch_id":1,"rows":[]})",
    }, /*gzip=*/true);
    writeLog(tmp / "s" / "device.log", {
        R"({"type":"kernel_event_batch","session_id":"s","batch_id":2,"rows":[]})",
        R"({"type":"shutdown","session_id":"s","ts_ns":99})",
    });

    CaptureServer srv;
    gpufl::UploadOptions opts;
    opts.log_path        = tmp.string();
    opts.backend_url     = srv.base_url();
    opts.api_key         = "x";
    opts.report_progress = false;

    const auto result = gpufl::uploadLogs(opts);
    ASSERT_TRUE(result.success);

    const auto events = srv.allEvents();
    ASSERT_EQ(events.size(), 4u);
    EXPECT_EQ(events.front().first, "job_start")
        << "job_start must be the first NDJSON line shipped - it heads "
        << "the oldest rotated file, which is POSTed first.";
    EXPECT_EQ(events.back().first, "shutdown")
        << "shutdown must be the last NDJSON line shipped - it tails "
        << "the active file, which is POSTed last.";

    fs::remove_all(tmp);
}

// ─────────────────────────────────────────────────────────────────────
// File-unit upload - one POST per file regardless of line count.
// ─────────────────────────────────────────────────────────────────────

TEST(UploadLogs, ManyEventsShipInOneRequestPerFile) {
    // U1 regression guard: a file with far more lines than the old
    // 5000-line chunk cap still ships in exactly ONE request - the
    // rotator's file size is the upload unit now.
    const fs::path tmp = fs::temp_directory_path() / "gpufl_upload_test_chunks";
    fs::remove_all(tmp);
    fs::create_directories(tmp);

    constexpr int kLines = 5500;
    std::vector<std::string> lines;
    lines.reserve(kLines + 2);
    lines.push_back(R"({"type":"job_start","session_id":"big","ts_ns":1})");
    for (int i = 0; i < kLines; ++i) {
        lines.push_back(
            std::string(R"({"type":"kernel_event_batch","session_id":"big","batch_id":)")
            + std::to_string(i) + R"(,"rows":[]})");
    }
    lines.push_back(R"({"type":"shutdown","session_id":"big","ts_ns":99})");
    fs::create_directories(tmp / "big");
    writeLog(tmp / "big" / "device.log", lines);

    CaptureServer srv;
    gpufl::UploadOptions opts;
    opts.log_path        = tmp.string();
    opts.backend_url     = srv.base_url();
    opts.api_key         = "x";
    opts.report_progress = false;

    const auto result = gpufl::uploadLogs(opts);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.events_uploaded, static_cast<std::size_t>(kLines + 2));

    const auto chunks = srv.snapshot();
    ASSERT_EQ(chunks.size(), 1u)
        << "One file must ship as one request - no client-side chunking.";
    EXPECT_EQ(chunks.front().lines.size(), static_cast<std::size_t>(kLines + 2));
    EXPECT_EQ(chunks.back().events.back().first, "shutdown");

    fs::remove_all(tmp);
}

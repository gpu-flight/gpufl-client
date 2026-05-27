// End-to-end tests for gpufl::uploadLogs.
//
// Each test writes a synthetic NDJSON log directory in a tmp path,
// spins up an embedded httplib::Server to capture POSTs, runs
// uploadLogs against the captured server, and asserts on what was
// captured (event types, routing paths, auth headers, ordering).
//
// These replace the URL-routing + version-header coverage that used
// to live in test_http_log_sink.cpp before HttpLogSink was removed.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#include <httplib.h>
#include <zlib.h>

#include "gpufl/upload/upload_logs.hpp"
#include "gpufl/core/version.hpp"

namespace fs = std::filesystem;

namespace {

/// Unescape a JSON string literal (used to pull the inner NDJSON line
/// back out of the EventWrapper.data field). Only handles the escape
/// sequences gpufl::json::escape() produces — \", \\, \n, \r, \t.
std::string unescapeJsonString(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            switch (s[i + 1]) {
                case '"':  out += '"';  i++; break;
                case '\\': out += '\\'; i++; break;
                case '/':  out += '/';  i++; break;
                case 'n':  out += '\n'; i++; break;
                case 'r':  out += '\r'; i++; break;
                case 't':  out += '\t'; i++; break;
                default:   out += s[i]; break;
            }
        } else {
            out += s[i];
        }
    }
    return out;
}

/// Pull out the `"data":"<...escaped NDJSON...>"` field from an
/// EventWrapper body and return its unescaped contents. Returns the
/// original body unchanged if the field can't be found — defensive
/// fallback so tests don't blow up if a future test exercises an
/// already-unwrapped path.
std::string extractInnerData(const std::string& wrapped_body) {
    static const std::string kKey = "\"data\":\"";
    const auto pos = wrapped_body.find(kKey);
    if (pos == std::string::npos) return wrapped_body;
    const std::size_t start = pos + kKey.size();
    // Find the terminating quote, respecting backslash escapes.
    std::size_t i = start;
    while (i < wrapped_body.size()) {
        if (wrapped_body[i] == '\\' && i + 1 < wrapped_body.size()) {
            i += 2;
            continue;
        }
        if (wrapped_body[i] == '"') break;
        ++i;
    }
    if (i >= wrapped_body.size()) return wrapped_body;
    return unescapeJsonString(wrapped_body.substr(start, i - start));
}

/// Localhost capture server. Stores every POSTed body keyed by the
/// event-type path component, plus the request headers, so individual
/// tests can verify routing, auth, and ordering.
///
/// Each Capture exposes both the raw `body` (the full EventWrapper
/// envelope JSON as POSTed) and `inner_data` (the unwrapped inner
/// NDJSON line). Tests should generally assert on inner_data — that's
/// the per-event payload semantics.
class CaptureServer {
   public:
    struct Capture {
        std::string event_type;
        std::string body;        // raw EventWrapper envelope
        std::string inner_data;  // unwrapped inner NDJSON line
        std::string user_agent;
        std::string authorization;
        std::string client_version;
        std::string wire_version;
    };

    /// Construct with an optional status override (default 200). Setting
    /// `force_status = 401` lets tests simulate auth failures; 500 lets
    /// them simulate transient errors.
    explicit CaptureServer(int force_status = 200) : force_status_(force_status) {
        server_.Post(R"(/api/v1/events/([a-z_]+))",
            [this](const httplib::Request& req, httplib::Response& res) {
                std::lock_guard<std::mutex> lk(mu_);
                Capture c;
                c.event_type     = req.matches.size() > 1 ? req.matches[1].str() : "";
                c.body           = req.body;
                c.inner_data     = extractInnerData(req.body);
                c.user_agent     = req.get_header_value("User-Agent");
                c.authorization  = req.get_header_value("Authorization");
                c.client_version = req.get_header_value("X-GpuFlight-Client-Version");
                c.wire_version   = req.get_header_value("X-GpuFlight-Wire-Version");
                captured_.push_back(std::move(c));
                res.status = force_status_;
                res.set_content("{\"ok\":true}", "application/json");
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

    std::vector<Capture> snapshot() {
        std::lock_guard<std::mutex> lk(mu_);
        return captured_;
    }

   private:
    httplib::Server      server_;
    std::thread          thread_;
    int                  port_{0};
    int                  force_status_;
    std::mutex           mu_;
    std::vector<Capture> captured_;
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

/// Build a minimal log dir with one job_start, one kernel batch, one
/// shutdown — spread across rotated .gz + active .log so the lifecycle
/// ordering is non-trivial.
std::string makeMinimalSession(const fs::path& root, const std::string& prefix = "test") {
    fs::create_directories(root);
    // device channel — rotated (older) + active (newer)
    writeLog(root / (prefix + ".device.1.log.gz"), {
        R"({"type":"job_start","session_id":"s1","app":"test","pid":1,"ts_ns":100})",
        R"({"type":"kernel_event_batch","session_id":"s1","batch_id":1,"rows":[[0,1,0,1000,101,0,32,1]]})",
    }, /*gzip=*/true);
    writeLog(root / (prefix + ".device.log"), {
        R"({"type":"kernel_event_batch","session_id":"s1","batch_id":2,"rows":[[2000,2,0,1500,102,0,64,1]]})",
        R"({"type":"shutdown","session_id":"s1","app":"test","pid":1,"ts_ns":9999})",
    });
    writeLog(root / (prefix + ".scope.log"), {
        R"({"type":"scope_event_batch","session_id":"s1","batch_id":1,"rows":[[0,1,1,0,0],[5000,1,1,1,0]]})",
    });
    return (root / prefix).string();  // returns the log_path prefix
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

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.events_uploaded, 5u)
        << "Expected 5 NDJSON events to be POSTed (job_start, "
        << "two kernel batches, one scope batch, one shutdown).";
    EXPECT_EQ(result.files_processed, 3u);  // device.1.log.gz + device.log + scope.log
    EXPECT_TRUE(result.warnings.empty()) << "Unexpected warnings on happy path";

    const auto caps = srv.snapshot();
    ASSERT_EQ(caps.size(), 5u);

    // job_start must be POSTed first (so the backend creates the session
    // row before any event tries to reference its session_id).
    EXPECT_EQ(caps.front().event_type, "job_start");
    // shutdown must be POSTed last (deferred to end of upload).
    EXPECT_EQ(caps.back().event_type, "shutdown");

    fs::remove_all(tmp);
}

// Regression guard for the silent-data-loss bug shipped in the first
// uploadLogs() draft: the backend's EventIngestionController binds
// every request body to an `EventWrapper(data, agentSendingTime,
// hostname, ipAddr)` record. Posting the bare NDJSON line leaves
// every wrapper field null, the inner readValue(null, ...) throws,
// the exception is swallowed, and the controller returns 200 anyway
// — so the client reports success while the dashboard sees nothing.
// This test asserts the wrapper is present on every POST.
TEST(UploadLogs, EveryPostIsWrappedInEventWrapper) {
    const fs::path tmp = fs::temp_directory_path() / "gpufl_upload_test_wrapper";
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

    const auto caps = srv.snapshot();
    ASSERT_FALSE(caps.empty());
    for (const auto& c : caps) {
        // The raw POST body MUST be an EventWrapper envelope, not the
        // bare NDJSON event. Check for each wrapper field by name.
        EXPECT_NE(c.body.find("\"data\":\""), std::string::npos)
            << "Body missing wrapper field `data`: " << c.body;
        EXPECT_NE(c.body.find("\"agentSendingTime\":"), std::string::npos)
            << "Body missing wrapper field `agentSendingTime`: " << c.body;
        EXPECT_NE(c.body.find("\"hostname\":"), std::string::npos)
            << "Body missing wrapper field `hostname`: " << c.body;
        EXPECT_NE(c.body.find("\"ipAddr\":"), std::string::npos)
            << "Body missing wrapper field `ipAddr`: " << c.body;

        // The unwrapped inner content must be the original NDJSON event.
        EXPECT_NE(c.inner_data.find("\"type\":\""), std::string::npos)
            << "Inner data should be NDJSON event: " << c.inner_data;
        EXPECT_NE(c.inner_data.find("\"session_id\":\""), std::string::npos)
            << "Inner data should carry session_id: " << c.inner_data;
    }

    fs::remove_all(tmp);
}

TEST(UploadLogs, AuthHeaderAndVersionHeadersPresentOnEveryPost) {
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

    const auto caps = srv.snapshot();
    ASSERT_FALSE(caps.empty());
    for (const auto& c : caps) {
        EXPECT_EQ(c.authorization,  "Bearer secret_token");
        EXPECT_EQ(c.client_version, gpufl::kClientVersion);
        EXPECT_EQ(c.wire_version,   gpufl::kWireVersion);
        EXPECT_FALSE(c.user_agent.empty());
    }

    fs::remove_all(tmp);
}

// ─────────────────────────────────────────────────────────────────────
// Cursor file
// ─────────────────────────────────────────────────────────────────────

TEST(UploadLogs, CursorRefusesRepeatUploadOfSameSession) {
    // Cursor v2 tracks completed session_ids — re-running the default
    // (single-session) upload on a session that already shipped should
    // refuse with success=false and a warning suggesting --force.
    const fs::path tmp = fs::temp_directory_path() / "gpufl_upload_test_cursor_refuse";
    fs::remove_all(tmp);
    const std::string log_path = makeMinimalSession(tmp);

    CaptureServer srv;
    gpufl::UploadOptions opts;
    opts.log_path        = log_path;
    opts.backend_url     = srv.base_url();
    opts.api_key         = "x";
    opts.report_progress = false;

    // First run: uploads everything.
    const auto r1 = gpufl::uploadLogs(opts);
    ASSERT_TRUE(r1.success);
    const auto caps_after_first = srv.snapshot().size();
    EXPECT_GE(caps_after_first, 1u);
    EXPECT_TRUE(fs::exists(tmp / ".gpufl-upload-cursor.json"));

    // Second run: same session_id is in cursor.completed_sessions, so
    // the default mode refuses and emits a warning suggesting --force.
    const auto r2 = gpufl::uploadLogs(opts);
    EXPECT_FALSE(r2.success);
    EXPECT_EQ(r2.events_uploaded, 0u)
        << "No events should be uploaded when refusing — backend should "
        << "see no additional POSTs.";
    ASSERT_FALSE(r2.warnings.empty());
    EXPECT_NE(r2.warnings.front().find("already uploaded"), std::string::npos);
    EXPECT_NE(r2.warnings.front().find("force"), std::string::npos);
    EXPECT_EQ(srv.snapshot().size(), caps_after_first)
        << "Server should see no further POSTs on refusal.";

    fs::remove_all(tmp);
}

TEST(UploadLogs, ForceOverridesCursorRefusal) {
    // --force re-uploads even completed sessions.
    const fs::path tmp = fs::temp_directory_path() / "gpufl_upload_test_force";
    fs::remove_all(tmp);
    const std::string log_path = makeMinimalSession(tmp);

    CaptureServer srv;
    gpufl::UploadOptions opts;
    opts.log_path        = log_path;
    opts.backend_url     = srv.base_url();
    opts.api_key         = "x";
    opts.report_progress = false;

    // First run completes; cursor now lists session s1.
    const auto r1 = gpufl::uploadLogs(opts);
    ASSERT_TRUE(r1.success);
    const auto first_count = srv.snapshot().size();

    // Re-run with force=true → events are uploaded again.
    opts.force = true;
    const auto r2 = gpufl::uploadLogs(opts);
    EXPECT_TRUE(r2.success);
    EXPECT_EQ(r2.events_uploaded, r1.events_uploaded)
        << "force=true should re-upload every event from the session.";
    EXPECT_EQ(srv.snapshot().size(), first_count * 2)
        << "Server should see double the captures after a forced re-upload.";

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
    opts.max_retries     = 0;  // don't retry — we expect immediate abort
    opts.report_progress = false;

    const auto result = gpufl::uploadLogs(opts);

    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.warnings.empty());
    // First POST returns 401 → abort. Subsequent events not attempted.
    // The server should see exactly one capture.
    EXPECT_EQ(srv.snapshot().size(), 1u)
        << "Auth failure should short-circuit — no further POSTs expected";

    fs::remove_all(tmp);
}

TEST(UploadLogs, MissingLogDirReturnsFailureNotThrow) {
    gpufl::UploadOptions opts;
    opts.log_path    = "/nonexistent/path/that/should/not/exist";
    opts.backend_url = "http://127.0.0.1:1";  // arbitrary, won't be reached
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
    opts.log_path        = (tmp / "test").string();
    opts.backend_url     = srv.base_url();
    opts.api_key         = "x";
    opts.report_progress = false;

    const auto result = gpufl::uploadLogs(opts);
    EXPECT_TRUE(result.success);  // nothing to do is success, not failure
    EXPECT_EQ(result.events_uploaded, 0u);
    EXPECT_EQ(result.files_processed, 0u);
    EXPECT_TRUE(srv.snapshot().empty());

    fs::remove_all(tmp);
}

TEST(UploadLogs, MalformedNdjsonLineSkippedWithWarning) {
    const fs::path tmp = fs::temp_directory_path() / "gpufl_upload_test_bad_line";
    fs::remove_all(tmp);
    fs::create_directories(tmp);

    writeLog(tmp / "test.device.log", {
        R"({"type":"job_start","session_id":"s1"})",
        "this is not json at all — should be skipped with a warning",
        R"({"type":"kernel_event_batch","session_id":"s1","rows":[]})",
        R"({"type":"shutdown","session_id":"s1"})",
    });

    CaptureServer srv;
    gpufl::UploadOptions opts;
    opts.log_path        = (tmp / "test").string();
    opts.backend_url     = srv.base_url();
    opts.api_key         = "x";
    opts.report_progress = false;

    const auto result = gpufl::uploadLogs(opts);
    EXPECT_TRUE(result.success);  // bad lines are skipped, upload continues
    EXPECT_EQ(result.events_uploaded, 3u);  // 3 good lines uploaded
    EXPECT_FALSE(result.warnings.empty()) << "Should warn about the bad line";

    fs::remove_all(tmp);
}

TEST(UploadLogs, EmptyBackendUrlReturnsFailure) {
    gpufl::UploadOptions opts;
    opts.log_path = "/tmp";
    opts.api_key  = "x";
    // backend_url intentionally empty

    const auto result = gpufl::uploadLogs(opts);
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.warnings.empty());
}

TEST(UploadLogs, EmptyApiKeyReturnsFailure) {
    gpufl::UploadOptions opts;
    opts.log_path    = "/tmp";
    opts.backend_url = "http://example.com";
    // api_key intentionally empty

    const auto result = gpufl::uploadLogs(opts);
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.warnings.empty());
}

// ─────────────────────────────────────────────────────────────────────
// Lifecycle ordering
// ─────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────
// Session selection (default=latest, --session-id, --all-sessions)
// ─────────────────────────────────────────────────────────────────────

namespace {

/// Build a directory with two independent sessions sharing the same
/// log_path prefix. `s_old` has a lower job_start.ts_ns than `s_new`,
/// so the default "latest" selection should pick `s_new`.
std::string makeTwoSessions(const fs::path& root,
                            const std::string& prefix = "multi") {
    fs::create_directories(root);
    writeLog(root / (prefix + ".device.log"), {
        R"({"type":"job_start","session_id":"s_old","app":"x","pid":1,"ts_ns":100})",
        R"({"type":"kernel_event_batch","session_id":"s_old","batch_id":1,"rows":[[0,1,0,1,1,0,0,0]]})",
        R"({"type":"shutdown","session_id":"s_old","app":"x","pid":1,"ts_ns":200})",
        R"({"type":"job_start","session_id":"s_new","app":"x","pid":2,"ts_ns":1000})",
        R"({"type":"kernel_event_batch","session_id":"s_new","batch_id":1,"rows":[[0,1,0,1,1,0,0,0]]})",
        R"({"type":"shutdown","session_id":"s_new","app":"x","pid":2,"ts_ns":2000})",
    });
    return (root / prefix).string();
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
    // 3 events: s_new's job_start, kernel_event_batch, shutdown.
    // s_old must NOT be uploaded.
    EXPECT_EQ(result.events_uploaded, 3u)
        << "Default mode should upload only the latest session (s_new), "
        << "not s_old.";

    const auto caps = srv.snapshot();
    ASSERT_EQ(caps.size(), 3u);
    for (const auto& c : caps) {
        // Every uploaded event should carry session_id "s_new" in the
        // unwrapped inner NDJSON line.
        EXPECT_NE(c.inner_data.find("\"session_id\":\"s_new\""), std::string::npos)
            << "Expected only s_new events; got inner: " << c.inner_data;
    }

    fs::remove_all(tmp);
}

TEST(UploadLogs, SessionIdFilterUploadsOnlyMatchingSession) {
    const fs::path tmp = fs::temp_directory_path() / "gpufl_upload_test_filter";
    fs::remove_all(tmp);
    const std::string log_path = makeTwoSessions(tmp);

    CaptureServer srv;
    gpufl::UploadOptions opts;
    opts.log_path          = log_path;
    opts.backend_url       = srv.base_url();
    opts.api_key           = "x";
    opts.session_id_filter = "s_old";  // pick the older one explicitly
    opts.report_progress   = false;

    const auto result = gpufl::uploadLogs(opts);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.events_uploaded, 3u);

    const auto caps = srv.snapshot();
    ASSERT_EQ(caps.size(), 3u);
    for (const auto& c : caps) {
        EXPECT_NE(c.inner_data.find("\"session_id\":\"s_old\""), std::string::npos);
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

    // Per-session lifecycle ordering: s_old should appear as a complete
    // block (job_start→...→shutdown) before s_new starts. We don't
    // require any specific ordering between the two sessions, just
    // that each session's events form a contiguous block.
    const auto caps = srv.snapshot();
    ASSERT_EQ(caps.size(), 6u);
    // Find each session's first and last event index. They should not
    // interleave.
    int s_old_first = -1, s_old_last = -1, s_new_first = -1, s_new_last = -1;
    for (int i = 0; i < static_cast<int>(caps.size()); ++i) {
        const bool is_old = caps[i].inner_data.find("s_old") != std::string::npos;
        const bool is_new = caps[i].inner_data.find("s_new") != std::string::npos;
        if (is_old) {
            if (s_old_first < 0) s_old_first = i;
            s_old_last = i;
        }
        if (is_new) {
            if (s_new_first < 0) s_new_first = i;
            s_new_last = i;
        }
    }
    const bool no_interleave =
        (s_old_last < s_new_first) || (s_new_last < s_old_first);
    EXPECT_TRUE(no_interleave)
        << "Sessions should not interleave — per-session lifecycle "
        << "ordering requires each session's events to form a "
        << "contiguous block of POSTs.";

    // Each session's block must start with job_start and end with shutdown.
    EXPECT_EQ(caps[s_old_first].event_type, "job_start");
    EXPECT_EQ(caps[s_old_last].event_type,  "shutdown");
    EXPECT_EQ(caps[s_new_first].event_type, "job_start");
    EXPECT_EQ(caps[s_new_last].event_type,  "shutdown");

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

    // First run: ships both sessions.
    const auto r1 = gpufl::uploadLogs(opts);
    ASSERT_TRUE(r1.success);
    EXPECT_EQ(r1.events_uploaded, 6u);

    // Second run with --all-sessions: both sessions are in the cursor.
    // Should silently skip both, success=true, zero new events.
    const auto r2 = gpufl::uploadLogs(opts);
    EXPECT_TRUE(r2.success)
        << "All-sessions mode with everything already done is a success "
        << "no-op, not a failure.";
    EXPECT_EQ(r2.events_uploaded, 0u);
    EXPECT_EQ(srv.snapshot().size(), 6u)
        << "Server should see no additional POSTs.";

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

    // Put shutdown in the active file (last in file order). job_start
    // in the rotated .gz (oldest). Mix in some other events to ensure
    // they don't accidentally get reordered.
    writeLog(tmp / "test.device.1.log.gz", {
        R"({"type":"job_start","session_id":"s","ts_ns":1})",
        R"({"type":"kernel_event_batch","session_id":"s","batch_id":1,"rows":[]})",
    }, /*gzip=*/true);
    writeLog(tmp / "test.device.log", {
        R"({"type":"kernel_event_batch","session_id":"s","batch_id":2,"rows":[]})",
        R"({"type":"shutdown","session_id":"s","ts_ns":99})",
    });

    CaptureServer srv;
    gpufl::UploadOptions opts;
    opts.log_path        = (tmp / "test").string();
    opts.backend_url     = srv.base_url();
    opts.api_key         = "x";
    opts.report_progress = false;

    const auto result = gpufl::uploadLogs(opts);
    ASSERT_TRUE(result.success);

    const auto caps = srv.snapshot();
    ASSERT_EQ(caps.size(), 4u);
    EXPECT_EQ(caps.front().event_type, "job_start")
        << "job_start must be the very first POST (so the backend "
        << "creates the session row before downstream events arrive).";
    EXPECT_EQ(caps.back().event_type, "shutdown")
        << "shutdown must be the very last POST (deferred to end of "
        << "upload regardless of which file it lived in).";

    fs::remove_all(tmp);
}

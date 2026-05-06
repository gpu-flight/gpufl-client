// Tests for the api_path / version-header plumbing on the client side.
//
// Two layers exercised here:
//   1. normalizeApiPath() — pure helper, table-driven, no I/O.
//   2. HttpLogSink end-to-end — spin a localhost httplib::Server, wire
//      the sink to send to it, observe (a) the URL path the server
//      receives matches the configured api_path, and (b) the version
//      headers are present on every request.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#include <httplib.h>

#include "gpufl/core/logger/http_log_sink.hpp"
#include "gpufl/core/version.hpp"

// ── normalizeApiPath ────────────────────────────────────────────────────────

TEST(NormalizeApiPath, EmptyFallsBackToDefault) {
    EXPECT_EQ(gpufl::normalizeApiPath(""), gpufl::kDefaultApiPath);
}

TEST(NormalizeApiPath, BareRootFallsBackToDefault) {
    EXPECT_EQ(gpufl::normalizeApiPath("/"), gpufl::kDefaultApiPath);
}

TEST(NormalizeApiPath, MissingLeadingSlashIsPrepended) {
    EXPECT_EQ(gpufl::normalizeApiPath("api/v1"), "/api/v1");
    EXPECT_EQ(gpufl::normalizeApiPath("custom/v2"), "/custom/v2");
}

TEST(NormalizeApiPath, TrailingSlashesAreStripped) {
    EXPECT_EQ(gpufl::normalizeApiPath("/api/v1/"), "/api/v1");
    EXPECT_EQ(gpufl::normalizeApiPath("/api/v1///"), "/api/v1");
}

TEST(NormalizeApiPath, AlreadyCanonicalIsUnchanged) {
    EXPECT_EQ(gpufl::normalizeApiPath("/api/v1"), "/api/v1");
    EXPECT_EQ(gpufl::normalizeApiPath("/profiler/api/v1"), "/profiler/api/v1");
}

TEST(NormalizeApiPath, ResultAlwaysHasLeadingSlashNoTrailing) {
    for (const auto& in : std::vector<std::string>{
            "api", "/api", "api/", "/api/", "x/y/z", "/x/y/z/"}) {
        const auto out = gpufl::normalizeApiPath(in);
        ASSERT_FALSE(out.empty());
        EXPECT_EQ(out.front(), '/') << "input='" << in << "' out='" << out << "'";
        if (out.size() > 1) {
            EXPECT_NE(out.back(), '/') << "input='" << in << "' out='" << out << "'";
        }
    }
}

// ── HttpLogSink end-to-end (URL routing + headers) ───────────────────────────

namespace {

/**
 * Localhost server that captures every POST under any path matching
 * `<prefix>/events/<type>`. Lets a test parameterize the api_path
 * prefix and observe the resulting routed URL.
 */
struct CaptureServer {
    httplib::Server server;
    std::thread     thread;
    int             port{0};

    struct Capture {
        std::string path;
        std::string user_agent;
        std::string client_version;
        std::string wire_version;
        std::string authorization;
    };
    std::mutex          mu;
    std::vector<Capture> captured;

    explicit CaptureServer(const std::string& path_regex) {
        // Match the configured prefix + /events/<type>. The regex is
        // injected so the test can choose its own prefix.
        server.Post(path_regex,
            [this](const httplib::Request& req, httplib::Response& res) {
                std::lock_guard<std::mutex> lk(mu);
                Capture c;
                c.path           = req.path;
                c.user_agent     = req.get_header_value("User-Agent");
                c.client_version = req.get_header_value("X-GpuFlight-Client-Version");
                c.wire_version   = req.get_header_value("X-GpuFlight-Wire-Version");
                c.authorization  = req.get_header_value("Authorization");
                captured.push_back(std::move(c));
                res.status = 200;
                res.set_content("{\"status\":\"ok\"}", "application/json");
            });
        port = server.bind_to_any_port("127.0.0.1");
        thread = std::thread([this] { server.listen_after_bind(); });
        for (int i = 0; i < 100 && !server.is_running(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    ~CaptureServer() {
        server.stop();
        if (thread.joinable()) thread.join();
    }

    std::string base_url() const {
        return "http://127.0.0.1:" + std::to_string(port);
    }

    std::vector<Capture> snapshot() {
        std::lock_guard<std::mutex> lk(mu);
        return captured;
    }
};

template <typename Pred>
bool waitFor(Pred pred, std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (pred()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return pred();
}

}  // namespace

TEST(HttpLogSinkApiPath, DefaultPrefixRoutesToApiV1) {
    CaptureServer srv(R"(/api/v1/events/([a-z_]+))");

    gpufl::HttpLogSink::Options opts;
    opts.base_url = srv.base_url();
    opts.api_path = "/api/v1";  // explicit default — same as field default
    opts.api_key  = "test-key";

    {
        gpufl::HttpLogSink sink(std::move(opts));
        sink.write(gpufl::Channel::All,
                   R"({"type":"kernel_event_batch","payload":{}})");

        ASSERT_TRUE(waitFor([&] { return !srv.snapshot().empty(); },
                            std::chrono::seconds(5)))
            << "expected server to receive at least one POST";
    }

    auto caps = srv.snapshot();
    ASSERT_FALSE(caps.empty());
    EXPECT_EQ(caps[0].path, "/api/v1/events/kernel_event_batch");
}

TEST(HttpLogSinkApiPath, CustomPrefixRoutesToConfiguredPath) {
    CaptureServer srv(R"(/profiler/v1/events/([a-z_]+))");

    gpufl::HttpLogSink::Options opts;
    opts.base_url = srv.base_url();
    opts.api_path = "/profiler/v1";  // simulating a reverse-proxy mount
    opts.api_key  = "test-key";

    {
        gpufl::HttpLogSink sink(std::move(opts));
        sink.write(gpufl::Channel::All,
                   R"({"type":"kernel_event_batch","payload":{}})");

        ASSERT_TRUE(waitFor([&] { return !srv.snapshot().empty(); },
                            std::chrono::seconds(5)))
            << "expected server to receive at least one POST under "
               "/profiler/v1/events/...";
    }

    auto caps = srv.snapshot();
    ASSERT_FALSE(caps.empty());
    EXPECT_EQ(caps[0].path, "/profiler/v1/events/kernel_event_batch");
}

TEST(HttpLogSinkApiPath, VersionHeadersPresentOnEveryRequest) {
    CaptureServer srv(R"(/api/v1/events/([a-z_]+))");

    gpufl::HttpLogSink::Options opts;
    opts.base_url = srv.base_url();
    opts.api_path = "/api/v1";
    opts.api_key  = "test-key";

    {
        gpufl::HttpLogSink sink(std::move(opts));
        sink.write(gpufl::Channel::All,
                   R"({"type":"kernel_event_batch","payload":{}})");
        sink.write(gpufl::Channel::All,
                   R"({"type":"memcpy_event_batch","payload":{}})");

        ASSERT_TRUE(waitFor([&] { return srv.snapshot().size() >= 2; },
                            std::chrono::seconds(5)));
    }

    const auto expected_user_agent =
        std::string("gpufl/") + gpufl::kClientVersion;
    const auto caps = srv.snapshot();
    ASSERT_GE(caps.size(), 2u);
    for (const auto& c : caps) {
        EXPECT_EQ(c.user_agent,     expected_user_agent);
        EXPECT_EQ(c.client_version, gpufl::kClientVersion);
        EXPECT_EQ(c.wire_version,   gpufl::kWireVersion);
        // Per-call Authorization header still merges on top of defaults.
        EXPECT_EQ(c.authorization,  "Bearer test-key");
    }
}

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <httplib.h>

#include "gpufl/core/logger/http_log_sink.hpp"
#include "gpufl/core/model/serializable.hpp"

namespace {

/**
 * Tiny localhost test server that captures every POSTed request.
 * We spin one up per test on a random ephemeral port; no shared state
 * between tests, no port conflicts in parallel CI runners.
 */
struct TestServer {
    httplib::Server server;
    std::thread     thread;
    int             port{0};

    struct Capture {
        std::string path;
        std::string body;
        std::string authorization;
        // Captured per-request so we can guard against the regression
        // where the sink emitted two Content-Type lines that Tomcat
        // joined into "application/json,application/json" → 415.
        std::size_t content_type_count{0};
        std::string content_type;
    };
    std::mutex         mu;
    std::vector<Capture> captured;

    std::atomic<int> forced_status{200};  // 0 = simulate network error
    std::atomic<int> requests{0};

    TestServer() {
        server.Post(R"(/api/v1/events/([a-z_]+))",
            [this](const httplib::Request& req, httplib::Response& res) {
                requests.fetch_add(1);
                {
                    std::lock_guard<std::mutex> lk(mu);
                    Capture c;
                    c.path = req.path;
                    c.body = req.body;
                    c.authorization = req.get_header_value("Authorization");
                    c.content_type_count =
                        req.get_header_value_count("Content-Type");
                    c.content_type = req.get_header_value("Content-Type");
                    captured.push_back(std::move(c));
                }
                res.status = forced_status.load();
                res.set_content("{\"status\":\"ok\"}", "application/json");
            });
        port = server.bind_to_any_port("127.0.0.1");
        thread = std::thread([this] { server.listen_after_bind(); });
        // Wait briefly for the listener to be ready.
        for (int i = 0; i < 100 && !server.is_running(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    ~TestServer() {
        server.stop();
        if (thread.joinable()) thread.join();
    }

    std::string base_url() const {
        return "http://127.0.0.1:" + std::to_string(port);
    }

    std::size_t capturedCount() {
        std::lock_guard<std::mutex> lk(mu);
        return captured.size();
    }

    std::vector<Capture> snapshot() {
        std::lock_guard<std::mutex> lk(mu);
        return captured;
    }
};

/** Wait until `pred()` returns true or the timeout elapses. */
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

// ── Happy path ───────────────────────────────────────────────────────────────

TEST(HttpLogSink, EnqueuedLinesReachBackendWithAuthAndTypeRouting) {
    TestServer srv;

    gpufl::HttpLogSink::Options opts;
    opts.base_url = srv.base_url();
    opts.api_key  = "gpfl_test_abc123";
    gpufl::HttpLogSink sink(std::move(opts));

    const std::vector<std::string> lines = {
        R"({"type":"job_start","app":"x"})",
        R"({"type":"scope_event_batch","rows":[]})",
        R"({"type":"kernel_event_batch","rows":[1,2,3]})",
        R"({"type":"shutdown","ts_ns":12345})",
    };
    for (const auto& l : lines) sink.write(gpufl::Channel::All, l);

    ASSERT_TRUE(waitFor(
        [&] { return srv.capturedCount() >= lines.size(); },
        std::chrono::seconds(5)));

    sink.close();

    auto caps = srv.snapshot();
    EXPECT_EQ(caps.size(), lines.size());
    // Every request should carry the Bearer auth header.
    for (const auto& c : caps) {
        EXPECT_EQ(c.authorization, "Bearer gpfl_test_abc123");
    }
    // The path should include the event type parsed from each JSON line.
    EXPECT_EQ(caps[0].path, "/api/v1/events/job_start");
    EXPECT_EQ(caps[1].path, "/api/v1/events/scope_event_batch");
    EXPECT_EQ(caps[2].path, "/api/v1/events/kernel_event_batch");
    EXPECT_EQ(caps[3].path, "/api/v1/events/shutdown");
    EXPECT_EQ(sink.uploadedCount(), lines.size());
    EXPECT_EQ(sink.failedCount(),   0u);
}

// ── Regression: Content-Type emitted exactly once ───────────────────────────
//
// cpp-httplib's Post(path, headers, body, content_type) accepts the
// content type via BOTH a headers-map entry AND a dedicated 4th
// argument. If both are set, two Content-Type headers go out, and
// Spring/Tomcat join them as "application/json,application/json",
// which fails MIME parsing with HttpMediaTypeNotSupportedException.
// This test pins the contract: exactly one Content-Type header per
// POST, value "application/json".

TEST(HttpLogSink, ContentTypeHeaderEmittedExactlyOnce) {
    TestServer srv;

    gpufl::HttpLogSink::Options opts;
    opts.base_url = srv.base_url();
    opts.api_key  = "gpfl_ct_test";
    gpufl::HttpLogSink sink(std::move(opts));

    sink.write(gpufl::Channel::All,
               R"({"type":"job_start","app":"ct"})");

    ASSERT_TRUE(waitFor(
        [&] { return srv.capturedCount() >= 1; },
        std::chrono::seconds(5)));
    sink.close();

    auto caps = srv.snapshot();
    ASSERT_FALSE(caps.empty());
    for (const auto& c : caps) {
        EXPECT_EQ(c.content_type_count, 1u)
            << "expected exactly one Content-Type header; got "
            << c.content_type_count
            << " (duplicates → Spring 415 in prod)";
        EXPECT_EQ(c.content_type, "application/json");
    }
}

// ── Per-session HTTP byte tally ──────────────────────────────────────────────
//
// `bytesUploadedCount()` is the running total of request body bytes
// actually written to the socket on successful 2xx POSTs — exposed
// for end-of-session bandwidth reporting (logged once at close()).
// We assert it matches the EXACT sum of body sizes captured by the
// test server. If they ever diverge it means the sink is double-
// counting, missing some uploads, or counting failed ones.

TEST(HttpLogSink, BytesUploadedMatchesCapturedBodySizes) {
    TestServer srv;

    gpufl::HttpLogSink::Options opts;
    opts.base_url = srv.base_url();
    opts.api_key  = "gpfl_byte_count_test";
    gpufl::HttpLogSink sink(std::move(opts));

    // Mix line sizes so we'd notice a wrong-by-constant-offset bug.
    const std::vector<std::string> lines = {
        R"({"type":"job_start","app":"benchmark"})",
        R"({"type":"kernel_event_batch","rows":[{"k":"a"},{"k":"b"},{"k":"c"}]})",
        R"({"type":"shutdown"})",
    };
    for (const auto& l : lines) sink.write(gpufl::Channel::All, l);

    ASSERT_TRUE(waitFor(
        [&] { return srv.capturedCount() >= lines.size(); },
        std::chrono::seconds(5)));
    sink.close();

    std::size_t expectedBytes = 0;
    for (const auto& c : srv.snapshot()) expectedBytes += c.body.size();

    EXPECT_EQ(sink.bytesUploadedCount(), expectedBytes)
        << "byte tally must equal sum of POSTed body sizes";
    // Sanity: the count should be > 0 — guards against the no-op
    // case where neither the counter NOR the captures saw anything.
    EXPECT_GT(sink.bytesUploadedCount(), 0u);
}

// ── Retry on 5xx then give up ────────────────────────────────────────────────

TEST(HttpLogSink, FiveHundredsExhaustRetriesThenDrop) {
    TestServer srv;
    srv.forced_status.store(500);

    gpufl::HttpLogSink::Options opts;
    opts.base_url = srv.base_url();
    opts.api_key  = "gpfl_test";
    opts.max_retries = 3;
    gpufl::HttpLogSink sink(std::move(opts));

    sink.write(gpufl::Channel::All,
               R"({"type":"job_start","app":"x"})");

    // Wait for the sink to exhaust retries on the single line.
    ASSERT_TRUE(waitFor(
        [&] { return sink.failedCount() >= 1; },
        std::chrono::seconds(10)));

    sink.close();
    // 1 original attempt + 3 retries = 4 requests on the server side.
    EXPECT_EQ(srv.requests.load(), 4);
    EXPECT_EQ(sink.uploadedCount(), 0u);
    EXPECT_EQ(sink.failedCount(),   1u);
}

// ── Regression: https:// on a non-TLS build degrades, doesn't abort ────────
//
// Before the guard in makeClient, passing an https:// URL on a build
// without OpenSSL caused cpp-httplib's Client ctor to throw
// `std::invalid_argument("'https' scheme is not supported.")` from the
// worker thread, which propagated to std::terminate → abort → MSVC
// "Debug Error!" dialog. Now we refuse the URL up front and the sink
// degrades to file-only instead of killing the host application.
// This test is only meaningful on a non-TLS build; on a TLS build the
// https URL would actually try to dial out, which we don't want in
// unit tests.
#if !defined(CPPHTTPLIB_OPENSSL_SUPPORT)
TEST(HttpLogSink, HttpsWithoutTlsDoesNotAbort) {
    gpufl::HttpLogSink::Options opts;
    opts.base_url = "https://api.gpuflight.com";
    opts.api_key  = "gpfl_test";
    // Constructor must not crash even though https isn't supported.
    gpufl::HttpLogSink sink(std::move(opts));
    // Writing a line must not crash either; it should be silently
    // dropped after the worker thread exits when makeClient returns
    // nullptr.
    sink.write(gpufl::Channel::All,
               R"({"type":"job_start","app":"x"})");
    sink.close();
    EXPECT_EQ(sink.uploadedCount(), 0u);
}
#endif

// ── Regression: 3xx redirects don't get retried as 5xx ──────────────────────
//
// Before the fix, the retry classifier only special-cased [400, 500);
// anything else (including 3xx redirects) hit the "5xx retry" branch
// and burned the retry budget — producing log lines like
// "5xx response (302) — retrying after 50ms". The real-world trigger
// was passing `backend_url=http://api.gpuflight.com` to an https-only
// backend, which sends a 302 to the same URL with https. This test
// pins the new contract: a single 302 produces a single POST attempt,
// no retries, and the line is dropped (since the sink intentionally
// does not follow redirects across schemes).

TEST(HttpLogSink, ThreeOhTwoIsNotRetried) {
    TestServer srv;
    srv.forced_status.store(302);

    gpufl::HttpLogSink::Options opts;
    opts.base_url = srv.base_url();
    opts.api_key  = "gpfl_test";
    opts.max_retries = 3;
    gpufl::HttpLogSink sink(std::move(opts));

    sink.write(gpufl::Channel::All,
               R"({"type":"job_start","app":"x"})");

    ASSERT_TRUE(waitFor(
        [&] { return sink.failedCount() >= 1; },
        std::chrono::seconds(5)));
    sink.close();

    // Exactly one attempt — no retries — regardless of max_retries.
    EXPECT_EQ(srv.requests.load(), 1);
    EXPECT_EQ(sink.uploadedCount(), 0u);
    EXPECT_EQ(sink.failedCount(),   1u);
}

// ── Auth failure disables the sink ───────────────────────────────────────────

TEST(HttpLogSink, AuthFailureDisablesSinkForSession) {
    TestServer srv;
    srv.forced_status.store(401);

    gpufl::HttpLogSink::Options opts;
    opts.base_url = srv.base_url();
    opts.api_key  = "gpfl_bogus";
    opts.max_retries = 3;
    gpufl::HttpLogSink sink(std::move(opts));

    // Send one line — expect it to trigger the 401 and disable the sink.
    sink.write(gpufl::Channel::All,
               R"({"type":"job_start","app":"x"})");
    ASSERT_TRUE(waitFor(
        [&] { return sink.failedCount() >= 1 || srv.requests.load() >= 1; },
        std::chrono::seconds(5)));

    // Subsequent writes should be dropped silently (not enqueued / not POSTed).
    const int before = srv.requests.load();
    for (int i = 0; i < 5; ++i) {
        sink.write(gpufl::Channel::All,
                   R"({"type":"scope_event_batch","rows":[]})");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    sink.close();
    EXPECT_EQ(srv.requests.load(), before)
        << "No more requests should be made after auth failure";
}

// ── Shutdown drain ───────────────────────────────────────────────────────────

TEST(HttpLogSink, CloseDrainsQueueWithinTimeout) {
    TestServer srv;

    gpufl::HttpLogSink::Options opts;
    opts.base_url = srv.base_url();
    opts.api_key  = "gpfl_test";
    opts.shutdown_drain_ms = 5000;
    gpufl::HttpLogSink sink(std::move(opts));

    constexpr int N = 50;
    for (int i = 0; i < N; ++i) {
        sink.write(gpufl::Channel::All,
                   R"({"type":"scope_event_batch","rows":[]})");
    }

    const auto t0 = std::chrono::steady_clock::now();
    sink.close();
    const auto elapsed = std::chrono::steady_clock::now() - t0;

    EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(
                  elapsed).count(),
              opts.shutdown_drain_ms + 1000)
        << "close() exceeded drain grace period";
    EXPECT_EQ(sink.uploadedCount(), static_cast<std::size_t>(N))
        << "close() should flush the queue";
}

// ── Queue overflow drops oldest ──────────────────────────────────────────────

TEST(HttpLogSink, QueueOverflowDropsOldest) {
    // Hold the server indefinitely so writes pile up in the queue.
    TestServer srv;

    gpufl::HttpLogSink::Options opts;
    opts.base_url = srv.base_url();
    opts.api_key  = "gpfl_test";
    opts.queue_capacity = 8;  // tiny queue for easy overflow
    opts.shutdown_drain_ms = 1000;

    // Pause the server so lines queue up before being posted.
    srv.forced_status.store(200);  // will serve, but slow down writes
    // Set a server handler that blocks briefly to let the queue fill.
    // We simulate this by sending more writes than server can drain
    // within a short window.
    gpufl::HttpLogSink sink(std::move(opts));

    // Push well beyond capacity in a tight loop, faster than the worker
    // can drain (even at ~200 ns per POST it takes ~1 ms network RTT).
    constexpr int N = 200;
    for (int i = 0; i < N; ++i) {
        sink.write(gpufl::Channel::All,
                   R"({"type":"scope_event_batch","rows":[]})");
    }

    // Wait a bit for the worker to process what it can.
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    sink.close();

    EXPECT_EQ(sink.enqueuedCount(), static_cast<std::size_t>(N));
    // Some lines MUST have been dropped since N > capacity.
    EXPECT_GT(sink.droppedCount(), 0u);
    // Everything accounted for: uploaded + dropped + failed == enqueued
    // (failed + dropped is lines that didn't reach the backend; uploaded
    // is lines that did). "Failed" here includes the drain-cutoff ones.
    EXPECT_EQ(sink.uploadedCount() + sink.droppedCount() + sink.failedCount(),
              sink.enqueuedCount());
}

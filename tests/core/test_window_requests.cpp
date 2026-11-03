// Tests for the window request channel: another process of the launch asks
// this one for a bounded deep window by dropping a file.
//
// The coordinator is faked through the channel's hooks, so these cover what
// the channel decides - which file is a request, what is answered, the caps -
// and the coordinator's own behaviour stays covered by test_deep_window.cpp.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "gpufl/core/json/json.hpp"
#include "gpufl/core/model/deep_window_model.hpp"
#include "gpufl/core/window_requests.hpp"

namespace {

using gpufl::DeepWindowSpec;
using gpufl::OpenRequestResult;
using gpufl::OpenRequestStatus;
using gpufl::detail::WindowRequestChannel;
using gpufl::detail::WindowRequestLimits;

constexpr int64_t kSecond = 1'000'000'000;
constexpr int64_t kStart = 1'800'000'000 * kSecond;

/// A coordinator that answers what the test says, and records what it was asked.
struct FakeCoordinator {
    OpenRequestStatus answer = OpenRequestStatus::Accepted;
    uint64_t next_token = 1;
    uint64_t pending = 0;
    uint64_t last_opened = 0;
    std::vector<DeepWindowSpec> asked;

    WindowRequestChannel::Hooks hooks() {
        WindowRequestChannel::Hooks h;
        h.ctx = this;
        h.request_open = [](void* ctx, const DeepWindowSpec& spec) {
            auto* self = static_cast<FakeCoordinator*>(ctx);
            self->asked.push_back(spec);
            if (self->answer != OpenRequestStatus::Accepted) return OpenRequestResult{0, self->answer};
            self->pending = self->next_token++;
            return OpenRequestResult{self->pending, OpenRequestStatus::Accepted};
        };
        h.last_opened_token = [](void* ctx) { return static_cast<FakeCoordinator*>(ctx)->last_opened; };
        h.pending_open_token = [](void* ctx) { return static_cast<FakeCoordinator*>(ctx)->pending; };
        return h;
    }

    /// The next launch boundary: the queued request opens.
    void openQueued() {
        last_opened = pending;
        pending = 0;
    }
};

class WindowRequestTest : public ::testing::Test {
   protected:
    void SetUp() override {
        dir_ = std::filesystem::temp_directory_path() / "gpufl_window_request_test";
        std::filesystem::remove_all(dir_);
        std::filesystem::create_directories(dir_);
    }

    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
    }

    WindowRequestChannel channel(WindowRequestLimits limits = {}) {
        return WindowRequestChannel(dir_, 4242, limits, coordinator_.hooks());
    }

    void request(const std::string& id, int64_t duration_ms = 1000, int64_t expires_in_ns = 30 * kSecond,
                 const std::string& schema = "gpufl.window_request/1") {
        std::ofstream(dir_ / (id + ".json"))
            << "{\"schema\":\"" << schema << "\",\"request_id\":\"" << id << "\""
            << ",\"rule_id\":\"slow-and-loaded\",\"expression\":\"running+waiting>=4 and ttft_p95>500ms\""
            << ",\"duration_ms\":" << duration_ms << ",\"requested_ns\":" << kStart
            << ",\"expires_ns\":" << kStart + expires_in_ns << "}";
    }

    gpufl::json::JsonValue answer(const std::string& id) {
        return gpufl::json::loadFile((dir_ / (id + ".4242.result.json")).string());
    }

    std::filesystem::path dir_;
    FakeCoordinator coordinator_;
};

TEST_F(WindowRequestTest, ARequestFromAnotherProcessOpensExactlyOneWindow) {
    auto ch = channel();
    request("r1");

    ch.poll(kStart + kSecond);
    ASSERT_EQ(coordinator_.asked.size(), 1u);
    // Queued, not opened: no answer until the window really opens.
    EXPECT_FALSE(answer("r1").is_object());

    coordinator_.openQueued();
    ch.poll(kStart + 2 * kSecond);
    EXPECT_EQ(answer("r1").value<std::string>("status", ""), "opened");
    EXPECT_EQ(answer("r1").value<int64_t>("token", 0), 1);

    // Later beats never ask again for a request already answered.
    ch.poll(kStart + 3 * kSecond);
    ch.poll(kStart + 4 * kSecond);
    EXPECT_EQ(coordinator_.asked.size(), 1u);

    const DeepWindowSpec& spec = coordinator_.asked[0];
    EXPECT_EQ(spec.max_duration_ms, 1000);
    EXPECT_EQ(spec.cooldown_ms, 60000);
    EXPECT_TRUE(spec.trigger.present);
    EXPECT_EQ(spec.trigger.source, "request");
    EXPECT_EQ(spec.trigger.request_id, "r1");
    EXPECT_EQ(spec.trigger.rule_id, "slow-and-loaded");
    EXPECT_EQ(spec.trigger.fired_ns, kStart);
}

TEST_F(WindowRequestTest, TheWindowIsCappedAndTheHourlyBudgetsRefuse) {
    WindowRequestLimits limits;
    limits.max_window_ms = 500;
    limits.max_windows_per_hour = 2;
    limits.max_capture_ms_per_hour = 10'000;
    auto ch = channel(limits);

    for (const char* id : {"a", "b", "c"}) {
        request(id, 5000);
        ch.poll(kStart + kSecond);
        coordinator_.openQueued();
        ch.poll(kStart + kSecond);
    }
    EXPECT_EQ(coordinator_.asked.size(), 2u);
    EXPECT_EQ(coordinator_.asked[0].max_duration_ms, 500);
    EXPECT_EQ(answer("b").value<std::string>("status", ""), "opened");
    EXPECT_EQ(answer("c").value<std::string>("status", ""), "budget");

    // An hour later the budget is free again.
    request("d", 5000, 2 * 3600 * kSecond);
    ch.poll(kStart + kSecond + 3600 * kSecond);
    EXPECT_EQ(coordinator_.asked.size(), 3u);
}

TEST_F(WindowRequestTest, CaptureTimeHasItsOwnCap) {
    WindowRequestLimits limits;
    limits.max_window_ms = 1000;
    limits.max_capture_ms_per_hour = 1500;
    auto ch = channel(limits);

    request("a");
    ch.poll(kStart + kSecond);
    coordinator_.openQueued();
    ch.poll(kStart + kSecond);
    request("b");
    ch.poll(kStart + 2 * kSecond);
    EXPECT_EQ(answer("b").value<std::string>("status", ""), "budget");
}

TEST_F(WindowRequestTest, AProcessWithoutACudaContextLeavesTheRequestAlone) {
    coordinator_.answer = OpenRequestStatus::PreparationPending;
    auto ch = channel();
    request("r1");

    ch.poll(kStart + kSecond);
    ch.poll(kStart + 2 * kSecond);
    EXPECT_FALSE(answer("r1").is_object());

    // An engine that gets its context before the request expires acts then.
    coordinator_.answer = OpenRequestStatus::Accepted;
    ch.poll(kStart + 3 * kSecond);
    coordinator_.openQueued();
    ch.poll(kStart + 4 * kSecond);
    EXPECT_EQ(answer("r1").value<std::string>("status", ""), "opened");
}

TEST_F(WindowRequestTest, ARefusalIsAnsweredWithItsReason) {
    coordinator_.answer = OpenRequestStatus::Cooldown;
    auto ch = channel();
    request("cool");
    ch.poll(kStart + kSecond);
    EXPECT_EQ(answer("cool").value<std::string>("status", ""), "cooldown");

    coordinator_.answer = OpenRequestStatus::Busy;
    request("busy");
    ch.poll(kStart + kSecond);
    EXPECT_EQ(answer("busy").value<std::string>("status", ""), "busy");
}

TEST_F(WindowRequestTest, ExpiredHalfWrittenAndMalformedRequestsOpenNothing) {
    auto ch = channel();
    request("late", 1000, kSecond);
    // A writer's temporary file is not a request yet.
    std::ofstream(dir_ / ".pending.json.tmp") << "{\"schema\":";
    std::ofstream(dir_ / "broken.json") << "{\"schema\":";
    request("wrong-schema", 1000, 30 * kSecond, "gpufl.window_request/0");
    // A name that could reach outside the directory is never read.
    std::ofstream(dir_ / "a b.json") << "{}";

    ch.poll(kStart + 5 * kSecond);
    EXPECT_TRUE(coordinator_.asked.empty());
    EXPECT_EQ(answer("late").value<std::string>("status", ""), "expired");
    EXPECT_EQ(answer("broken").value<std::string>("status", ""), "invalid");
    EXPECT_EQ(answer("wrong-schema").value<std::string>("status", ""), "invalid");
    EXPECT_FALSE(std::filesystem::exists(dir_ / "a b.4242.result.json"));
    // The channel's own answers are never taken for requests.
    ch.poll(kStart + 6 * kSecond);
    EXPECT_TRUE(coordinator_.asked.empty());
}

TEST_F(WindowRequestTest, TheWindowRecordSaysWhoAskedForIt) {
    gpufl::DeepWindowEvent event;
    event.trigger.present = true;
    event.trigger.source = "request";
    event.trigger.request_id = "r1";
    event.trigger.expression = "ttft_p95>500ms";
    const std::string json = gpufl::model::DeepWindowModel(event).buildJson();
    EXPECT_NE(json.find("\"source\":\"request\""), std::string::npos);
    EXPECT_NE(json.find("\"request_id\":\"r1\""), std::string::npos);
    EXPECT_NE(json.find("\"expression\":\"ttft_p95>500ms\""), std::string::npos);

    // A window nothing new asked for reads as it always did.
    gpufl::DeepWindowEvent plain;
    plain.trigger.present = true;
    EXPECT_EQ(gpufl::model::DeepWindowModel(plain).buildJson().find("\"source\""), std::string::npos);
}

}  // namespace

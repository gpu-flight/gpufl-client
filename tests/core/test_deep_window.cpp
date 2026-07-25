// Tests for the bounded deep-profiling window.
//
// The window's job is to arm the deep engines for a short region and then
// close ITSELF, so what matters here is the state machine: which bound
// fires first, that a repeated trigger doesn't extend an open window, and
// that the close reason reported is the one that actually fired. The CUPTI
// arming behind it needs a GPU and is covered by the E2E runs.
//
// A real runtime is initialized (backend None, no sampler) because Open()
// deliberately refuses to report a window as open when gpufl isn't running.

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <string>
#include <thread>

#include "gpufl/core/deep_window.hpp"
#include "gpufl/core/events.hpp"
#include "gpufl/core/model/deep_window_model.hpp"
#include "gpufl/core/runtime.hpp"
#include "gpufl/gpufl.hpp"

namespace {

class DeepWindowTest : public ::testing::Test {
   protected:
    void SetUp() override {
        log_dir_ = (std::filesystem::temp_directory_path() /
                    "gpufl_deep_window_test")
                       .string();
        std::filesystem::remove_all(log_dir_);

        gpufl::InitOptions opts;
        opts.app_name = "deep-window-test";
        opts.log_path = log_dir_;
        opts.backend = gpufl::BackendKind::None;
        opts.system_sample_rate_ms = 0;
        opts.continuous_system_sampling = false;
        opts.enable_stack_trace = false;
        opts.enable_source_collection = false;
        ASSERT_TRUE(gpufl::init(opts));
        gpufl::DeepWindow::ResetForTesting();
    }

    void TearDown() override {
        gpufl::DeepWindow::ResetForTesting();
        gpufl::shutdown();
        std::error_code ec;
        std::filesystem::remove_all(log_dir_, ec);
    }

    std::string log_dir_;
};

gpufl::DeepWindowSpec Spec(const int64_t ms, const uint64_t launches,
                           const int64_t cooldown_ms = 0) {
    gpufl::DeepWindowSpec spec;
    spec.max_duration_ms = ms;
    spec.max_launches = launches;
    spec.cooldown_ms = cooldown_ms;
    return spec;
}

}  // namespace

// ── open / close state machine ──────────────────────────────────────────────

TEST_F(DeepWindowTest, StartsInactive) {
    EXPECT_FALSE(gpufl::DeepWindow::Active());
}

TEST_F(DeepWindowTest, OpenActivatesAndManualCloseDeactivates) {
    EXPECT_TRUE(gpufl::DeepWindow::Open(Spec(0, 0)));
    EXPECT_TRUE(gpufl::DeepWindow::Active());

    gpufl::DeepWindow::Close(gpufl::DeepWindowClose::Manual);
    EXPECT_FALSE(gpufl::DeepWindow::Active());
}

TEST_F(DeepWindowTest, CloseWithoutAnOpenWindowIsHarmless) {
    gpufl::DeepWindow::Close(gpufl::DeepWindowClose::Manual);
    EXPECT_FALSE(gpufl::DeepWindow::Active());
}

TEST_F(DeepWindowTest, SecondOpenIsIgnoredNotAnExtension) {
    // The motivating case: a trigger that re-fires every training step must
    // not hold the window open past its bound.
    ASSERT_TRUE(gpufl::DeepWindow::Open(Spec(0, 3)));
    EXPECT_FALSE(gpufl::DeepWindow::Open(Spec(0, 1000)));

    // Still bounded by the FIRST spec's budget of 3.
    gpufl::DeepWindow::OnLaunch();
    EXPECT_FALSE(gpufl::DeepWindow::Open(Spec(0, 1000)));
    gpufl::DeepWindow::OnLaunch();
    EXPECT_TRUE(gpufl::DeepWindow::Active());
    gpufl::DeepWindow::OnLaunch();
    EXPECT_FALSE(gpufl::DeepWindow::Active());
}

TEST_F(DeepWindowTest, ReopensAfterClosing) {
    ASSERT_TRUE(gpufl::DeepWindow::Open(Spec(0, 1)));
    gpufl::DeepWindow::OnLaunch();
    ASSERT_FALSE(gpufl::DeepWindow::Active());

    EXPECT_TRUE(gpufl::DeepWindow::Open(Spec(0, 1)));
    EXPECT_TRUE(gpufl::DeepWindow::Active());
}

// ── bounds ──────────────────────────────────────────────────────────────────

TEST_F(DeepWindowTest, LaunchBudgetClosesOnTheNthLaunch) {
    ASSERT_TRUE(gpufl::DeepWindow::Open(Spec(0, 2)));
    gpufl::DeepWindow::OnLaunch();
    EXPECT_TRUE(gpufl::DeepWindow::Active()) << "budget of 2 spent after 1";
    gpufl::DeepWindow::OnLaunch();
    EXPECT_FALSE(gpufl::DeepWindow::Active());
}

TEST_F(DeepWindowTest, NoBoundsMeansOnlyAManualCloseEndsIt) {
    ASSERT_TRUE(gpufl::DeepWindow::Open(Spec(0, 0)));
    for (int i = 0; i < 100; ++i) gpufl::DeepWindow::OnLaunch();
    EXPECT_TRUE(gpufl::DeepWindow::Active());
}

TEST_F(DeepWindowTest, DeadlineClosesOnTheNextLaunch) {
    ASSERT_TRUE(gpufl::DeepWindow::Open(Spec(/*ms=*/1, 0)));
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    // The deadline is only observed at a launch boundary - that is the one
    // place a mid-session CUPTI stop is safe.
    EXPECT_TRUE(gpufl::DeepWindow::Active());
    gpufl::DeepWindow::OnLaunch();
    EXPECT_FALSE(gpufl::DeepWindow::Active());
}

TEST_F(DeepWindowTest, LaunchBudgetWinsWhenItIsReachedFirst) {
    ASSERT_TRUE(gpufl::DeepWindow::Open(Spec(/*ms=*/60000, /*launches=*/1)));
    gpufl::DeepWindow::OnLaunch();
    EXPECT_FALSE(gpufl::DeepWindow::Active());
}

TEST_F(DeepWindowTest, OnLaunchWithNoWindowOpenIsHarmless) {
    for (int i = 0; i < 10; ++i) gpufl::DeepWindow::OnLaunch();
    EXPECT_FALSE(gpufl::DeepWindow::Active());
}

// ── periodic tick fallback ──────────────────────────────────────────────────

TEST_F(DeepWindowTest, PeriodicTickClosesAnExpiredWindowWhenAllowed) {
    ASSERT_TRUE(gpufl::DeepWindow::Open(Spec(/*ms=*/1, 0)));
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    gpufl::DeepWindow::OnPeriodicTick(/*may_close_here=*/true);
    EXPECT_FALSE(gpufl::DeepWindow::Active());
}

TEST_F(DeepWindowTest, PeriodicTickDefersToTheNextLaunchWhenNotAllowed) {
    // Windows injection: the collector thread must not run the CUPTI
    // teardown, so it only flags the close.
    ASSERT_TRUE(gpufl::DeepWindow::Open(Spec(/*ms=*/1, 0)));
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    gpufl::DeepWindow::OnPeriodicTick(/*may_close_here=*/false);
    EXPECT_TRUE(gpufl::DeepWindow::Active()) << "must wait for a launch";

    gpufl::DeepWindow::OnLaunch();
    EXPECT_FALSE(gpufl::DeepWindow::Active());
}

TEST_F(DeepWindowTest, PeriodicTickLeavesAnUnexpiredWindowAlone) {
    ASSERT_TRUE(gpufl::DeepWindow::Open(Spec(/*ms=*/60000, 0)));
    gpufl::DeepWindow::OnPeriodicTick(/*may_close_here=*/true);
    EXPECT_TRUE(gpufl::DeepWindow::Active());
}

TEST_F(DeepWindowTest, PeriodicTickOnAnUnboundedWindowNeverCloses) {
    ASSERT_TRUE(gpufl::DeepWindow::Open(Spec(0, 0)));
    gpufl::DeepWindow::OnPeriodicTick(/*may_close_here=*/true);
    EXPECT_TRUE(gpufl::DeepWindow::Active());
}

// ── cooldown ────────────────────────────────────────────────────────────────

TEST_F(DeepWindowTest, CooldownBlocksAnImmediateReopen) {
    // The trap this exists for: a condition that stays true reopens a window
    // the instant the last one expired, and the run pays deep cost forever.
    ASSERT_TRUE(gpufl::DeepWindow::Open(Spec(0, 1, /*cooldown_ms=*/60000)));
    gpufl::DeepWindow::OnLaunch();
    ASSERT_FALSE(gpufl::DeepWindow::Active());

    EXPECT_FALSE(gpufl::DeepWindow::Open(Spec(0, 1, 60000)));
    EXPECT_FALSE(gpufl::DeepWindow::Active());
}

TEST_F(DeepWindowTest, CooldownExpiresAndReopeningWorksAgain) {
    ASSERT_TRUE(gpufl::DeepWindow::Open(Spec(0, 1, /*cooldown_ms=*/5)));
    gpufl::DeepWindow::OnLaunch();
    ASSERT_FALSE(gpufl::DeepWindow::Active());

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_TRUE(gpufl::DeepWindow::Open(Spec(0, 1, 5)));
}

TEST_F(DeepWindowTest, NoCooldownMeansImmediateReopenIsAllowed) {
    ASSERT_TRUE(gpufl::DeepWindow::Open(Spec(0, 1)));
    gpufl::DeepWindow::OnLaunch();
    ASSERT_FALSE(gpufl::DeepWindow::Active());
    EXPECT_TRUE(gpufl::DeepWindow::Open(Spec(0, 1)));
}

// ── deferred arm ────────────────────────────────────────────────────────────

TEST_F(DeepWindowTest, RequestOpenArmsOnTheNextLaunchNotImmediately) {
    // A trigger off the app thread must not arm: the CUPTI calls behind an
    // arm are only safe on the app thread at a launch boundary.
    gpufl::DeepWindow::RequestOpen(Spec(60000, 0));
    EXPECT_FALSE(gpufl::DeepWindow::Active()) << "must wait for a launch";

    gpufl::DeepWindow::OnLaunch();
    EXPECT_TRUE(gpufl::DeepWindow::Active());
}

TEST_F(DeepWindowTest, RequestOpenIsConsumedOnce) {
    gpufl::DeepWindow::RequestOpen(Spec(60000, 0));
    gpufl::DeepWindow::OnLaunch();
    ASSERT_TRUE(gpufl::DeepWindow::Active());

    gpufl::DeepWindow::Close(gpufl::DeepWindowClose::Manual);
    // The request was spent on the first arm; a closed window stays closed.
    gpufl::DeepWindow::OnLaunch();
    EXPECT_FALSE(gpufl::DeepWindow::Active());
}

TEST_F(DeepWindowTest, RequestOpenCarriesItsBounds) {
    gpufl::DeepWindow::RequestOpen(Spec(0, /*launches=*/2));
    gpufl::DeepWindow::OnLaunch();  // arms; does not consume budget
    ASSERT_TRUE(gpufl::DeepWindow::Active());

    gpufl::DeepWindow::OnLaunch();
    EXPECT_TRUE(gpufl::DeepWindow::Active());
    gpufl::DeepWindow::OnLaunch();
    EXPECT_FALSE(gpufl::DeepWindow::Active());
}

TEST_F(DeepWindowTest, NewestPendingSpecWins) {
    gpufl::DeepWindow::RequestOpen(Spec(0, 1));
    gpufl::DeepWindow::RequestOpen(Spec(60000, 0));
    gpufl::DeepWindow::OnLaunch();
    ASSERT_TRUE(gpufl::DeepWindow::Active());

    // Had the first spec won, this launch would spend its budget of 1.
    gpufl::DeepWindow::OnLaunch();
    EXPECT_TRUE(gpufl::DeepWindow::Active());
}

TEST_F(DeepWindowTest, ScheduledOpenWaitsOutItsDelay) {
    gpufl::DeepWindow::ScheduleOpenAfter(/*delay_ms=*/50, Spec(60000, 0));
    gpufl::DeepWindow::OnLaunch();
    EXPECT_FALSE(gpufl::DeepWindow::Active()) << "not due yet";

    std::this_thread::sleep_for(std::chrono::milliseconds(70));
    gpufl::DeepWindow::OnLaunch();
    EXPECT_TRUE(gpufl::DeepWindow::Active());
}

TEST_F(DeepWindowTest, RequestOpenWhileAWindowIsOpenDoesNotDisturbIt) {
    ASSERT_TRUE(gpufl::DeepWindow::Open(Spec(60000, 0)));
    gpufl::DeepWindow::RequestOpen(Spec(0, 1));
    gpufl::DeepWindow::OnLaunch();
    EXPECT_TRUE(gpufl::DeepWindow::Active())
        << "the open window's bounds still govern";
}

// ── public API surface ──────────────────────────────────────────────────────

TEST_F(DeepWindowTest, PublicApiOpensAndCloses) {
    gpufl::deepWindow(/*max_duration_ms=*/60000);
    EXPECT_TRUE(gpufl::deepWindowActive());
    gpufl::deepWindowClose();
    EXPECT_FALSE(gpufl::deepWindowActive());
}

// ── close reasons ───────────────────────────────────────────────────────────

TEST_F(DeepWindowTest, CloseReasonWireNames) {
    EXPECT_STREQ("deadline",
                 gpufl::DeepWindowCloseName(gpufl::DeepWindowClose::Deadline));
    EXPECT_STREQ(
        "launch_budget",
        gpufl::DeepWindowCloseName(gpufl::DeepWindowClose::LaunchBudget));
    EXPECT_STREQ("manual",
                 gpufl::DeepWindowCloseName(gpufl::DeepWindowClose::Manual));
    EXPECT_STREQ(
        "session_stop",
        gpufl::DeepWindowCloseName(gpufl::DeepWindowClose::SessionStop));
}

// ── event serialization ─────────────────────────────────────────────────────

TEST(DeepWindowModelTest, SerializesRequestedBoundsAlongsideTheOutcome) {
    // Both sides are on the wire on purpose: "asked for 3000ms, got 12
    // launches, closed by deadline" is the reading that makes a short
    // window legible instead of looking like a failure.
    gpufl::DeepWindowEvent e;
    e.pid = 4242;
    e.app = "trainer";
    e.session_id = "sess-1";
    e.name = "deep_window";
    e.close_reason = "deadline";
    e.engine = "nvidia.pc_sampling";
    e.start_ns = 1000;
    e.end_ns = 3000;
    e.duration_ns = 2000;
    e.launches_covered = 12;
    e.requested_duration_ms = 3000;
    e.requested_max_launches = 0;

    const std::string json = gpufl::model::DeepWindowModel(e).buildJson();
    EXPECT_NE(json.find("\"type\":\"deep_window_event\""), std::string::npos);
    EXPECT_NE(json.find("\"close_reason\":\"deadline\""), std::string::npos);
    EXPECT_NE(json.find("\"engine\":\"nvidia.pc_sampling\""), std::string::npos);
    EXPECT_NE(json.find("\"launches_covered\":12"), std::string::npos);
    EXPECT_NE(json.find("\"requested_duration_ms\":3000"), std::string::npos);
    EXPECT_NE(json.find("\"duration_ns\":2000"), std::string::npos);
    EXPECT_EQ(gpufl::model::DeepWindowModel(e).channel(), gpufl::Channel::Scope);
}

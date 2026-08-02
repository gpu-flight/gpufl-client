#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "gpufl/core/segment_coordinator.hpp"

namespace {

struct Harness {
    int64_t steady = 1'000'000'000;
    int64_t event = 10'000'000'000;
    bool deep = false;
    std::vector<gpufl::SegmentBoundaryRequest> requests;
    bool accept = true;

    gpufl::SegmentCoordinator make(int64_t every_ms, uint64_t max_rows,
                                   int64_t roll_every_ms = 0,
                                   uint64_t roll_max_bytes = 0) {
        gpufl::SegmentCoordinator::Options options;
        options.segment_every_ms = every_ms;
        options.segment_max_rows = max_rows;
        options.run_roll_every_ms = roll_every_ms;
        options.run_roll_max_bytes = roll_max_bytes;
        options.steady_now_ns = [this] { return steady; };
        options.event_now_ns = [this] { return event; };
        options.deep_window_active = [this] { return deep; };
        options.cutover = [this](const gpufl::SegmentBoundaryRequest& request) {
            requests.push_back(request);
            return accept;
        };
        return gpufl::SegmentCoordinator(std::move(options));
    }
};

TEST(SegmentCoordinatorTest, TimeBoundaryUsesSeparateClockDomains) {
    Harness h;
    auto coordinator = h.make(60'000, 0);
    ASSERT_TRUE(coordinator.start(0, h.steady, h.event));

    h.steady += 60'000'000'000;
    h.event += 61'000'000'000;  // wall/event clock moved independently
    EXPECT_TRUE(coordinator.service());
    ASSERT_EQ(h.requests.size(), 1u);
    EXPECT_EQ(h.requests[0].requested_event_ns, 70'000'000'000);
    EXPECT_EQ(h.requests[0].actual_event_ns, 71'000'000'000);
    EXPECT_EQ(h.requests[0].boundary_delay_ns, 0);
    EXPECT_EQ(coordinator.currentSegmentIndex(), 1u);
}

TEST(SegmentCoordinatorTest, RowCrossingBatchStaysInTheRetiringSegment) {
    Harness h;
    auto coordinator = h.make(0, 100);
    ASSERT_TRUE(coordinator.start(3, h.steady, h.event));

    coordinator.noteRows(3, 60, h.steady + 10, h.event + 10);
    coordinator.noteRows(3, 50, h.steady + 20, h.event + 20);
    EXPECT_EQ(coordinator.currentRows(), 110u);
    EXPECT_TRUE(coordinator.service());

    ASSERT_EQ(h.requests.size(), 1u);
    EXPECT_EQ(h.requests[0].reason,
              gpufl::SegmentBoundaryReason::RowBudget);
    EXPECT_EQ(h.requests[0].retiring_segment_index, 3u);
    EXPECT_EQ(h.requests[0].requested_steady_ns, h.steady + 20);
    EXPECT_EQ(coordinator.currentRows(), 0u);
}

TEST(SegmentCoordinatorTest, EqualTriggerTimestampDeterministicallyChoosesTime) {
    Harness h;
    auto coordinator = h.make(60'000, 1);
    ASSERT_TRUE(coordinator.start(0, h.steady, h.event));

    h.steady += 60'000'000'000;
    h.event += 60'000'000'000;
    coordinator.noteRows(0, 1, h.steady, h.event);
    EXPECT_TRUE(coordinator.service());
    ASSERT_EQ(h.requests.size(), 1u);
    EXPECT_EQ(h.requests[0].reason, gpufl::SegmentBoundaryReason::Time);
}

TEST(SegmentCoordinatorTest, DeepWindowDefersButDoesNotLoseTheBoundary) {
    Harness h;
    auto coordinator = h.make(0, 10);
    ASSERT_TRUE(coordinator.start(0, h.steady, h.event));
    coordinator.noteRows(0, 10, h.steady, h.event);

    h.deep = true;
    EXPECT_FALSE(coordinator.service());
    EXPECT_TRUE(coordinator.boundaryPending());
    EXPECT_TRUE(h.requests.empty());

    h.deep = false;
    h.steady += 100;
    h.event += 100;
    EXPECT_TRUE(coordinator.service());
    ASSERT_EQ(h.requests.size(), 1u);
    EXPECT_EQ(h.requests[0].deferred_by, "deep_window");
    EXPECT_EQ(h.requests[0].boundary_delay_ns, 100);
}

TEST(SegmentCoordinatorTest, RollDeadlineWaitsForTheNextOrdinaryBoundary) {
    Harness h;
    const int64_t start_event = h.event;
    // 60s segments inside 90s run parts: the roll comes due halfway through
    // the second segment.
    auto coordinator = h.make(60'000, 0, 90'000, 0);
    ASSERT_TRUE(coordinator.start(0, h.steady, h.event));

    h.steady += 60'000'000'000;
    h.event += 60'000'000'000;
    ASSERT_TRUE(coordinator.service());
    ASSERT_EQ(h.requests.size(), 1u);
    EXPECT_FALSE(h.requests[0].ends_run) << "segment 0 is inside the budget";

    // t=90s: the run budget is spent, but we are only 30s into segment 1.
    h.steady += 30'000'000'000;
    h.event += 30'000'000'000;
    EXPECT_FALSE(coordinator.service()) << "a roll must not cut mid-segment";
    EXPECT_EQ(h.requests.size(), 1u);
    EXPECT_TRUE(coordinator.runRollPending());
    EXPECT_FALSE(coordinator.boundaryPending());

    // t=120s: segment 1's own cadence is due, and it carries the run end.
    h.steady += 30'000'000'000;
    h.event += 30'000'000'000;
    ASSERT_TRUE(coordinator.service());
    ASSERT_EQ(h.requests.size(), 2u);

    const auto& roll = h.requests[1];
    EXPECT_TRUE(roll.ends_run);
    EXPECT_EQ(roll.reason, gpufl::SegmentBoundaryReason::Time)
        << "the segment was still cut by its own cadence";
    EXPECT_EQ(roll.rollover_reason, gpufl::SegmentBoundaryReason::RunRollTime);
    // Due at 90s, cut at 120s: 30s of recorded overshoot, which is exactly why
    // these are two fields and not one.
    EXPECT_EQ(roll.requested_rollover_event_ns, start_event + 90'000'000'000);
    EXPECT_EQ(roll.actual_rollover_event_ns, roll.actual_event_ns);
    EXPECT_EQ(roll.requested_event_ns, start_event + 120'000'000'000);
}

TEST(SegmentCoordinatorTest, LateRowsFromARetiredContextCannotRetrigger) {
    Harness h;
    auto coordinator = h.make(0, 10);
    ASSERT_TRUE(coordinator.start(0, h.steady, h.event));
    coordinator.noteRows(0, 10, h.steady, h.event);
    ASSERT_TRUE(coordinator.service());

    coordinator.noteRows(0, 1'000, h.steady + 1, h.event + 1);
    EXPECT_EQ(coordinator.currentRows(), 0u);
    EXPECT_FALSE(coordinator.boundaryPending());
}

TEST(SegmentCoordinatorTest, RejectedCutoverRemainsPendingForRetry) {
    Harness h;
    h.accept = false;
    auto coordinator = h.make(0, 1);
    ASSERT_TRUE(coordinator.start(0, h.steady, h.event));
    coordinator.noteRows(0, 1, h.steady, h.event);

    EXPECT_FALSE(coordinator.service());
    EXPECT_TRUE(coordinator.boundaryPending());
    h.accept = true;
    EXPECT_TRUE(coordinator.service());
    EXPECT_EQ(h.requests.size(), 2u);
}

// The byte budget behaves like the time budget: crossing it mid-segment does
// not cut the segment short.
TEST(SegmentCoordinatorTest, ByteBudgetWaitsForTheNextOrdinaryBoundary) {
    Harness h;
    const int64_t start_event = h.event;
    auto coordinator = h.make(60'000, 0, 0, 1000);
    ASSERT_TRUE(coordinator.start(0, h.steady, h.event));

    h.steady += 10'000'000'000;
    h.event += 10'000'000'000;
    coordinator.noteBytes(0, 1000, h.steady, h.event);
    EXPECT_FALSE(coordinator.service());
    EXPECT_TRUE(coordinator.runRollPending());
    EXPECT_TRUE(h.requests.empty());

    h.steady += 50'000'000'000;
    h.event += 50'000'000'000;
    ASSERT_TRUE(coordinator.service());
    ASSERT_EQ(h.requests.size(), 1u);
    EXPECT_TRUE(h.requests[0].ends_run);
    EXPECT_EQ(h.requests[0].rollover_reason,
              gpufl::SegmentBoundaryReason::RunRollBytes);
    EXPECT_EQ(h.requests[0].requested_rollover_event_ns,
              start_event + 10'000'000'000);
    EXPECT_EQ(coordinator.currentRunPartBytes(), 0u);
}

// The one place run-part state differs from segment state. Rows belong to the
// segment and reset at every cut; bytes belong to the part and must survive
// one, or a long part never reaches its budget.
TEST(SegmentCoordinatorTest, OnlyARollResetsTheRunPartByteCounter) {
    Harness h;
    auto coordinator = h.make(60'000, 0, 0, 1000);
    ASSERT_TRUE(coordinator.start(0, h.steady, h.event));

    coordinator.noteRows(0, 7, h.steady, h.event);
    coordinator.noteBytes(0, 400, h.steady, h.event);
    h.steady += 60'000'000'000;
    h.event += 60'000'000'000;
    ASSERT_TRUE(coordinator.service());
    ASSERT_EQ(h.requests.size(), 1u);
    EXPECT_FALSE(h.requests[0].ends_run);
    EXPECT_EQ(coordinator.currentRows(), 0u) << "rows are per segment";
    EXPECT_EQ(coordinator.currentRunPartBytes(), 400u) << "bytes are per part";

    // 600 more crosses 1000, accumulated across the segment boundary.
    coordinator.noteBytes(1, 600, h.steady, h.event);
    EXPECT_EQ(coordinator.currentRunPartBytes(), 1000u);
    EXPECT_TRUE(coordinator.runRollPending());

    h.steady += 60'000'000'000;
    h.event += 60'000'000'000;
    ASSERT_TRUE(coordinator.service());
    ASSERT_EQ(h.requests.size(), 2u);
    EXPECT_TRUE(h.requests[1].ends_run);
    EXPECT_EQ(coordinator.currentRunPartBytes(), 0u);
}

TEST(SegmentCoordinatorTest, TheEarlierRunBudgetCrossingNamesTheRollReason) {
    {  // Bytes crossed first, so bytes name the roll even though time is spent.
        Harness h;
        auto coordinator = h.make(60'000, 0, 90'000, 1000);
        ASSERT_TRUE(coordinator.start(0, h.steady, h.event));
        coordinator.noteBytes(0, 1000, h.steady + 10, h.event + 10);

        h.steady += 120'000'000'000;
        h.event += 120'000'000'000;
        ASSERT_TRUE(coordinator.service());
        ASSERT_EQ(h.requests.size(), 1u);
        EXPECT_TRUE(h.requests[0].ends_run);
        EXPECT_EQ(h.requests[0].rollover_reason,
                  gpufl::SegmentBoundaryReason::RunRollBytes);
    }
    {  // Exact tie resolves to time rather than to whichever armed first.
        Harness h;
        auto coordinator = h.make(60'000, 0, 90'000, 1000);
        ASSERT_TRUE(coordinator.start(0, h.steady, h.event));
        coordinator.noteBytes(0, 1000, h.steady + 90'000'000'000,
                              h.event + 90'000'000'000);

        h.steady += 120'000'000'000;
        h.event += 120'000'000'000;
        ASSERT_TRUE(coordinator.service());
        ASSERT_EQ(h.requests.size(), 1u);
        EXPECT_EQ(h.requests[0].rollover_reason,
                  gpufl::SegmentBoundaryReason::RunRollTime);
    }
}

// Arming happens before the deep-window check, so a budget spent inside a
// window is recorded rather than skipped.
TEST(SegmentCoordinatorTest, ADeepWindowDefersTheRollWithoutLosingIt) {
    Harness h;
    auto coordinator = h.make(60'000, 0, 60'000, 0);
    ASSERT_TRUE(coordinator.start(0, h.steady, h.event));

    h.deep = true;
    h.steady += 60'000'000'000;
    h.event += 60'000'000'000;
    EXPECT_FALSE(coordinator.service());
    EXPECT_TRUE(coordinator.boundaryPending());
    EXPECT_TRUE(coordinator.runRollPending())
        << "the roll must arm during the window, not be skipped";
    EXPECT_TRUE(h.requests.empty());

    h.deep = false;
    h.steady += 5'000'000'000;
    h.event += 5'000'000'000;
    ASSERT_TRUE(coordinator.service());
    ASSERT_EQ(h.requests.size(), 1u);
    EXPECT_TRUE(h.requests[0].ends_run);
    EXPECT_EQ(h.requests[0].deferred_by, "deep_window");
    EXPECT_EQ(h.requests[0].boundary_delay_ns, 5'000'000'000);
}

// The bound the whole design rests on, driven by a real tick loop rather than
// hand-placed clock jumps: a part runs at least its budget and overshoots by
// at most one segment.
TEST(SegmentCoordinatorTest, RunPartOvershootStaysWithinOneSegment) {
    Harness h;
    constexpr int64_t kSegmentNs = 60'000'000'000;
    constexpr int64_t kRollNs = 200'000'000'000;  // not a multiple of cadence
    auto coordinator = h.make(60'000, 0, 200'000, 0);
    ASSERT_TRUE(coordinator.start(0, h.steady, h.event));
    const int64_t part_start_event = h.event;

    for (int tick = 0; tick < 300; ++tick) {
        h.steady += 1'000'000'000;
        h.event += 1'000'000'000;
        coordinator.service();
    }

    const auto roll = std::find_if(
        h.requests.begin(), h.requests.end(),
        [](const gpufl::SegmentBoundaryRequest& r) { return r.ends_run; });
    ASSERT_NE(roll, h.requests.end());
    EXPECT_EQ(std::count_if(
                  h.requests.begin(), h.requests.end(),
                  [](const gpufl::SegmentBoundaryRequest& r) {
                      return r.ends_run;
                  }),
              1) << "the budget must restart, not re-fire every segment";

    const int64_t part_duration = roll->actual_event_ns - part_start_event;
    EXPECT_GE(part_duration, kRollNs) << "a part must not end early";
    EXPECT_LE(part_duration, kRollNs + kSegmentNs);
    EXPECT_EQ(roll->deferred_by, "");
}

}  // namespace

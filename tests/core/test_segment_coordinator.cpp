#include <gtest/gtest.h>

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

    gpufl::SegmentCoordinator make(int64_t every_ms, uint64_t max_rows) {
        gpufl::SegmentCoordinator::Options options;
        options.segment_every_ms = every_ms;
        options.segment_max_rows = max_rows;
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

}  // namespace

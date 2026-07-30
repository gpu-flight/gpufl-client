#include "gpufl/core/segment_coordinator.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <utility>

#include "gpufl/core/common.hpp"
#include "gpufl/core/deep_window.hpp"

namespace gpufl {
namespace {

int64_t defaultSteadyNowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

}  // namespace

const char* segmentBoundaryReasonName(const SegmentBoundaryReason reason) {
    switch (reason) {
        case SegmentBoundaryReason::Time: return "time";
        case SegmentBoundaryReason::RowBudget: return "row_budget";
    }
    return "time";
}

SegmentCoordinator::SegmentCoordinator(Options options)
    : options_(std::move(options)) {
    if (!options_.steady_now_ns) options_.steady_now_ns = defaultSteadyNowNs;
    if (!options_.event_now_ns) {
        options_.event_now_ns = [] { return detail::GetTimestampNs(); };
    }
    if (!options_.deep_window_active) {
        options_.deep_window_active = [] { return DeepWindow::Active(); };
    }
}

bool SegmentCoordinator::start(const uint32_t segment_index,
                               const int64_t steady_start_ns,
                               const int64_t event_start_ns) {
    std::lock_guard lock(mu_);
    if (started_ || finished_ || steady_start_ns < 0 || event_start_ns <= 0) {
        return false;
    }
    started_ = true;
    current_segment_index_ = segment_index;
    segment_start_steady_ns_ = steady_start_ns;
    segment_start_event_ns_ = event_start_ns;
    return true;
}

void SegmentCoordinator::noteRows(const uint32_t segment_index,
                                  const uint64_t rows,
                                  const int64_t committed_steady_ns,
                                  const int64_t committed_event_ns) {
    if (rows == 0) return;
    std::lock_guard lock(mu_);
    if (!started_ || finished_ || segment_index != current_segment_index_) {
        return;
    }
    const uint64_t remaining =
        (std::numeric_limits<uint64_t>::max)() - current_rows_;
    current_rows_ += (std::min)(remaining, rows);
    if (options_.segment_max_rows > 0 &&
        current_rows_ >= options_.segment_max_rows && !rows_.present) {
        rows_ = Pending{true, SegmentBoundaryReason::RowBudget,
                        committed_steady_ns, committed_event_ns};
    }
}

void SegmentCoordinator::considerTimeLocked_(const int64_t steady_now_ns) {
    if (options_.segment_every_ms <= 0 || time_.present) return;
    const int64_t cadence_ns =
        options_.segment_every_ms > (std::numeric_limits<int64_t>::max)() /
                                         1'000'000
            ? (std::numeric_limits<int64_t>::max)()
            : options_.segment_every_ms * 1'000'000;
    const int64_t deadline =
        segment_start_steady_ns_ >
                (std::numeric_limits<int64_t>::max)() - cadence_ns
            ? (std::numeric_limits<int64_t>::max)()
            : segment_start_steady_ns_ + cadence_ns;
    if (steady_now_ns < deadline) return;

    const int64_t projected_event =
        segment_start_event_ns_ >
                (std::numeric_limits<int64_t>::max)() -
                    (deadline - segment_start_steady_ns_)
            ? (std::numeric_limits<int64_t>::max)()
            : segment_start_event_ns_ +
                  (deadline - segment_start_steady_ns_);
    time_ = Pending{true, SegmentBoundaryReason::Time, deadline,
                    projected_event};
}

SegmentCoordinator::Pending SegmentCoordinator::winnerLocked_() const {
    if (!time_.present) return rows_;
    if (!rows_.present) return time_;
    if (time_.steady_ns <= rows_.steady_ns) return time_;
    return rows_;
}

bool SegmentCoordinator::service() {
    SegmentBoundaryRequest request;
    {
        std::lock_guard lock(mu_);
        if (!started_ || finished_ || cutover_in_progress_) return false;
        const int64_t steady_now = options_.steady_now_ns();
        considerTimeLocked_(steady_now);
        const Pending pending = winnerLocked_();
        if (!pending.present) return false;
        if (options_.deep_window_active &&
            options_.deep_window_active()) {
            deferred_by_deep_window_ = true;
            return false;
        }

        request.reason = pending.reason;
        request.retiring_segment_index = current_segment_index_;
        request.requested_steady_ns = pending.steady_ns;
        request.requested_event_ns = pending.event_ns;
        request.actual_steady_ns = steady_now;
        request.actual_event_ns = options_.event_now_ns();
        request.boundary_delay_ns =
            (std::max)(int64_t{0}, steady_now - pending.steady_ns);
        if (deferred_by_deep_window_) request.deferred_by = "deep_window";
        cutover_in_progress_ = true;
    }

    const bool completed = options_.cutover && options_.cutover(request);

    {
        std::lock_guard lock(mu_);
        cutover_in_progress_ = false;
        if (!completed || finished_) return false;
        ++current_segment_index_;
        current_rows_ = 0;
        segment_start_steady_ns_ = request.actual_steady_ns;
        segment_start_event_ns_ = request.actual_event_ns;
        time_ = {};
        rows_ = {};
        deferred_by_deep_window_ = false;
    }
    return true;
}

void SegmentCoordinator::finish() {
    std::lock_guard lock(mu_);
    finished_ = true;
    time_ = {};
    rows_ = {};
}

uint32_t SegmentCoordinator::currentSegmentIndex() const {
    std::lock_guard lock(mu_);
    return current_segment_index_;
}

uint64_t SegmentCoordinator::currentRows() const {
    std::lock_guard lock(mu_);
    return current_rows_;
}

bool SegmentCoordinator::boundaryPending() const {
    std::lock_guard lock(mu_);
    return time_.present || rows_.present;
}

}  // namespace gpufl

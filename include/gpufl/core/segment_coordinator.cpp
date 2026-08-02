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

struct ProjectedDeadline {
    int64_t steady_ns = 0;
    int64_t event_ns = 0;
};

ProjectedDeadline projectDeadline(const int64_t every_ms,
                                  const int64_t anchor_steady_ns,
                                  const int64_t anchor_event_ns) {
    constexpr int64_t kMax = (std::numeric_limits<int64_t>::max)();
    const int64_t span_ns = every_ms > kMax / 1'000'000 ? kMax : every_ms * 1'000'000;
    const int64_t steady_ns =
        anchor_steady_ns > kMax - span_ns ? kMax : anchor_steady_ns + span_ns;
    const int64_t elapsed_ns = steady_ns - anchor_steady_ns;
    const int64_t event_ns = anchor_event_ns > kMax - elapsed_ns
                                 ? kMax
                                 : anchor_event_ns + elapsed_ns;
    return {steady_ns, event_ns};
}
}  // namespace

const char* segmentBoundaryReasonName(const SegmentBoundaryReason reason) {
    switch (reason) {
        case SegmentBoundaryReason::Time: return "time";
        case SegmentBoundaryReason::RowBudget: return "row_budget";
        case SegmentBoundaryReason::RunRollTime: return "run_roll_time";
        case SegmentBoundaryReason::RunRollBytes: return "run_roll_bytes";
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
    run_part_start_steady_ns_ = steady_start_ns;
    run_part_start_event_ns_ = event_start_ns;
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

void SegmentCoordinator::noteBytes(const uint32_t segment_index,
                                   const uint64_t bytes,
                                   const int64_t committed_steady_ns,
                                   const int64_t committed_event_ns) {
    if (bytes == 0) return;
    std::lock_guard lock(mu_);
    if (!started_ || finished_ || segment_index != current_segment_index_) {
        return;
    }
    const uint64_t remaining =
        (std::numeric_limits<uint64_t>::max)() - run_part_bytes_;
    run_part_bytes_ += (std::min)(remaining, bytes);
    if (options_.run_roll_max_bytes > 0 &&
        run_part_bytes_ >= options_.run_roll_max_bytes &&
        !roll_bytes_.present) {
        roll_bytes_ = Pending{true, SegmentBoundaryReason::RunRollBytes,
                              committed_steady_ns, committed_event_ns};
    }
}

void SegmentCoordinator::considerTimeLocked_(const int64_t steady_now_ns) {
    if (options_.segment_every_ms <= 0 || time_.present) return;
    const ProjectedDeadline due =
        projectDeadline(options_.segment_every_ms, segment_start_steady_ns_,
                        segment_start_event_ns_);
    if (steady_now_ns < due.steady_ns) return;
    time_ = Pending{true, SegmentBoundaryReason::Time, due.steady_ns,
                    due.event_ns};
}

void SegmentCoordinator::considerRollLocked_(const int64_t steady_now_ns) {
    if (options_.run_roll_every_ms <= 0 || roll_time_.present) return;
    // Anchored on the run PART, not the segment: an ordinary cut must not
    // push the roll deadline out.
    const ProjectedDeadline due =
        projectDeadline(options_.run_roll_every_ms, run_part_start_steady_ns_,
                        run_part_start_event_ns_);
    if (steady_now_ns < due.steady_ns) return;
    roll_time_ = Pending{true, SegmentBoundaryReason::RunRollTime,
                         due.steady_ns, due.event_ns};
}

SegmentCoordinator::Pending SegmentCoordinator::winnerLocked_() const {
    if (!time_.present) return rows_;
    if (!rows_.present) return time_;
    if (time_.steady_ns <= rows_.steady_ns) return time_;
    return rows_;
}

SegmentCoordinator::Pending SegmentCoordinator::rollWinnerLocked_() const {
    if (!roll_time_.present) return roll_bytes_;
    if (!roll_bytes_.present) return roll_time_;
    // Earlier crossing wins; an exact tie resolves to time so the outcome does
    // not depend on which trigger happened to arm first.
    if (roll_time_.steady_ns <= roll_bytes_.steady_ns) return roll_time_;
    return roll_bytes_;
}

bool SegmentCoordinator::service() {
    SegmentBoundaryRequest request;
    {
        std::lock_guard lock(mu_);
        if (!started_ || finished_ || cutover_in_progress_) return false;
        const int64_t steady_now = options_.steady_now_ns();
        considerTimeLocked_(steady_now);
        considerRollLocked_(steady_now);
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
        if (const Pending roll = rollWinnerLocked_(); roll.present) {
            request.ends_run = true;
            request.rollover_reason = roll.reason;
            request.requested_rollover_event_ns = roll.event_ns;
            request.actual_rollover_event_ns = request.actual_event_ns;
        }
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
        if (request.ends_run) {
            // Only a roll resets these. An ordinary cut deliberately leaves
            // the byte counter and the part anchor running, because the budget
            // belongs to the part and spans many segments.
            run_part_bytes_ = 0;
            run_part_start_steady_ns_ = request.actual_steady_ns;
            run_part_start_event_ns_ = request.actual_event_ns;
            roll_time_ = {};
            roll_bytes_ = {};
        }
    }
    return true;
}

void SegmentCoordinator::finish() {
    std::lock_guard lock(mu_);
    finished_ = true;
    time_ = {};
    rows_ = {};
    roll_time_ = {};
    roll_bytes_ = {};
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

uint64_t SegmentCoordinator::currentRunPartBytes() const {
    std::lock_guard lock(mu_);
    return run_part_bytes_;
}

bool SegmentCoordinator::runRollPending() const {
    std::lock_guard lock(mu_);
    return roll_time_.present || roll_bytes_.present;
}

}  // namespace gpufl

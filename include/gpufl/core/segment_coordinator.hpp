#pragma once

#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace gpufl {

enum class SegmentBoundaryReason { Time, RowBudget, RunRollTime, RunRollBytes };

struct SegmentBoundaryRequest {
    SegmentBoundaryReason reason = SegmentBoundaryReason::Time;
    uint32_t retiring_segment_index = 0;
    int64_t requested_steady_ns = 0;
    int64_t requested_event_ns = 0;
    int64_t actual_steady_ns = 0;
    int64_t actual_event_ns = 0;
    int64_t boundary_delay_ns = 0;
    std::string deferred_by;
    bool ends_run = false;
    SegmentBoundaryReason rollover_reason = SegmentBoundaryReason::RunRollTime;
    int64_t requested_rollover_event_ns = 0;
    int64_t actual_rollover_event_ns = 0;
};

/**
 * Trigger arbitration for long-running session segmentation.
 *
 * This class deliberately does not know how a logger or dictionary is cut
 * over. It decides when one boundary is due and invokes a supplied cutover
 * transaction. The callback returns true only after the new SegmentContext is
 * published. Tests can therefore drive every state with fake clocks while the
 * production callback owns the filesystem and lifecycle work.
 */
class SegmentCoordinator {
   public:
    struct Options {
        int64_t segment_every_ms = 0;
        uint64_t segment_max_rows = 0;
        int64_t run_roll_every_ms = 0;
        uint64_t run_roll_max_bytes = 0;
        std::function<int64_t()> steady_now_ns;
        std::function<int64_t()> event_now_ns;
        std::function<bool()> deep_window_active;
        // Production may refine actual_* after its final drain, immediately
        // before SegmentContext publication. requested_* remain immutable.
        std::function<bool(SegmentBoundaryRequest&)> cutover;
    };

    explicit SegmentCoordinator(Options options);

    /** Initialize segment zero's two clock anchors. */
    bool start(uint32_t segment_index, int64_t steady_start_ns,
               int64_t event_start_ns);

    /**
     * Account one atomically committed batch. The batch remains wholly in
     * segment_index; crossing only requests a later cutover.
     */
    void noteRows(uint32_t segment_index, uint64_t rows,
                  int64_t committed_steady_ns,
                  int64_t committed_event_ns);

    /**
     * Account serialized output against the run-part byte budget. Unlike
     * noteRows this accumulates ACROSS ordinary segment boundaries: the budget
     * belongs to the run part, so only a roll reset it.
     */
    void noteBytes(uint32_t segment_index, uint64_t bytes,
                   int64_t committed_steady_ns,
                   int64_t committed_event_ns);

    /** Evaluate due triggers and perform at most one cutover. */
    bool service();

    /** Prevent every future boundary request. Idempotent. */
    void finish();

    uint32_t currentSegmentIndex() const;
    uint64_t currentRows() const;
    bool boundaryPending() const;
    uint64_t currentRunPartBytes() const;

    bool runRollPending() const;

   private:
    struct Pending {
        bool present = false;
        SegmentBoundaryReason reason = SegmentBoundaryReason::Time;
        int64_t steady_ns = 0;
        int64_t event_ns = 0;
    };

    void considerTimeLocked_(int64_t steady_now_ns);
    Pending winnerLocked_() const;

    void considerRollLocked_(int64_t steady_now_ns);
    Pending rollWinnerLocked_() const;

    Options options_;
    mutable std::mutex mu_;
    bool started_ = false;
    bool finished_ = false;
    bool cutover_in_progress_ = false;
    uint32_t current_segment_index_ = 0;
    uint64_t current_rows_ = 0;
    int64_t segment_start_steady_ns_ = 0;
    int64_t segment_start_event_ns_ = 0;
    Pending time_;
    Pending rows_;
    Pending roll_time_;
    Pending roll_bytes_;
    uint64_t run_part_bytes_ = 0;
    int64_t run_part_start_steady_ns_ = 0;
    int64_t run_part_start_event_ns_ = 0;
    bool deferred_by_deep_window_ = false;
};

const char* segmentBoundaryReasonName(SegmentBoundaryReason reason);

}  // namespace gpufl

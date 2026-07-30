#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

#include "gpufl/core/events.hpp"
#include "gpufl/core/logger/logger.hpp"
#include "gpufl/core/segment_coordinator.hpp"

namespace gpufl {

struct Runtime;
struct SegmentContext;

/**
 * Production transaction around SegmentCoordinator.
 *
 * SegmentCoordinator decides when a boundary is due. SegmentRuntime owns the
 * filesystem/logger handoff: acquire the next directory lock by opening its
 * logger, write bootstrap records, publish the immutable context, then retire
 * the old context on a coordinator-owned worker after outstanding writers
 * release it.
 */
class SegmentRuntime {
   public:
    struct Options {
        Runtime* runtime = nullptr;  // process-lifetime owner
        Logger::Options logger_options;
        InitEvent init_template;
        int64_t segment_every_ms = 0;
        uint64_t segment_max_rows = 0;
        // A leaked writer must not make shutdown unkillable. On timeout the
        // segment is deliberately left incomplete and its context is retained
        // for process lifetime rather than closing a sink under a live writer.
        uint32_t retirement_drain_timeout_ms = 5000;
    };

    explicit SegmentRuntime(Options options);
    ~SegmentRuntime();

    SegmentRuntime(const SegmentRuntime&) = delete;
    SegmentRuntime& operator=(const SegmentRuntime&) = delete;

    bool start();
    bool service();
    void noteRows(uint32_t segment_index, uint64_t rows,
                  int64_t committed_steady_ns, int64_t committed_event_ns);

    /** Finalize the current segment and the run after all producers stop. */
    void finish(int64_t ended_ns);

   private:
    struct RetiredSegment {
        std::shared_ptr<const SegmentContext> context;
        SegmentBoundaryRequest boundary;
    };

    bool cutover_(SegmentBoundaryRequest& boundary);
    void enqueueRetirement_(RetiredSegment retired);
    void retirementLoop_();
    bool retire_(RetiredSegment retired);
    bool awaitWriterDrain_(
        const std::shared_ptr<const SegmentContext>& context,
        const char* phase);
    void stopRetirementWorker_();
    void writeShutdown_(const std::shared_ptr<const SegmentContext>& context,
                        int64_t ts_ns);

    Options options_;
    SegmentCoordinator coordinator_;
    std::mutex lifecycle_mu_;
    bool started_ = false;
    bool finished_ = false;

    std::mutex retirement_mu_;
    std::condition_variable retirement_cv_;
    std::deque<RetiredSegment> retirement_queue_;
    bool retirement_stopping_ = false;
    std::thread retirement_thread_;
};

}  // namespace gpufl

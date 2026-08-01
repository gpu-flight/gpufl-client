#include "gpufl/core/segment_runtime.hpp"

#include <algorithm>
#include <chrono>
#include <utility>

#include "gpufl/core/common.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/deep_window_rules.hpp"
#include "gpufl/core/dictionary_manager.hpp"
#include "gpufl/core/model/lifecycle_model.hpp"
#include "gpufl/core/monitor.hpp"
#include "gpufl/core/runtime.hpp"

namespace gpufl {
namespace {

int64_t steadyNowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void quarantineUndrainedContext(
    std::shared_ptr<const SegmentContext> context) {
    // Intentionally process-lifetime. Releasing this reference from
    // SegmentRuntime::~SegmentRuntime could make the final writer destroy and
    // close its Logger on an arbitrary application thread. Keeping the
    // ownership lock also prevents the agent from treating an actively-written
    // directory as complete. The process exit releases both safely.
    static auto* const mutex = new std::mutex;
    static auto* const contexts =
        new std::vector<std::shared_ptr<const SegmentContext>>;
    std::lock_guard<std::mutex> lock(*mutex);
    contexts->push_back(std::move(context));
}

}  // namespace

SegmentRuntime::SegmentRuntime(Options options)
    : options_(std::move(options)),
      coordinator_([this] {
          SegmentCoordinator::Options coordinator_options;
          // Integers survive the move into options_; naming the constructor
          // argument explicitly also keeps older MSVC frontends from treating
          // this member-initializer lambda as captureless.
          coordinator_options.segment_every_ms =
              options_.segment_every_ms;
          coordinator_options.segment_max_rows =
              options_.segment_max_rows;
          coordinator_options.run_roll_every_ms =
              options_.run_roll_every_ms;
          coordinator_options.run_roll_max_bytes =
              options_.run_roll_max_bytes;
          coordinator_options.cutover =
              [this](SegmentBoundaryRequest& boundary) {
              return cutover_(boundary);
          };
          return coordinator_options;
      }()) {}

SegmentRuntime::~SegmentRuntime() {
    coordinator_.finish();
    stopRetirementWorker_();
}

bool SegmentRuntime::start() {
    std::lock_guard lock(lifecycle_mu_);
    if (started_ || finished_ || !options_.runtime) return false;
    const auto context = options_.runtime->peekSegmentContext();
    if (!context || !context->logger || context->run_id.empty()) return false;

    const int64_t steady_ns = steadyNowNs();
    const int64_t event_ns = context->actual_start_ns;
    if (!coordinator_.start(context->segment_index, steady_ns, event_ns)) {
        return false;
    }

    SegmentStartEvent start;
    start.session_id = context->session_id;
    start.run_id = context->run_id;
    start.segment_index = context->segment_index;
    start.ts_ns = event_ns;
    start.actual_start_ns = event_ns;
    context->logger->write(model::SegmentStartEventModel(start));
    if (context->dictionary) {
        Monitor::FlushSegmentDictionarySnapshot(
            *context->dictionary, *context->logger, context->session_id);
    }

    retirement_thread_ = std::thread([this] { retirementLoop_(); });
    started_ = true;
    return true;
}

bool SegmentRuntime::service() {
    return coordinator_.service();
}

void SegmentRuntime::noteRows(const uint32_t segment_index,
                              const uint64_t rows,
                              const int64_t committed_steady_ns,
                              const int64_t committed_event_ns) {
    coordinator_.noteRows(segment_index, rows, committed_steady_ns,
                          committed_event_ns);
}

void SegmentRuntime::noteBytes(const uint32_t segment_index,
                               const uint64_t bytes,
                               const int64_t committed_steady_ns,
                               const int64_t committed_event_ns) {
    coordinator_.noteBytes(segment_index, bytes, committed_steady_ns,
                           committed_event_ns);
}

bool SegmentRuntime::cutover_(SegmentBoundaryRequest& boundary) {
    Runtime* const rt = options_.runtime;
    if (!rt) return false;

    const auto retiring = rt->peekSegmentContext();
    if (!retiring ||
        retiring->segment_index != boundary.retiring_segment_index) {
        GFL_LOG_ERROR("[SegmentRuntime] retiring context changed during cutover");
        return false;
    }

    const std::string next_session_id = detail::GenerateSessionId();
    Logger::Options next_options = options_.logger_options;
    next_options.session_id = next_session_id;
    auto next_logger = std::make_shared<Logger>();
    if (!next_logger->open(next_options)) {
        GFL_LOG_ERROR("[SegmentRuntime] failed to open segment ",
                      boundary.retiring_segment_index + 1, " logger");
        return false;
    }

    const uint32_t next_index = boundary.retiring_segment_index + 1;
    auto next_dictionary = std::make_shared<SegmentDictionaryEmitter>();
    const bool published = Monitor::CommitSegmentBoundary(
        [&](const int64_t actual_event_ns,
            const std::vector<ScopeBatchRow>& closes,
            const std::vector<ScopeBatchRow>& opens) {
            // The real boundary is chosen only after the final drain, directly
            // before bootstrap/publication. Setup time never masquerades as
            // segment data time.
            boundary.actual_steady_ns = steadyNowNs();
            boundary.actual_event_ns = actual_event_ns;
            boundary.boundary_delay_ns = (std::max)(
                int64_t{0},
                boundary.actual_steady_ns - boundary.requested_steady_ns);

            std::shared_ptr<const RunPartContext> next_run_part;
            if (boundary.ends_run) {
                const auto& prev = retiring->run_part;
                next_run_part = std::make_shared<const RunPartContext>(
                    prev ? prev->roll_chain_id : retiring->run_id,
                    detail::GenerateSessionId(),
                    prev ? prev->run_id : retiring->run_id,
                    (prev ? prev->part_index : 1u) + 1u,
                    boundary.actual_steady_ns, next_index);
            } else {
                next_run_part = retiring->run_part;
            }
            const std::string next_run_id =
                next_run_part ? next_run_part->run_id : retiring->run_id;
            const uint32_t wire_index =
                next_run_part
                    ? next_index - next_run_part->first_segment_index
                    : next_index;


            InitEvent job_start = options_.init_template;
            job_start.session_id = next_session_id;
            job_start.ts_ns = actual_event_ns;
            job_start.run_id = next_run_id;
            job_start.segment_index = wire_index;
            if (next_run_part) {
                job_start.roll_chain_id = next_run_part->roll_chain_id;
                job_start.previous_run_id = next_run_part->previous_run_id;
                job_start.part_index = next_run_part->part_index;
            }
            next_logger->write(model::InitEventModel(job_start));

            SegmentStartEvent segment_start;
            segment_start.session_id = next_session_id;
            segment_start.run_id = next_run_id;
            segment_start.segment_index = wire_index;
            segment_start.ts_ns = actual_event_ns;
            segment_start.actual_start_ns = actual_event_ns;
            segment_start.previous_session_id = retiring->session_id;
            segment_start.has_requested_boundary = true;
            segment_start.requested_boundary_ns =
                boundary.requested_event_ns;
            segment_start.boundary_delay_ns = boundary.boundary_delay_ns;
            segment_start.deferred_by = boundary.deferred_by;
            next_logger->write(
                model::SegmentStartEventModel(segment_start));

            // Bootstrap is complete before publication. Any ID interned after
            // this snapshot is emitted by the new context's emitter before
            // the referencing batch.
            Monitor::FlushSegmentDictionarySnapshot(
                *next_dictionary, *next_logger, next_session_id);

            // Both halves carry the exact same timestamp and logical scope ID.
            // New continuation rows are bootstrap and therefore precede
            // context publication; old closes remain valid while old writers
            // drain because the retiring logger stays open.
            Monitor::WriteScopeRows(
                *retiring->logger, retiring->session_id, closes);
            Monitor::WriteScopeRows(*next_logger, next_session_id, opens);

            // The backend remains process-live, but each ordinary segment must
            // describe what it actually captured. Emit after continuation
            // closes and before rule/counter deltas, matching the terminal
            // snapshot contract while the retiring context is still active.
            Monitor::EmitSegmentCaptureCapabilities();

            // Snapshot without finishing: the rule state machine, cooldown,
            // rate baseline, and max-window budget remain run-global.
            detail::DeepWindowRules::SnapshotSegment();
            const auto next = std::make_shared<SegmentContext>(
                next_run_id, next_session_id, next_index,
                actual_event_ns, next_logger, next_dictionary, next_run_part);
            return rt->publishSegmentContext(next);
        });
    if (!published) {
        next_logger->close();
        return false;
    }

    enqueueRetirement_({retiring, boundary});
    return true;
}

void SegmentRuntime::enqueueRetirement_(RetiredSegment retired) {
    {
        std::lock_guard lock(retirement_mu_);
        retirement_queue_.push_back(std::move(retired));
    }
    retirement_cv_.notify_one();
}

void SegmentRuntime::retirementLoop_() {
    for (;;) {
        RetiredSegment retired;
        {
            std::unique_lock lock(retirement_mu_);
            retirement_cv_.wait(lock, [this] {
                return retirement_stopping_ || !retirement_queue_.empty();
            });
            if (retirement_queue_.empty()) {
                if (retirement_stopping_) return;
                continue;
            }
            retired = std::move(retirement_queue_.front());
            retirement_queue_.pop_front();
        }
        retire_(std::move(retired));
    }
}

bool SegmentRuntime::awaitWriterDrain_(
    const std::shared_ptr<const SegmentContext>& context,
    const char* const phase) {
    if (!context) return true;
    uint64_t remaining = 0;
    if (context->waitForWriters(
            std::chrono::milliseconds(
                options_.retirement_drain_timeout_ms),
            &remaining)) {
        return true;
    }
    GFL_LOG_ERROR(
        "[SegmentRuntime] writer-drain timeout during ", phase,
        "; run=", context->run_id, " session=", context->session_id,
        " segment=", context->segment_index,
        " active_writers=", remaining,
        " owners={", context->activeWriterSummary(), "}",
        ". The segment is intentionally left incomplete; its logger and "
        "ownership lock remain live until process exit.");
    quarantineUndrainedContext(context);
    return false;
}

bool SegmentRuntime::retire_(RetiredSegment retired) {
    const auto& context = retired.context;
    if (!context || !context->logger) return true;
    if (!awaitWriterDrain_(context, "segment retirement")) return false;

    SegmentEndEvent end;
    end.session_id = context->session_id;
    end.run_id = context->run_id;
    end.segment_index = wireSegmentIndex(*context);
    end.ts_ns = retired.boundary.actual_event_ns;
    end.actual_end_ns = retired.boundary.actual_event_ns;
    end.has_requested_boundary = true;
    end.requested_boundary_ns = retired.boundary.requested_event_ns;
    end.boundary_delay_ns = retired.boundary.boundary_delay_ns;
    end.end_reason = retired.boundary.ends_run
                        ? "rolled"
                        : segmentBoundaryReasonName(retired.boundary.reason);
    end.deferred_by = retired.boundary.deferred_by;
    context->logger->write(model::SegmentEndEventModel(end));

    if (retired.boundary.ends_run) {
        RunEndEvent run_end;
        run_end.session_id = context->session_id;
        run_end.run_id = context->run_id;
        run_end.final_segment_index = wireSegmentIndex(*context);
        run_end.ts_ns = retired.boundary.actual_event_ns;
        run_end.ended_ns = retired.boundary.actual_event_ns;
        run_end.end_reason = "rolled";
        run_end.rollover_reason =
            segmentBoundaryReasonName(retired.boundary.rollover_reason);
        run_end.requested_rollover_ns =
            retired.boundary.requested_rollover_event_ns;
        run_end.actual_rollover_ns =
            retired.boundary.actual_rollover_event_ns;
        context->logger->write(model::RunEndEventModel(run_end));
    }

    writeShutdown_(context, retired.boundary.actual_event_ns);
    context->logger->close();
    return true;
}

void SegmentRuntime::writeShutdown_(
    const std::shared_ptr<const SegmentContext>& context,
    const int64_t ts_ns) const {
    ShutdownEvent shutdown;
    shutdown.pid = options_.init_template.pid;
    shutdown.app = options_.init_template.app;
    shutdown.session_id = context->session_id;
    shutdown.ts_ns = ts_ns;
    context->logger->write(model::ShutdownEventModel(shutdown));
}

void SegmentRuntime::finish(const int64_t ended_ns) {
    std::lock_guard lock(lifecycle_mu_);
    if (finished_) return;
    finished_ = true;
    coordinator_.finish();

    const auto context = options_.runtime
                             ? options_.runtime->sealActiveSegmentContext()
                             : nullptr;
    const bool final_drained =
        !context || awaitWriterDrain_(context, "run finalization");

    // Drain prior segments before publishing run_end. Network delivery can
    // still reorder independent session uploads, so the backend contract must
    // remain order-independent, but the local normal path is deterministic.
    stopRetirementWorker_();

    if (final_drained && context && context->logger) {
        SegmentEndEvent end;
        end.session_id = context->session_id;
        end.run_id = context->run_id;
        end.segment_index = wireSegmentIndex(*context);
        end.ts_ns = ended_ns;
        end.actual_end_ns = ended_ns;
        end.end_reason = "final";
        context->logger->write(model::SegmentEndEventModel(end));

        RunEndEvent run_end;
        run_end.session_id = context->session_id;
        run_end.run_id = context->run_id;
        run_end.final_segment_index = wireSegmentIndex(*context);
        run_end.ts_ns = ended_ns;
        run_end.ended_ns = ended_ns;
        context->logger->write(model::RunEndEventModel(run_end));
        writeShutdown_(context, ended_ns);
        context->logger->close();
    }
}

void SegmentRuntime::stopRetirementWorker_() {
    {
        std::lock_guard lock(retirement_mu_);
        retirement_stopping_ = true;
    }
    retirement_cv_.notify_all();
    if (retirement_thread_.joinable()) retirement_thread_.join();
}

}  // namespace gpufl

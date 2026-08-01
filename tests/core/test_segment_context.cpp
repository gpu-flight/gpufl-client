#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "gpufl/core/common.hpp"
#include "gpufl/core/dictionary_manager.hpp"
#include "gpufl/core/events.hpp"
#include "gpufl/core/logger/log_sink.hpp"
#include "gpufl/core/logger/logger.hpp"
#include "gpufl/core/logger/session_ownership.hpp"
#include "gpufl/core/monitor_batch_manager.hpp"
#include "gpufl/core/model/lifecycle_model.hpp"
#include "gpufl/core/runtime.hpp"
#include "gpufl/core/segment_runtime.hpp"

namespace {

namespace fs = std::filesystem;

class RecordingSink final : public gpufl::ILogSink {
   public:
    explicit RecordingSink(
        std::shared_ptr<std::vector<std::string>> lines)
        : lines_(std::move(lines)) {}

    void write(gpufl::Channel, std::string_view json) override {
        lines_->emplace_back(json);
    }
    void close() override {}

   private:
    std::shared_ptr<std::vector<std::string>> lines_;
};

std::shared_ptr<const gpufl::SegmentContext> makeContext(
    uint32_t index) {
    return std::make_shared<gpufl::SegmentContext>(
        "12345678-1234-4123-8123-123456789abc",
        "session-" + std::to_string(index), index,
        1000 + static_cast<int64_t>(index),
        std::make_shared<gpufl::Logger>());
}

TEST(SegmentContextTest, RefusesAnUnusablePublication) {
    gpufl::Runtime runtime;
    EXPECT_FALSE(runtime.hasSegmentContext());
    EXPECT_FALSE(runtime.publishSegmentContext(nullptr));
    EXPECT_EQ(runtime.acquireSegmentContext(), nullptr);

    auto missing_logger = std::make_shared<gpufl::SegmentContext>(
        "run", "session", 0, 1, nullptr);
    EXPECT_FALSE(runtime.publishSegmentContext(missing_logger));
    EXPECT_EQ(runtime.acquireSegmentContext(), nullptr);
    ASSERT_TRUE(runtime.publishSegmentContext(makeContext(0)));
    EXPECT_TRUE(runtime.hasSegmentContext());
}

TEST(SegmentContextTest, OldLeaseRemainsImmutableAcrossPublication) {
    gpufl::Runtime runtime;
    const auto first = makeContext(0);
    ASSERT_TRUE(runtime.publishSegmentContext(first));

    const auto old_lease = runtime.acquireSegmentContext();
    ASSERT_TRUE(old_lease);
    ASSERT_TRUE(runtime.publishSegmentContext(makeContext(1)));

    const auto current = runtime.acquireSegmentContext();
    ASSERT_TRUE(current);
    EXPECT_EQ(current->segment_index, 1u);
    EXPECT_EQ(current->session_id, "session-1");
    EXPECT_EQ(old_lease->segment_index, 0u);
    EXPECT_EQ(old_lease->session_id, "session-0");
    EXPECT_NE(old_lease->logger, current->logger);
}

TEST(SegmentContextTest, ConcurrentReadersSeeACompletePublishedContext) {
    gpufl::Runtime runtime;
    ASSERT_TRUE(runtime.publishSegmentContext(makeContext(0)));

    std::atomic<bool> stop{false};
    std::atomic<bool> inconsistent{false};
    std::vector<std::thread> readers;
    for (int thread = 0; thread < 4; ++thread) {
        readers.emplace_back([&] {
            while (!stop.load(std::memory_order_acquire)) {
                const auto context = runtime.acquireSegmentContext();
                if (!context || !context->logger ||
                    context->session_id !=
                        "session-" + std::to_string(context->segment_index) ||
                    context->actual_start_ns !=
                        1000 + static_cast<int64_t>(context->segment_index)) {
                    inconsistent.store(true, std::memory_order_release);
                    return;
                }
            }
        });
    }

    for (uint32_t index = 1; index <= 1000; ++index) {
        ASSERT_TRUE(runtime.publishSegmentContext(makeContext(index)));
    }
    stop.store(true, std::memory_order_release);
    for (auto& reader : readers) reader.join();

    EXPECT_FALSE(inconsistent.load(std::memory_order_acquire));
    EXPECT_EQ(runtime.acquireSegmentContext()->segment_index, 1000u);
}

TEST(SegmentContextTest, DictionaryEmissionIsIndependentPerSegment) {
    gpufl::DictionaryManager registry;
    gpufl::SegmentDictionaryEmitter first;
    gpufl::SegmentDictionaryEmitter second;
    gpufl::Logger logger;
    auto lines = std::make_shared<std::vector<std::string>>();
    logger.addSink(std::make_unique<RecordingSink>(lines));

    EXPECT_EQ(registry.internKernel("kernel_a"), 1u);
    first.flush(registry, logger, "s0");
    ASSERT_EQ(lines->size(), 1u);
    EXPECT_NE(lines->back().find("\"kernel_dict\":{\"1\":\"kernel_a\"}"),
              std::string::npos);
    first.flush(registry, logger, "s0");
    EXPECT_EQ(lines->size(), 1u);

    second.flush(registry, logger, "s1");
    ASSERT_EQ(lines->size(), 2u);
    EXPECT_NE(lines->back().find("\"session_id\":\"s1\""),
              std::string::npos);

    EXPECT_EQ(registry.internKernel("kernel_b"), 2u);
    first.flush(registry, logger, "s0");
    second.flush(registry, logger, "s1");
    ASSERT_EQ(lines->size(), 4u);
    EXPECT_NE((*lines)[2].find("\"2\":\"kernel_b\""), std::string::npos);
    EXPECT_NE((*lines)[3].find("\"2\":\"kernel_b\""), std::string::npos);
}

TEST(SegmentContextTest, PendingKernelDetailUsesTheFlushSegmentIdentity) {
    gpufl::Runtime runtime;
    runtime.session_id = "initial-process-session";
    auto lines = std::make_shared<std::vector<std::string>>();
    auto logger = std::make_shared<gpufl::Logger>();
    logger->addSink(std::make_unique<RecordingSink>(lines));
    ASSERT_TRUE(runtime.publishSegmentContext(
        std::make_shared<gpufl::SegmentContext>(
            "12345678-1234-4123-8123-123456789abc",
            "current-segment-session", 1, 1000, logger)));

    gpufl::detail::MonitorBatchManager batches;
    batches.bindFlushRuntime(&runtime);
    gpufl::KernelBatchRow kernel;
    kernel.kernel_id = 1;
    gpufl::KernelDetailRow detail;
    detail.session_id = "stale-previous-segment";
    detail.corr_id = 7;
    ASSERT_FALSE(batches.pushKernel(kernel, &detail));
    batches.flushAll(
        gpufl::detail::MonitorBatchManager::FlushMode::Full);

    const auto detail_line = std::find_if(
        lines->begin(), lines->end(), [](const std::string& line) {
            return line.find("\"type\":\"kernel_detail\"") !=
                   std::string::npos;
        });
    ASSERT_NE(detail_line, lines->end());
    EXPECT_NE(detail_line->find(
                  "\"session_id\":\"current-segment-session\""),
              std::string::npos);
    EXPECT_EQ(detail_line->find("stale-previous-segment"),
              std::string::npos);
    EXPECT_EQ(detail_line->find("initial-process-session"),
              std::string::npos);
}

TEST(SegmentContextTest, ProductionRuntimePublishesAndRetiresTwoSegments) {
    const fs::path root =
        fs::temp_directory_path() /
        ("gpufl_segment_runtime_" + std::to_string(gpufl::detail::GetPid()));
    std::error_code ec;
    fs::remove_all(root, ec);

    gpufl::Runtime runtime;
    runtime.app_name = "segment-test";
    runtime.run_id = "12345678-1234-4123-8123-123456789abc";
    runtime.session_id = "segment-zero";
    runtime.logger = std::make_shared<gpufl::Logger>();

    gpufl::Logger::Options logger_options;
    logger_options.base_path = root.string();
    logger_options.session_id = runtime.session_id;
    logger_options.compress_rotated = false;
    logger_options.max_spool_bytes = 0;
    logger_options.min_free_bytes = 0;
    ASSERT_TRUE(runtime.logger->open(logger_options));

    auto dictionary = std::make_shared<gpufl::SegmentDictionaryEmitter>();
    ASSERT_TRUE(runtime.publishSegmentContext(
        std::make_shared<gpufl::SegmentContext>(
            runtime.run_id, runtime.session_id, 0,
            gpufl::detail::GetTimestampNs(), runtime.logger, dictionary)));

    gpufl::InitEvent init;
    init.pid = gpufl::detail::GetPid();
    init.app = runtime.app_name;
    init.session_id = runtime.session_id;
    init.log_path = root.string();
    init.ts_ns = gpufl::detail::GetTimestampNs();
    init.run_id = runtime.run_id;
    init.segment_index = 0;
    runtime.logger->write(gpufl::model::InitEventModel(init));

    gpufl::SegmentRuntime::Options options;
    options.runtime = &runtime;
    options.logger_options = logger_options;
    options.init_template = init;
    options.segment_max_rows = 1;
    auto segmented =
        std::make_shared<gpufl::SegmentRuntime>(std::move(options));
    runtime.segment_runtime = segmented;
    ASSERT_TRUE(segmented->start());

    // Pin one old-context writer across publication. Retirement must not write
    // segment_end or close this sink until the explicit lease is released.
    auto held_old_writer = runtime.acquireSegmentContext();
    ASSERT_TRUE(held_old_writer);
    // A control-plane snapshot is not a writer. Keeping this shared owner must
    // not pin retirement once the real writer lease drains.
    const auto unrelated_owner = runtime.peekSegmentContext();
    ASSERT_TRUE(unrelated_owner);

    const int64_t steady_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count();
    segmented->noteRows(0, 1, steady_ns,
                        gpufl::detail::GetTimestampNs());
    ASSERT_TRUE(segmented->service());
    ASSERT_EQ(runtime.acquireSegmentContext()->segment_index, 1u);
    const std::string next_session =
        runtime.acquireSegmentContext()->session_id;
    EXPECT_NE(next_session, runtime.session_id);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_FALSE(fs::exists(root / runtime.session_id / "device.1.log"));
    held_old_writer.reset();

    const auto retirement_deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!fs::exists(root / runtime.session_id / "device.1.log") &&
           std::chrono::steady_clock::now() < retirement_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    ASSERT_TRUE(fs::exists(root / runtime.session_id / "device.1.log"));

    segmented->finish(gpufl::detail::GetTimestampNs());
    runtime.segment_runtime.reset();
    segmented.reset();

    EXPECT_TRUE(fs::exists(root / runtime.session_id / "device.1.log"));
    EXPECT_TRUE(fs::exists(root / next_session / "device.1.log"));
    const auto read = [](const fs::path& path) {
        std::ifstream input(path);
        return std::string(std::istreambuf_iterator<char>(input),
                           std::istreambuf_iterator<char>());
    };
    const std::string first =
        read(root / runtime.session_id / "device.1.log");
    const std::string second =
        read(root / next_session / "device.1.log");
    const auto first_job = first.find("\"type\":\"job_start\"");
    const auto first_start = first.find("\"type\":\"segment_start\"");
    const auto first_end = first.find("\"type\":\"segment_end\"");
    const auto first_shutdown = first.find("\"type\":\"shutdown\"");
    ASSERT_NE(first_job, std::string::npos);
    ASSERT_NE(first_start, std::string::npos);
    ASSERT_NE(first_end, std::string::npos);
    ASSERT_NE(first_shutdown, std::string::npos);
    EXPECT_LT(first_job, first_start);
    EXPECT_LT(first_start, first_end);
    EXPECT_LT(first_end, first_shutdown);

    const auto second_job = second.find("\"type\":\"job_start\"");
    const auto second_start = second.find("\"type\":\"segment_start\"");
    const auto second_end = second.find("\"type\":\"segment_end\"");
    const auto run_end = second.find("\"type\":\"run_end\"");
    const auto second_shutdown = second.find("\"type\":\"shutdown\"");
    ASSERT_NE(second_job, std::string::npos);
    ASSERT_NE(second_start, std::string::npos);
    ASSERT_NE(second_end, std::string::npos);
    ASSERT_NE(run_end, std::string::npos);
    ASSERT_NE(second_shutdown, std::string::npos);
    EXPECT_LT(second_job, second_start);
    EXPECT_LT(second_start, second_end);
    EXPECT_LT(second_end, run_end);
    EXPECT_LT(run_end, second_shutdown);

    std::string lock_error;
    EXPECT_TRUE(gpufl::SessionOwnershipLock::tryAcquire(
        root / runtime.session_id, &lock_error));
    EXPECT_TRUE(gpufl::SessionOwnershipLock::tryAcquire(
        root / next_session, &lock_error));
    fs::remove_all(root, ec);
}

TEST(SegmentContextTest, ProductionRuntimeRollsToANewRunPart) {
    const fs::path root =
        fs::temp_directory_path() /
        ("gpufl_segment_roll_" + std::to_string(gpufl::detail::GetPid()));
    std::error_code ec;
    fs::remove_all(root, ec);

    gpufl::Runtime runtime;
    runtime.app_name = "roll-test";
    runtime.run_id = "12345678-1234-4123-8123-123456789abc";
    runtime.session_id = "part1-seg0";
    runtime.logger = std::make_shared<gpufl::Logger>();

    gpufl::Logger::Options logger_options;
    logger_options.base_path = root.string();
    logger_options.session_id = runtime.session_id;
    logger_options.compress_rotated = false;
    logger_options.max_spool_bytes = 0;
    logger_options.min_free_bytes = 0;
    ASSERT_TRUE(runtime.logger->open(logger_options));

    // Part 1 identity. gpufl.cpp mints this in production (2c-ii-C); the test
    // constructs it so the runtime has a chain to extend.
    auto part1 = std::make_shared<const gpufl::RunPartContext>(
        runtime.run_id, runtime.run_id, std::string(), 1u,
        gpufl::detail::GetTimestampNs(), 0u);
    auto dictionary = std::make_shared<gpufl::SegmentDictionaryEmitter>();
    ASSERT_TRUE(runtime.publishSegmentContext(
        std::make_shared<gpufl::SegmentContext>(
            runtime.run_id, runtime.session_id, 0,
            gpufl::detail::GetTimestampNs(), runtime.logger, dictionary,
            part1)));

    gpufl::InitEvent init;
    init.pid = gpufl::detail::GetPid();
    init.app = runtime.app_name;
    init.session_id = runtime.session_id;
    init.ts_ns = gpufl::detail::GetTimestampNs();
    init.run_id = runtime.run_id;
    init.segment_index = 0;
    runtime.logger->write(gpufl::model::InitEventModel(init));

    gpufl::SegmentRuntime::Options options;
    options.runtime = &runtime;
    options.logger_options = logger_options;
    options.init_template = init;
    options.segment_max_rows = 1;     // arms the segment boundary
    options.run_roll_max_bytes = 1;   // arms the roll that rides it
    auto segmented =
        std::make_shared<gpufl::SegmentRuntime>(std::move(options));
    runtime.segment_runtime = segmented;
    ASSERT_TRUE(segmented->start());

    const int64_t steady_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count();
    segmented->noteRows(0, 1, steady_ns, gpufl::detail::GetTimestampNs());
    segmented->noteBytes(0, 1, steady_ns, gpufl::detail::GetTimestampNs());
    ASSERT_TRUE(segmented->service());

    // The new part reset its wire index to 0 but advanced the chain to part 2.
    // peek, not acquire: a write lease here would pin part 2, so finish() would
    // time out draining it and never write its log.
    const auto current = runtime.peekSegmentContext();
    ASSERT_TRUE(current->run_part);
    EXPECT_EQ(current->run_part->part_index, 2u);
    EXPECT_EQ(current->run_part->previous_run_id, runtime.run_id);
    EXPECT_EQ(gpufl::wireSegmentIndex(*current), 0u);
    EXPECT_EQ(current->segment_index, 1u)
        << "the internal sequence stays monotonic";
    const std::string part2_session = current->session_id;
    EXPECT_NE(current->run_part->run_id, runtime.run_id)
        << "a roll mints a new run_id";


    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!fs::exists(root / runtime.session_id / "device.1.log") &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    segmented->finish(gpufl::detail::GetTimestampNs());
    runtime.segment_runtime.reset();
    segmented.reset();

    const auto read = [](const fs::path& path) {
        std::ifstream input(path);
        return std::string(std::istreambuf_iterator(input),
                           std::istreambuf_iterator<char>());
    };
    const std::string part1_log =
        read(root / runtime.session_id / "device.1.log");
    const std::string part2_log = read(root / part2_session / "device.1.log");

    // Part 1 retired as a roll: segment_end(rolled) then run_end(rolled).
    const auto p1_seg_end = part1_log.find("\"type\":\"segment_end\"");
    const auto p1_run_end = part1_log.find("\"type\":\"run_end\"");
    ASSERT_NE(p1_seg_end, std::string::npos);
    ASSERT_NE(p1_run_end, std::string::npos) << part1_log;
    EXPECT_LT(p1_seg_end, p1_run_end);
    EXPECT_NE(part1_log.find("\"end_reason\":\"rolled\""), std::string::npos);
    EXPECT_NE(part1_log.find("\"rollover_reason\":\"serialized_bytes\""),
              std::string::npos);

    // Part 2 opened the chain's next link at wire segment 0.
    EXPECT_NE(part2_log.find("\"part_index\":2"), std::string::npos) << part2_log;
    EXPECT_NE(part2_log.find("\"roll_chain_id\":\"" + runtime.run_id + "\""),
              std::string::npos);
    EXPECT_NE(part2_log.find("\"previous_run_id\":\"" + runtime.run_id + "\""),
              std::string::npos);
    EXPECT_NE(part2_log.find("\"segment_index\":0"), std::string::npos);

    fs::remove_all(root, ec);
}

TEST(SegmentContextTest, LeakedWriterTimesOutWithoutPublishingFalseFinality) {
    gpufl::Runtime runtime;
    runtime.run_id = "12345678-1234-4123-8123-123456789abc";
    runtime.session_id = "timeout-segment";
    auto lines = std::make_shared<std::vector<std::string>>();
    auto logger = std::make_shared<gpufl::Logger>();
    logger->addSink(std::make_unique<RecordingSink>(lines));
    runtime.logger = logger;
    ASSERT_TRUE(runtime.publishSegmentContext(
        std::make_shared<gpufl::SegmentContext>(
            runtime.run_id, runtime.session_id, 0,
            gpufl::detail::GetTimestampNs(), logger)));

    gpufl::InitEvent init;
    init.pid = gpufl::detail::GetPid();
    init.app = "timeout-test";
    init.session_id = runtime.session_id;
    init.run_id = runtime.run_id;
    init.segment_index = 0;

    gpufl::SegmentRuntime::Options options;
    options.runtime = &runtime;
    options.init_template = init;
    options.segment_max_rows = 1;
    options.retirement_drain_timeout_ms = 20;
    auto segmented =
        std::make_shared<gpufl::SegmentRuntime>(std::move(options));
    ASSERT_TRUE(segmented->start());

    auto leaked_writer = runtime.acquireSegmentContext();
    ASSERT_TRUE(leaked_writer);
    auto finishing = std::async(std::launch::async, [&] {
        segmented->finish(gpufl::detail::GetTimestampNs());
    });
    EXPECT_EQ(finishing.wait_for(std::chrono::milliseconds(500)),
              std::future_status::ready);
    finishing.get();

    const auto contains = [&](const char* type) {
        return std::any_of(lines->begin(), lines->end(),
                           [type](const std::string& line) {
                               return line.find(type) != std::string::npos;
                           });
    };
    EXPECT_FALSE(contains("\"type\":\"segment_end\""));
    EXPECT_FALSE(contains("\"type\":\"run_end\""));
    EXPECT_FALSE(contains("\"type\":\"shutdown\""));

    leaked_writer.reset();
    logger->close();
    segmented.reset();
}

TEST(RunPartContextTest, CarriesImmutableChainIdentity) {
    const auto part = std::make_shared<const gpufl::RunPartContext>(
        "chain-abc", "run-1", /*previous=*/std::string(), /*part_index=*/1u,
        /*run_started_mono_ns=*/5000);
    EXPECT_EQ(part->roll_chain_id, "chain-abc");
    EXPECT_EQ(part->run_id, "run-1");
    EXPECT_TRUE(part->previous_run_id.empty()) << "first part has no predecessor";
    EXPECT_EQ(part->part_index, 1u) << "part numbering is 1-based";
    EXPECT_EQ(part->run_started_mono_ns, 5000);
}

TEST(RunPartContextTest, TheOrdinaryPathHasNoRunPart) {
    // Everything built through the existing 5/6-arg constructor stays on the
    // non-rolled path: run_part is null and nothing reads chain identity.
    const auto context = makeContext(0);
    EXPECT_EQ(context->run_part, nullptr);
}

TEST(RunPartContextTest, AnOrdinaryCutSharesThePartWhileARollReplacesIt) {
    const auto logger = std::make_shared<gpufl::Logger>();
    const auto part1 = std::make_shared<const gpufl::RunPartContext>(
        "chain-abc", "run-1", std::string(), 1u, 5000);

    // Two segments of the SAME part share one RunPartContext instance - the
    // structure the runtime will rely on when an ordinary cut keeps identity.
    const auto seg0 = std::make_shared<gpufl::SegmentContext>(
        "run-1", "session-a", 0u, 1000, logger, nullptr, part1);
    const auto seg1 = std::make_shared<gpufl::SegmentContext>(
        "run-1", "session-b", 1u, 2000, logger, nullptr, part1);
    EXPECT_EQ(seg0->run_part.get(), seg1->run_part.get())
        << "an ordinary cut retains the same run part";

    // A roll mints a new part: fresh run_id, previous_run_id set, part_index++.
    const auto part2 = std::make_shared<const gpufl::RunPartContext>(
        "chain-abc", "run-2", "run-1", 2u, 9000);
    const auto rolled = std::make_shared<gpufl::SegmentContext>(
        "run-2", "session-c", 0u, 9000, logger, nullptr, part2);

    EXPECT_NE(rolled->run_part.get(), seg1->run_part.get());
    EXPECT_EQ(rolled->run_part->roll_chain_id, part1->roll_chain_id)
        << "same chain across the roll";
    EXPECT_EQ(rolled->run_part->previous_run_id, "run-1");
    EXPECT_EQ(rolled->run_part->part_index, 2u);
    EXPECT_EQ(rolled->segment_index, 0u) << "segment numbering restarts";
}

}  // namespace

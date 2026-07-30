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

}  // namespace

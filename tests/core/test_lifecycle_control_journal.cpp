#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "gpufl/core/logger/lifecycle_control_journal.hpp"
#include "gpufl/core/logger/logger.hpp"
#include "gpufl/core/model/lifecycle_model.hpp"

namespace fs = std::filesystem;

namespace {

class LifecycleControlJournalTest : public ::testing::Test {
   protected:
    void SetUp() override {
        const auto* info =
            ::testing::UnitTest::GetInstance()->current_test_info();
        base_ = fs::temp_directory_path() /
                (std::string("gpufl_control_journal_") + info->name());
        fs::remove_all(base_);
        fs::create_directories(base_);
    }

    void TearDown() override {
        std::error_code ec;
        fs::remove_all(base_, ec);
    }

    [[nodiscard]] fs::path sessionDir() const { return base_ / "s1"; }

    static std::string read(const fs::path& path) {
        std::ifstream in(path, std::ios::binary);
        return {std::istreambuf_iterator<char>(in),
                std::istreambuf_iterator<char>()};
    }

    fs::path base_;
};

TEST_F(LifecycleControlJournalTest,
       WritesAtomicImmutableEnvelopesAndRespectsAcknowledgementTombstones) {
    gpufl::LifecycleControlJournal journal(sessionDir(), "s1");
    ASSERT_TRUE(journal.append(
        "segment_end",
        R"({"version":1,"type":"segment_end","session_id":"s1"})"));
    const auto first = gpufl::LifecycleControlJournal::controlPath(sessionDir(), 1);
    ASSERT_TRUE(fs::is_regular_file(first));
    const std::string first_json = read(first);
    EXPECT_NE(first_json.find("\"schema_version\":" + std::to_string(
                                gpufl::LifecycleControlJournal::kEnvelopeSchemaVersion)),
              std::string::npos);
    EXPECT_NE(first_json.find(R"("control_sequence":1)"), std::string::npos);
    EXPECT_NE(first_json.find(R"("event_type":"segment_end")"),
              std::string::npos);
    EXPECT_NE(first_json.find("\"payload_json\":\"{\\\"version\\\":1"),
              std::string::npos);
    EXPECT_EQ(first_json.find(".part."), std::string::npos);

    ASSERT_TRUE(fs::remove(first));
    {
        std::ofstream ack(
            gpufl::LifecycleControlJournal::acknowledgementPath(sessionDir(), 1));
        ack << "{\"acknowledged\":true}\n";
    }

    gpufl::LifecycleControlJournal restarted(sessionDir(), "s1");
    ASSERT_TRUE(restarted.append(
        "run_end",
        R"({"version":1,"type":"run_end","session_id":"s1"})"));
    EXPECT_TRUE(fs::is_regular_file(
        gpufl::LifecycleControlJournal::controlPath(sessionDir(), 2)));
}

TEST_F(LifecycleControlJournalTest, RejectsNonLifecycleAndOversizedRecords) {
    gpufl::LifecycleControlJournal journal(sessionDir(), "s1");
    EXPECT_FALSE(journal.append("kernel_event", R"({"type":"kernel_event"})"));
    EXPECT_FALSE(journal.append(
        "segment_start",
        R"({"version":1,"type":"segment_start","session_id":"other"})"));
    EXPECT_FALSE(journal.append(
        "segment_start",
        std::string(gpufl::LifecycleControlJournal::kMaxPayloadBytes + 1, 'x')));
    EXPECT_FALSE(fs::exists(sessionDir()));
}

TEST_F(LifecycleControlJournalTest,
       LoggerWritesOnlyLifecycleModelsWhenCapabilityWasExplicitlyEnabled) {
    gpufl::Logger::Options options;
    options.base_path = base_.string();
    options.session_id = "s1";
    options.rotate_bytes = 0;
    options.lifecycle_control_journal_enabled = true;

    gpufl::Logger logger;
    ASSERT_TRUE(logger.open(options));

    gpufl::SegmentStartEvent event;
    event.session_id = "s1";
    event.run_id = "run-1";
    event.actual_start_ns = 10;
    event.ts_ns = 10;
    logger.write(gpufl::model::SegmentStartEventModel(event));

    const auto control =
        gpufl::LifecycleControlJournal::controlPath(sessionDir(), 1);
    ASSERT_TRUE(fs::is_regular_file(control));
    EXPECT_NE(read(control).find(R"("event_type":"segment_start")"),
              std::string::npos);
}

TEST_F(LifecycleControlJournalTest,
       LoggerLeavesNoControlFilesWithoutConfirmedAgentCapability) {
    gpufl::Logger::Options options;
    options.base_path = base_.string();
    options.session_id = "s1";
    options.rotate_bytes = 0;

    gpufl::Logger logger;
    ASSERT_TRUE(logger.open(options));
    gpufl::ShutdownEvent event;
    event.session_id = "s1";
    logger.write(gpufl::model::ShutdownEventModel(event));

    EXPECT_FALSE(fs::exists(
        gpufl::LifecycleControlJournal::controlPath(sessionDir(), 1)));
}

}  // namespace

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

#include "bootstrap_status.hpp"

namespace fs = std::filesystem;
using gpufl::launcher::BootstrapStatus;
using gpufl::launcher::InitWatchdog;
using gpufl::launcher::ReadBootstrapStatus;

namespace {

class BootstrapStatusTest : public ::testing::Test {
   protected:
    void SetUp() override {
        dir_ = fs::temp_directory_path() /
               ("gpufl-bootstrap-test-" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(dir_);
    }
    void TearDown() override { fs::remove_all(dir_); }

    void Write(long pid, const std::string& text) {
        std::ofstream(gpufl::launcher::BootstrapStatusPath(dir_, pid)) << text;
    }

    static std::string Record(const std::string& nonce, long pid, int seq,
                              const std::string& lifecycle, int64_t deadline) {
        return "{\"schema\":1,\"nonce\":\"" + nonce + "\",\"pid\":" + std::to_string(pid) +
               ",\"seq\":" + std::to_string(seq) + ",\"lifecycle\":\"" + lifecycle +
               "\",\"owner\":\"gate\",\"readiness\":\"pending\",\"reason\":\"\"," +
               "\"coverage_notes\":[\"unsupported_first_touch\"]," +
               "\"deadline_monotonic_ns\":" + std::to_string(deadline) + "}";
    }

    fs::path dir_;
};

BootstrapStatus Status(int seq, const std::string& lifecycle, int64_t deadline) {
    BootstrapStatus s;
    s.valid = true;
    s.seq = seq;
    s.lifecycle = lifecycle;
    s.readiness = "pending";
    s.deadline_monotonic_ns = deadline;
    return s;
}

}  // namespace

TEST_F(BootstrapStatusTest, ReadsOnlyThisChildsOwnRecord) {
    Write(42, Record("n1", 42, 3, "capture_initializing", 1000));
    const BootstrapStatus s = ReadBootstrapStatus(dir_, 42, "n1");
    ASSERT_TRUE(s.valid) << s.error;
    EXPECT_EQ(s.seq, 3);
    EXPECT_EQ(s.lifecycle, "capture_initializing");
    EXPECT_EQ(s.deadline_monotonic_ns, 1000);
    EXPECT_EQ(s.coverage_notes.count("unsupported_first_touch"), 1u);

    EXPECT_EQ(ReadBootstrapStatus(dir_, 42, "other").error, "foreign_nonce");
    Write(43, Record("n1", 42, 3, "finished", 1000));
    EXPECT_EQ(ReadBootstrapStatus(dir_, 43, "n1").error, "foreign_pid");
    EXPECT_EQ(ReadBootstrapStatus(dir_, 44, "n1").error, "missing");
    Write(45, "{not json");
    EXPECT_EQ(ReadBootstrapStatus(dir_, 45, "n1").error, "malformed");
    Write(46, std::string(5000, ' '));
    EXPECT_EQ(ReadBootstrapStatus(dir_, 46, "n1").error, "oversized");
}

TEST(InitWatchdog, KeepsTheFirstDeadlineAndFiresAfterTheGrace) {
    InitWatchdog watchdog(/*grace_ns=*/100);
    EXPECT_FALSE(watchdog.Observe(Status(1, "not_requested", 0), 10'000));
    EXPECT_FALSE(watchdog.armed());

    EXPECT_FALSE(watchdog.Observe(Status(2, "driver_initializing", 1'000), 500));
    EXPECT_TRUE(watchdog.armed());
    // A later record cannot renew the budget.
    EXPECT_FALSE(watchdog.Observe(Status(3, "capture_initializing", 900'000), 1'100));
    EXPECT_TRUE(watchdog.Observe(Status(3, "capture_initializing", 900'000), 1'101));
}

TEST(InitWatchdog, DisarmsWhenEveryParticipantReturnedAndIgnoresStaleRecords) {
    InitWatchdog watchdog(0);
    EXPECT_FALSE(watchdog.Observe(Status(5, "driver_initializing", 1'000), 0));
    EXPECT_FALSE(watchdog.Observe(Status(9, "finished", 1'000), 2'000));
    EXPECT_TRUE(watchdog.disarmed());
    // An older record after Finished changes nothing.
    EXPECT_FALSE(watchdog.Observe(Status(6, "capture_initializing", 1'000), 5'000));
    EXPECT_EQ(watchdog.last().lifecycle, "finished");

    // An unreadable file never arms or disarms.
    InitWatchdog unread(0);
    EXPECT_FALSE(unread.Observe(BootstrapStatus{}, 1'000'000));
    EXPECT_FALSE(unread.seen_valid());
}

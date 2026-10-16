#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

#include "bootstrap_status.hpp"

#if defined(__linux__)
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "gpufl/inject/capture_gate.hpp"
#endif

namespace fs = std::filesystem;
using gpufl::launcher::BootstrapStatus;
using gpufl::launcher::CombineFinalBootstrap;
using gpufl::launcher::InitWatchdog;
using gpufl::launcher::ReadBootstrapStatus;

namespace {

// A record in the library's exact shape (capture_gate.cpp GateStatusJson).
std::string Participant(bool expected, bool returned, bool ok) {
    const auto b = [](bool v) { return v ? std::string("true") : std::string("false"); };
    return "{\"expected\":" + b(expected) + ",\"returned\":" + b(returned) +
           ",\"ok\":" + b(ok) + "}";
}

std::string Record(const std::string& nonce, long pid, int seq,
                   const std::string& lifecycle, const std::string& readiness,
                   const std::string& reason = "", int64_t deadline = 1000,
                   const std::string& owner = "gate") {
    // Gate-owned participants consistent with the lifecycle and readiness.
    const bool finished = lifecycle == "finished";
    const std::string trigger = Participant(true, finished, readiness == "ready");
    const std::string worker = Participant(true, finished, readiness == "ready");
    const std::string handshake = Participant(false, false, false);
    return "{\"schema\":1,\"nonce\":\"" + nonce + "\",\"pid\":" + std::to_string(pid) +
           ",\"seq\":" + std::to_string(seq) + ",\"lifecycle\":\"" + lifecycle +
           "\",\"owner\":\"" + owner + "\",\"readiness\":\"" + readiness +
           "\",\"reason\":\"" + reason + "\"," +
           "\"coverage_notes\":[\"unsupported_first_touch\"]," +
           "\"deadline_monotonic_ns\":" + std::to_string(deadline) +
           ",\"trigger\":" + trigger + ",\"worker\":" + worker +
           ",\"handshake\":" + handshake +
           ",\"callbacks_seen\":1,\"handshake_reentries\":0," +
           "\"wait_config_invalid\":false,\"protocol\":\"2\"}\n";
}

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
    BootstrapStatus Read(long pid, const std::string& nonce = "n1") {
        return ReadBootstrapStatus(dir_, pid, nonce);
    }

    fs::path dir_;
};

BootstrapStatus Status(int seq, const std::string& lifecycle, int64_t deadline) {
    BootstrapStatus s;
    s.valid = true;
    s.seq = seq;
    s.lifecycle = lifecycle;
    s.readiness = lifecycle == "finished" ? "ready" : "pending";
    s.deadline_monotonic_ns = deadline;
    return s;
}

}  // namespace

TEST_F(BootstrapStatusTest, ReadsOnlyThisChildsOwnConsistentRecord) {
    Write(42, Record("n1", 42, 3, "capture_initializing", "pending"));
    const BootstrapStatus s = Read(42);
    ASSERT_TRUE(s.valid) << s.error;
    EXPECT_EQ(s.seq, 3);
    EXPECT_EQ(s.lifecycle, "capture_initializing");
    EXPECT_EQ(s.deadline_monotonic_ns, 1000);
    EXPECT_EQ(s.coverage_notes.count("unsupported_first_touch"), 1u);

    EXPECT_EQ(Read(42, "other").error, "foreign_nonce");
    Write(43, Record("n1", 42, 3, "finished", "ready"));
    EXPECT_EQ(Read(43).error, "foreign_pid");
    EXPECT_EQ(Read(44).error, "missing");
    Write(45, "{not json");
    EXPECT_EQ(Read(45).error, "malformed");
    Write(46, std::string(5000, ' '));
    EXPECT_EQ(Read(46).error, "oversized");
    Write(47, "{\"schema\":1,\"nonce\":\"n1\",\"pid\":47,\"lifecycle\":\"finished\","
              "\"readiness\":\"ready\"}");
    EXPECT_EQ(Read(47).error, "malformed");  // fields missing
}

TEST_F(BootstrapStatusTest, RefusesStatesTheLibraryCannotBeIn) {
    Write(50, Record("n1", 50, 2, "finished", "pending"));
    EXPECT_EQ(Read(50).error, "inconsistent");
    Write(51, Record("n1", 51, 2, "capture_initializing", "ready"));
    EXPECT_EQ(Read(51).error, "inconsistent");
    Write(52, Record("n1", 52, 2, "finished", "failed"));  // failed without a reason
    EXPECT_EQ(Read(52).error, "inconsistent");
    Write(53, Record("n1", 53, 2, "driver_ready", "pending", "", 0));  // no deadline
    EXPECT_EQ(Read(53).error, "inconsistent");
    Write(54, Record("n1", 54, 2, "sleeping", "pending"));
    EXPECT_EQ(Read(54).error, "inconsistent");
    Write(55, Record("n1", 55, 2, "finished", "failed", "capture_wait_timeout"));
    EXPECT_TRUE(Read(55).valid);
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

TEST(CombineFinalBootstrap, AFinalReadThatDoesNotConfirmIsReportedNotHidden) {
    InitWatchdog watchdog(0);
    watchdog.Observe(Status(9, "finished", 1'000), 0);

    auto confirmed = CombineFinalBootstrap(watchdog, Status(9, "finished", 1'000));
    EXPECT_TRUE(confirmed.final_error.empty());

    for (const char* error : {"missing", "malformed", "foreign_nonce"}) {
        BootstrapStatus lost;
        lost.error = error;
        const auto observed = CombineFinalBootstrap(watchdog, lost);
        EXPECT_EQ(observed.final_error, error);
        EXPECT_EQ(observed.accepted.seq, 9);  // history, flagged as such
    }
    EXPECT_EQ(CombineFinalBootstrap(watchdog, Status(4, "driver_ready", 1'000)).final_error,
              "stale_final_seq");

    InitWatchdog never(0);
    BootstrapStatus missing;
    missing.error = "missing";
    const auto none = CombineFinalBootstrap(never, missing);
    EXPECT_FALSE(none.accepted.valid);
    EXPECT_EQ(none.final_error, "missing");
}

#if defined(__linux__)
// Round trip against the library's own writer: every snapshot a real gate
// publishes on the success, trigger-failure, unowned-init-failure and manual
// paths parses as valid. A timeout path is not induced here.
namespace {

struct Published {
    std::mutex mu;
    std::vector<std::string> json;
    std::vector<std::thread> threads;
    ~Published() {
        for (auto& t : threads) t.join();
    }
};

gpufl::inject::GateHooks RoundTripHooks(Published& out, gpufl::inject::GateStep init) {
    gpufl::inject::GateHooks hooks;
    hooks.now_ns = gpufl::launcher::MonotonicNowNs;
    hooks.spawn = [&out](std::function<void()> body) {
        std::lock_guard lock(out.mu);
        out.threads.emplace_back(std::move(body));
        return true;
    };
    hooks.driver_barrier = [] { return gpufl::inject::GateStep{true, ""}; };
    hooks.initialize = [init] { return init; };
    hooks.publish = [&out](const gpufl::inject::GateSnapshot& s) {
        std::lock_guard lock(out.mu);
        out.json.push_back(gpufl::inject::GateStatusJson(s, "n1", 60));
    };
    return hooks;
}

}  // namespace

TEST_F(BootstrapStatusTest, EverySnapshotTheGatePublishesParsesAsValid) {
    using gpufl::inject::CaptureGate;
    using gpufl::inject::GateMode;
    using gpufl::inject::GateStep;
    const auto config = [](GateMode mode, int64_t wait_ms) {
        gpufl::inject::GateConfig c;
        c.mode = mode;
        c.wait_ns = wait_ms * 1'000'000LL;
        c.protocol = "2";
        return c;
    };
    std::vector<std::string> all;
    const auto collect = [&all](Published& p) {
        for (auto& t : p.threads) t.join();
        p.threads.clear();
        all.insert(all.end(), p.json.begin(), p.json.end());
    };
    {
        Published p;  // gate owner, success
        CaptureGate gate(config(GateMode::Automatic, 5'000), RoundTripHooks(p, {true, ""}));
        gate.Enter([&] { gate.OnInjectionCallback(); return GateStep{true, ""}; });
        collect(p);
    }
    {
        Published p;  // gate owner, trigger failure without a callback
        CaptureGate gate(config(GateMode::Automatic, 5'000), RoundTripHooks(p, {true, ""}));
        gate.Enter([] { return GateStep{false, "runtime_init_failed:35"}; });
        collect(p);
    }
    {
        Published p;  // unowned first touch, init failure
        CaptureGate gate(config(GateMode::Automatic, 5'000),
                         RoundTripHooks(p, {false, "init_failed"}));
        gate.OnInjectionCallback();
        collect(p);
    }
    {
        Published p;  // manual handshake after a counted callback
        CaptureGate gate(config(GateMode::Manual, 5'000), RoundTripHooks(p, {true, ""}));
        gate.OnInjectionCallback();
        EXPECT_EQ(gate.ManualHandshake(nullptr), 1);
        collect(p);
    }
    ASSERT_GT(all.size(), 8u);
    for (const std::string& json : all) {
        Write(60, json);
        const BootstrapStatus s = Read(60);
        EXPECT_TRUE(s.valid) << s.error << ": " << json;
    }
}
#endif
